"""
InterpretableCNN 域泛化基线的单元测试

测试覆盖:
    1. 模型前向传播 (输出形状、log-probs 性质)
    2. 可分离卷积架构验证 (pointwise + depthwise)
    3. BatchNorm track_running_stats=False 行为
    4. predict_proba 方法
    5. 端到端训练步骤
    6. 多轮训练收敛性
    7. 配置加载
    8. 模型参数量统计

运行:
    cd D:\\code\\SelfNet\\Fatigue-Contrast-exp
    python -m pytest tests/test_interpcnn.py -v
"""
import sys
import os
import importlib
import types as _types
import pytest
import torch
import torch.nn as nn
import numpy as np

# ---- 预加载模块 (绕过 models/__init__.py 的 timm 依赖) ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

_models_pkg = _types.ModuleType("models")
_models_pkg.__path__ = [os.path.join(_project_root, "models")]
_models_pkg.__package__ = "models"
sys.modules["models"] = _models_pkg


def _load(module_name, filename):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_project_root, "models", filename)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_interpcnn_mod = _load("models.interpcnn_model", "interpcnn_model.py")
InterpretableCNN = _interpcnn_mod.InterpretableCNN


# ========================================================================== #
#  前向传播测试                                                               #
# ========================================================================== #

class TestForward:

    def test_output_shape_default(self):
        """默认参数: (B, 3, 256) → (B, 2) log-probs"""
        model = InterpretableCNN(in_channels=3, seq_len=256, num_classes=2)
        x = torch.randn(8, 3, 256)
        out = model(x)
        assert out.shape == (8, 2), f"输出形状错误: {out.shape}"

    def test_output_is_log_probs(self):
        """输出应为 log-probabilities (≤0 且 exp 后求和≈1)"""
        model = InterpretableCNN(in_channels=3, seq_len=256)
        model.eval()
        x = torch.randn(16, 3, 256)
        with torch.no_grad():
            log_probs = model(x)
        assert (log_probs <= 0).all(), "log-probs 应全部 ≤ 0"
        probs = torch.exp(log_probs)
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"概率行和应≈1, 实际: {row_sums[:4]}"

    def test_different_batch_sizes(self):
        """测试不同 batch size"""
        model = InterpretableCNN(in_channels=3, seq_len=256)
        for bs in [2, 4, 32, 64]:
            x = torch.randn(bs, 3, 256)
            out = model(x)
            assert out.shape == (bs, 2), f"bs={bs} 输出形状错误: {out.shape}"

    def test_single_channel(self):
        """单通道输入 (use_adf=False)"""
        model = InterpretableCNN(in_channels=1, seq_len=256)
        x = torch.randn(8, 1, 256)
        out = model(x)
        assert out.shape == (8, 2)

    def test_different_seq_lens(self):
        """测试不同序列长度 (kernel_size 需要 ≤ seq_len)"""
        for seq_len in [128, 256, 512]:
            model = InterpretableCNN(in_channels=3, seq_len=seq_len, kernel_size=64)
            x = torch.randn(4, 3, seq_len)
            out = model(x)
            assert out.shape == (4, 2)

    def test_multiclass(self):
        """多分类 (num_classes=3)"""
        model = InterpretableCNN(in_channels=3, seq_len=256, num_classes=3)
        x = torch.randn(8, 3, 256)
        out = model(x)
        assert out.shape == (8, 3)


# ========================================================================== #
#  架构验证测试                                                               #
# ========================================================================== #

class TestArchitecture:

    def test_pointwise_depthwise_separation(self):
        """验证 pointwise 和 depthwise 卷积的分离结构"""
        model = InterpretableCNN(n_filters=16, depth_multiplier=2, kernel_size=64)
        # Pointwise: (C_in, N1, 1)
        assert model.pointwise.weight.shape == (16, 3, 1)
        # Depthwise: (N1*d, 1, K) with groups=N1
        assert model.depthwise.weight.shape == (32, 1, 64)
        assert model.depthwise.groups == 16

    def test_batchnorm_no_running_stats(self):
        """BatchNorm 应设置 track_running_stats=False"""
        model = InterpretableCNN()
        assert model.batchnorm.track_running_stats is False, \
            "BatchNorm 应 track_running_stats=False (匹配原论文)"

    def test_batchnorm_eval_uses_batch_stats(self):
        """eval 模式下 BatchNorm 仍使用 batch 统计量"""
        model = InterpretableCNN()
        model.eval()
        x1 = torch.randn(8, 3, 256)
        x2 = torch.randn(8, 3, 256) * 10  # 不同统计量
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        # 由于 BN 使用 batch 统计量，不同输入应产生不同输出
        assert not torch.allclose(out1, out2), "eval 模式 BN 应使用 batch 统计量"

    def test_num_features(self):
        """num_features 应等于 n_filters * depth_multiplier"""
        model = InterpretableCNN(n_filters=16, depth_multiplier=2)
        assert model.num_features == 32

    def test_fc_layer(self):
        """FC 层形状应正确"""
        model = InterpretableCNN(n_filters=16, depth_multiplier=2, num_classes=2)
        assert model.fc.in_features == 32
        assert model.fc.out_features == 2


# ========================================================================== #
#  predict_proba 测试                                                        #
# ========================================================================== #

class TestPredictProba:

    def test_predict_proba_sums_to_one(self):
        """predict_proba 输出概率行和应≈1"""
        model = InterpretableCNN()
        model.eval()
        x = torch.randn(8, 3, 256)
        with torch.no_grad():
            probs = model.predict_proba(x)
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_predict_proba_nonnegative(self):
        """概率应全部 ≥ 0"""
        model = InterpretableCNN()
        model.eval()
        x = torch.randn(8, 3, 256)
        with torch.no_grad():
            probs = model.predict_proba(x)
        assert (probs >= 0).all()


# ========================================================================== #
#  端到端训练测试                                                             #
# ========================================================================== #

class TestEndToEnd:

    def test_training_step(self):
        """完整训练步骤: 前向 → 损失 → 反向 → 优化"""
        torch.manual_seed(42)
        model = InterpretableCNN(in_channels=3, seq_len=256, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.NLLLoss()

        model.train()
        x = torch.randn(16, 3, 256)
        labels = torch.randint(0, 2, (16,))

        optimizer.zero_grad()
        log_probs = model(x)
        loss = criterion(log_probs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_convergence(self):
        """多轮训练损失应递减"""
        torch.manual_seed(42)
        model = InterpretableCNN(in_channels=3, seq_len=256, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.NLLLoss()

        x = torch.randn(32, 3, 256)
        labels = torch.randint(0, 2, (32,))

        losses = []
        model.train()
        for _ in range(30):
            optimizer.zero_grad()
            log_probs = model(x)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        early = np.mean(losses[:5])
        late = np.mean(losses[-5:])
        assert late < early, f"损失应递减: early={early:.4f}, late={late:.4f}"

    def test_eval_after_training(self):
        """训练后切换 eval 模式应正常工作"""
        model = InterpretableCNN(in_channels=3, seq_len=256)
        # 训练
        model.train()
        x = torch.randn(8, 3, 256)
        model(x).sum().backward()
        # eval
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(8, 3, 256))
        assert out.shape == (8, 2)
        assert not torch.isnan(out).any()


# ========================================================================== #
#  配置加载测试                                                               #
# ========================================================================== #

class TestConfig:

    def test_config_exists(self):
        """Fatigue_InterpCNN_baseline 配置应存在"""
        from configs.fatigue_temporal_baselines import fatigue_temporal_experiments
        assert "Fatigue_InterpCNN_baseline" in fatigue_temporal_experiments

    def test_config_fields(self):
        """配置应包含必要字段"""
        from configs.fatigue_temporal_baselines import fatigue_temporal_experiments
        cfg = fatigue_temporal_experiments["Fatigue_InterpCNN_baseline"]
        required = [
            "model_name", "num_classes", "training_type", "optimizer_name",
            "batch_size", "epochs", "lr", "n_filters", "depth_multiplier",
            "kernel_size", "data_dir", "window_size", "use_adf",
        ]
        for key in required:
            assert key in cfg, f"缺少字段: {key}"
        assert cfg["model_name"] == "interpcnn"
        assert cfg["training_type"] == "dg_interpcnn"

    def test_all_configs_mergeable(self):
        """所有配置应可合并"""
        from configs.fatigue_temporal_baselines import fatigue_temporal_experiments
        from configs.fatigue_fewshot_baselines import fatigue_fewshot_experiments
        from configs.fatigue_domain_adapt_baselines import fatigue_da_experiments
        all_exp = {}
        all_exp.update(fatigue_temporal_experiments)
        all_exp.update(fatigue_fewshot_experiments)
        all_exp.update(fatigue_da_experiments)
        assert "Fatigue_InterpCNN_baseline" in all_exp


# ========================================================================== #
#  模型参数统计                                                               #
# ========================================================================== #

class TestModelStats:

    def test_param_count(self):
        """参数量应在合理范围 (轻量级模型)"""
        model = InterpretableCNN(in_channels=3, seq_len=256)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total == trainable, "所有参数应可训练"
        assert total > 100, f"参数量过少: {total}"
        assert total < 100_000, f"参数量过多: {total}"
        print(f"InterpretableCNN 参数量: {total:,}")

    def test_lightweight(self):
        """应比 LSTM/Transformer 更轻量"""
        model = InterpretableCNN(in_channels=3, seq_len=256)
        total = sum(p.numel() for p in model.parameters())
        # 原论文模型仅 ~2.6K 参数 (30ch EEG)，我们 3ch 版本应更少
        print(f"InterpretableCNN (3ch) 参数量: {total:,}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
