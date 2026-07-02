"""
DANN 和 DeepCORAL 域适应方法的单元测试

测试覆盖:
    1. GRL (梯度反转层) 前向/反向传播
    2. DANN 模型推理/训练模式
    3. DeepCORAL 模型推理/训练模式
    4. CORAL 损失正确性
    5. 模型工厂 (get_model) 集成
    6. 配置加载 (两个新实验配置)
    7. 端到端训练步骤 (mini-batch)

运行:
    cd D:\\code\\SelfNet\\Fatigue-Contrast-exp
    python -m pytest tests/test_dann_deepcoral.py -v
"""
import sys
import os
import importlib
import pytest
import torch
import torch.nn as nn
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_module_direct(module_path: str, module_name: str):
    """直接导入指定模块，绕过 __init__.py 的全量导入链。

    这样可避免 models/__init__.py 中 timm 等第三方依赖的导入失败。
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- 预加载所需模块 (绕过 models/__init__.py) ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 创建空的 models 包占位符，防止 Python 加载真正的 __init__.py
import types as _types
_models_pkg = _types.ModuleType("models")
_models_pkg.__path__ = [os.path.join(_project_root, "models")]
_models_pkg.__package__ = "models"
sys.modules["models"] = _models_pkg

# 先确保 mlda_model 已加载 (dann_model 和 deepcoral_model 依赖它)
_mlda_mod = _import_module_direct(
    os.path.join(_project_root, "models", "mlda_model.py"), "models.mlda_model"
)

# 加载 DANN 和 DeepCORAL 模型
_dann_mod = _import_module_direct(
    os.path.join(_project_root, "models", "dann_model.py"), "models.dann_model"
)
_coral_mod = _import_module_direct(
    os.path.join(_project_root, "models", "deepcoral_model.py"), "models.deepcoral_model"
)


# ========================================================================== #
#  GRL (梯度反转层) 测试                                                      #
# ========================================================================== #

class TestGRL:
    """测试梯度反转层 (Gradient Reversal Layer)"""

    def test_grl_forward_training(self):
        """训练模式下 GRL 应为恒等映射"""
        from models.dann_model import GRL
        grl = GRL(alpha=1.0)
        grl.train()
        x = torch.randn(4, 32)
        out = grl(x)
        assert torch.allclose(out, x), "GRL 前向传播应为恒等映射"

    def test_grl_forward_eval(self):
        """评估模式下 GRL 应为恒等映射 (无反转)"""
        from models.dann_model import GRL
        grl = GRL(alpha=1.0)
        grl.eval()
        x = torch.randn(4, 32)
        out = grl(x)
        assert torch.allclose(out, x), "评估模式 GRL 应为恒等映射"

    def test_grl_backward_reversal(self):
        """反向传播时梯度应被反转"""
        from models.dann_model import GRL
        grl = GRL(alpha=1.0)
        grl.train()

        x = torch.randn(4, 32, requires_grad=True)
        out = grl(x)
        loss = out.sum()
        loss.backward()

        # 梯度应为 -1 (因为 alpha=1.0, 反转后 grad = -1 * ones)
        assert x.grad is not None, "梯度不应为 None"
        assert torch.all(x.grad < 0), f"梯度应全为负数, 实际: {x.grad[:2]}"

    def test_grl_alpha_scaling(self):
        """alpha 缩放应正确影响梯度幅度"""
        from models.dann_model import GRL
        grl = GRL(alpha=0.5)
        grl.train()

        x = torch.randn(4, 32, requires_grad=True)
        out = grl(x)
        loss = out.sum()
        loss.backward()

        # 梯度应为 -0.5
        assert torch.allclose(x.grad, torch.full_like(x.grad, -0.5), atol=1e-6)

    def test_grl_set_alpha(self):
        """set_alpha 应正确更新 alpha 值"""
        from models.dann_model import GRL
        grl = GRL(alpha=1.0)
        grl.set_alpha(0.7)
        assert grl.alpha == 0.7

    def test_grl_schedule_simulation(self):
        """模拟 DANN 论文 Eq.9 的调度: p 从 0→1, λ 从 ~0→~1"""
        from models.dann_model import GRL
        grl = GRL()
        gamma = 10.0
        lambdas = []
        for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
            lam = 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
            lambdas.append(lam)

        # λ 应单调递增
        for i in range(len(lambdas) - 1):
            assert lambdas[i] < lambdas[i + 1], \
                f"λ 调度应单调递增: λ[{i}]={lambdas[i]:.4f} >= λ[{i+1}]={lambdas[i+1]:.4f}"

        # λ(0) ≈ 0, λ(1) ≈ 1
        assert abs(lambdas[0]) < 0.01, f"λ(0) 应 ≈ 0, 实际: {lambdas[0]}"
        assert abs(lambdas[-1] - 1.0) < 0.01, f"λ(1) 应 ≈ 1, 实际: {lambdas[-1]}"


# ========================================================================== #
#  DANN 模型测试                                                              #
# ========================================================================== #

class TestDANNModel:
    """测试 DANN 域对抗网络"""

    @pytest.fixture
    def dann_model(self):
        from models.dann_model import DANNModel
        return DANNModel(input_dim=768, num_classes=2, feat_dim=32, dropout=0.05)

    def test_dann_inference_mode(self, dann_model):
        """推理模式: forward(data, None) 应返回 (features, logits)"""
        dann_model.eval()
        x = torch.randn(8, 768)
        with torch.no_grad():
            feature, logits = dann_model(x, None)

        assert feature.shape == (8, 32), f"特征形状错误: {feature.shape}"
        assert logits.shape == (8, 2), f"输出形状错误: {logits.shape}"

    def test_dann_training_mode(self, dann_model):
        """训练模式: forward(src, tar) 应返回 (features, logits, domain_logits)"""
        dann_model.train()
        src = torch.randn(8, 768)
        tar = torch.randn(8, 768)
        feature, logits, domain_logits = dann_model(src, tar)

        assert feature.shape == (8, 32), f"源域特征形状错误: {feature.shape}"
        assert logits.shape == (8, 2), f"分类输出形状错误: {logits.shape}"
        # 域分类器输出: (src_batch + tar_batch, 1)
        assert domain_logits.shape == (16, 1), \
            f"域分类输出形状错误: {domain_logits.shape}"

    def test_dann_gradient_flow(self, dann_model):
        """验证梯度可以通过 GRL 反向传播到编码器"""
        dann_model.train()
        src = torch.randn(8, 768)
        tar = torch.randn(8, 768)
        feature, logits, domain_logits = dann_model(src, tar)

        # 模拟 DANN 损失
        src_label = torch.zeros(8, dtype=torch.long)
        cls_loss = nn.CrossEntropyLoss()(logits, src_label)
        domain_label = torch.cat([torch.zeros(8, 1), torch.ones(8, 1)])
        domain_loss = nn.BCEWithLogitsLoss()(domain_logits, domain_label)

        total_loss = cls_loss + domain_loss
        total_loss.backward()

        # 检查编码器参数有梯度
        has_grad = False
        for name, param in dann_model.encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "编码器参数应有梯度"

    def test_dann_output_shapes_batch_sizes(self, dann_model):
        """测试不同 batch size 下的输出形状 (BatchNorm 要求 bs>=2)"""
        dann_model.train()
        for bs in [2, 4, 16, 64]:
            src = torch.randn(bs, 768)
            tar = torch.randn(bs, 768)
            feature, logits, domain_logits = dann_model(src, tar)
            assert feature.shape == (bs, 32)
            assert logits.shape == (bs, 2)
            assert domain_logits.shape == (2 * bs, 1)

    def test_dann_model_factory(self):
        """验证 get_model 工厂能正确创建 DANN 模型 (需要 torchvision)"""
        try:
            from models.get_model import get_model
        except (ImportError, ModuleNotFoundError):
            pytest.skip("torchvision 未安装，跳过工厂测试")
        model = get_model('dann', 2, None, 'cpu',
                          input_size=768, feat_dim=32, dropout=0.05,
                          domain_hidden=1024)
        assert isinstance(model, torch.nn.Module)
        x = torch.randn(4, 768)
        model.eval()
        with torch.no_grad():
            feat, logits = model(x, None)
        assert feat.shape == (4, 32)
        assert logits.shape == (4, 2)


# ========================================================================== #
#  DeepCORAL 模型测试                                                         #
# ========================================================================== #

class TestDeepCORALModel:
    """测试 DeepCORAL 域适应模型"""

    @pytest.fixture
    def coral_model(self):
        from models.deepcoral_model import DeepCORALModel
        return DeepCORALModel(input_dim=768, num_classes=2, feat_dim=32, dropout=0.05)

    def test_coral_inference_mode(self, coral_model):
        """推理模式: forward(data, None) 应返回 (features, logits)"""
        coral_model.eval()
        x = torch.randn(8, 768)
        with torch.no_grad():
            feature, logits = coral_model(x, None)

        assert feature.shape == (8, 32), f"特征形状错误: {feature.shape}"
        assert logits.shape == (8, 2), f"输出形状错误: {logits.shape}"

    def test_coral_training_mode(self, coral_model):
        """训练模式: forward(src, tar) 应返回 (src_feat, tar_feat, src_logits)"""
        coral_model.train()
        src = torch.randn(8, 768)
        tar = torch.randn(8, 768)
        src_feat, tar_feat, src_logits = coral_model(src, tar)

        assert src_feat.shape == (8, 32), f"源域特征形状错误: {src_feat.shape}"
        assert tar_feat.shape == (8, 32), f"目标域特征形状错误: {tar_feat.shape}"
        assert src_logits.shape == (8, 2), f"分类输出形状错误: {src_logits.shape}"

    def test_coral_gradient_flow(self, coral_model):
        """验证 CORAL 损失梯度可以回传到编码器"""
        from models.deepcoral_model import coral_loss
        coral_model.train()

        src = torch.randn(8, 768)
        tar = torch.randn(8, 768)
        src_feat, tar_feat, src_logits = coral_model(src, tar)

        cls_loss = nn.CrossEntropyLoss()(src_logits, torch.zeros(8, dtype=torch.long))
        coral = coral_loss(src_feat, tar_feat)
        total_loss = cls_loss + coral
        total_loss.backward()

        has_grad = False
        for name, param in coral_model.encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "编码器参数应有梯度"

    def test_coral_model_factory(self):
        """验证 get_model 工厂能正确创建 DeepCORAL 模型 (需要 torchvision)"""
        try:
            from models.get_model import get_model
        except (ImportError, ModuleNotFoundError):
            pytest.skip("torchvision 未安装，跳过工厂测试")
        model = get_model('deepcoral', 2, None, 'cpu',
                          input_size=768, feat_dim=32, dropout=0.05)
        assert isinstance(model, torch.nn.Module)
        x = torch.randn(4, 768)
        model.eval()
        with torch.no_grad():
            feat, logits = model(x, None)
        assert feat.shape == (4, 32)
        assert logits.shape == (4, 2)


# ========================================================================== #
#  CORAL 损失测试                                                             #
# ========================================================================== #

class TestCORALLoss:
    """测试 CORAL 损失函数"""

    def test_coral_same_distribution(self):
        """相同分布的特征 CORAL 损失应接近 0"""
        from models.deepcoral_model import coral_loss
        torch.manual_seed(42)
        x = torch.randn(100, 32)
        loss = coral_loss(x, x + torch.randn_like(x) * 0.01)
        assert loss.item() < 0.1, f"相同分布 CORAL 损失应接近 0, 实际: {loss.item():.6f}"

    def test_coral_different_distribution(self):
        """不同分布的特征 CORAL 损失应大于 0"""
        from models.deepcoral_model import coral_loss
        torch.manual_seed(42)
        src = torch.randn(100, 32)  # 均值 0
        tar = torch.randn(100, 32) + 5.0  # 均值 5 (协方差结构不同)
        loss = coral_loss(src, tar)
        assert loss.item() > 0, f"不同分布 CORAL 损失应 > 0, 实际: {loss.item():.6f}"

    def test_coral_nonnegative(self):
        """CORAL 损失应始终非负"""
        from models.deepcoral_model import coral_loss
        for _ in range(10):
            src = torch.randn(32, 16)
            tar = torch.randn(32, 16)
            loss = coral_loss(src, tar)
            assert loss.item() >= 0, f"CORAL 损失应非负, 实际: {loss.item():.6f}"

    def test_coral_differentiable(self):
        """CORAL 损失应可微"""
        from models.deepcoral_model import coral_loss
        src = torch.randn(16, 32, requires_grad=True)
        tar = torch.randn(16, 32)
        loss = coral_loss(src, tar)
        loss.backward()
        assert src.grad is not None, "CORAL 损失应可微"

    def test_coral_symmetric(self):
        """CORAL(src, tar) 应等于 CORAL(tar, src)"""
        from models.deepcoral_model import coral_loss
        torch.manual_seed(42)
        src = torch.randn(32, 16)
        tar = torch.randn(32, 16)
        loss1 = coral_loss(src, tar)
        loss2 = coral_loss(tar, src)
        assert torch.allclose(loss1, loss2, atol=1e-5), \
            f"CORAL 应是对称的: {loss1.item():.6f} vs {loss2.item():.6f}"

    def test_coral_batch_sizes(self):
        """测试不同 batch size 和特征维度"""
        from models.deepcoral_model import coral_loss
        for n, m, d in [(4, 4, 8), (16, 32, 64), (1, 1, 32)]:
            src = torch.randn(n, d)
            tar = torch.randn(m, d)
            loss = coral_loss(src, tar)
            assert not torch.isnan(loss), f"CORAL 不应为 NaN (n={n}, m={m}, d={d})"
            assert loss.item() >= 0, f"CORAL 应非负 (n={n}, m={m}, d={d})"


# ========================================================================== #
#  配置加载测试                                                               #
# ========================================================================== #

class TestConfigLoading:
    """测试新配置能否正确加载"""

    def test_dann_config_exists(self):
        """DANN 实验配置应存在"""
        from configs.fatigue_domain_adapt_baselines import fatigue_da_experiments
        assert "Fatigue_DANN_baseline" in fatigue_da_experiments, \
            "Fatigue_DANN_baseline 配置未找到"

    def test_deepcoral_config_exists(self):
        """DeepCORAL 实验配置应存在"""
        from configs.fatigue_domain_adapt_baselines import fatigue_da_experiments
        assert "Fatigue_DeepCORAL_baseline" in fatigue_da_experiments, \
            "Fatigue_DeepCORAL_baseline 配置未找到"

    def test_dann_config_fields(self):
        """DANN 配置应包含必要字段"""
        from configs.fatigue_domain_adapt_baselines import fatigue_da_experiments
        cfg = fatigue_da_experiments["Fatigue_DANN_baseline"]
        required_keys = [
            "model_name", "num_classes", "training_type", "optimizer_name",
            "batch_size", "epochs", "lr", "weight_decay", "feat_dim",
            "dann_gamma", "data_dir", "window_size", "use_adf",
        ]
        for key in required_keys:
            assert key in cfg, f"DANN 配置缺少字段: {key}"
        assert cfg["model_name"] == "dann"
        assert cfg["training_type"] == "dann"

    def test_deepcoral_config_fields(self):
        """DeepCORAL 配置应包含必要字段"""
        from configs.fatigue_domain_adapt_baselines import fatigue_da_experiments
        cfg = fatigue_da_experiments["Fatigue_DeepCORAL_baseline"]
        required_keys = [
            "model_name", "num_classes", "training_type", "optimizer_name",
            "batch_size", "epochs", "lr", "weight_decay", "feat_dim",
            "coral_weight", "data_dir", "window_size", "use_adf",
        ]
        for key in required_keys:
            assert key in cfg, f"DeepCORAL 配置缺少字段: {key}"
        assert cfg["model_name"] == "deepcoral"
        assert cfg["training_type"] == "deepcoral"

    def test_all_configs_mergeable(self):
        """所有配置字典应可合并且无键冲突"""
        from configs.fatigue_temporal_baselines import fatigue_temporal_experiments
        from configs.fatigue_fewshot_baselines import fatigue_fewshot_experiments
        from configs.fatigue_domain_adapt_baselines import fatigue_da_experiments

        all_exp = {}
        all_exp.update(fatigue_temporal_experiments)
        all_exp.update(fatigue_fewshot_experiments)
        all_exp.update(fatigue_da_experiments)

        expected_names = [
            "Fatigue_LSTM_baseline", "Fatigue_Transformer_baseline",
            "Fatigue_ProtoNet_baseline", "Fatigue_RelationNet_baseline",
            "Fatigue_MLDA_baseline", "Fatigue_DAEEGViT_baseline",
            "Fatigue_LA_MSDA_baseline",
            "Fatigue_DANN_baseline", "Fatigue_DeepCORAL_baseline",
        ]
        for name in expected_names:
            assert name in all_exp, f"实验 {name} 未在合并后的配置中找到"


# ========================================================================== #
#  端到端训练步骤测试                                                          #
# ========================================================================== #

class TestEndToEnd:
    """端到端 mini-batch 训练步骤测试"""

    def test_dann_training_step(self):
        """DANN 应能完成一个完整的训练步骤"""
        from models.dann_model import DANNModel

        torch.manual_seed(42)
        device = 'cpu'
        model = DANNModel(input_dim=768, num_classes=2, feat_dim=32, dropout=0.05).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.BCEWithLogitsLoss()

        model.train()
        src = torch.randn(16, 768).to(device)
        tar = torch.randn(16, 768).to(device)
        src_label = torch.randint(0, 2, (16,)).to(device)

        # GRL 调度
        model.grl.set_alpha(0.5)

        # 前向
        src_feat, src_logits, domain_logits = model(src, tar)

        # 损失
        cls_loss = criterion(src_logits, src_label)
        domain_labels = torch.cat([torch.zeros(16, 1), torch.ones(16, 1)]).to(device)
        domain_loss = domain_criterion(domain_logits, domain_labels)
        total_loss = cls_loss + 0.5 * domain_loss

        # 反向
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        assert total_loss.item() > 0, "损失应为正数"
        assert not torch.isnan(total_loss), "损失不应为 NaN"

    def test_deepcoral_training_step(self):
        """DeepCORAL 应能完成一个完整的训练步骤"""
        from models.deepcoral_model import DeepCORALModel, coral_loss

        torch.manual_seed(42)
        device = 'cpu'
        model = DeepCORALModel(input_dim=768, num_classes=2, feat_dim=32, dropout=0.05).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        src = torch.randn(16, 768).to(device)
        tar = torch.randn(16, 768).to(device)
        src_label = torch.randint(0, 2, (16,)).to(device)

        # 前向
        src_feat, tar_feat, src_logits = model(src, tar)

        # 损失
        cls_loss = criterion(src_logits, src_label)
        coral = coral_loss(src_feat, tar_feat)
        total_loss = cls_loss + 1.0 * coral

        # 反向
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        assert total_loss.item() > 0, "损失应为正数"
        assert not torch.isnan(total_loss), "损失不应为 NaN"

    def test_dann_eval_after_training(self):
        """DANN 训练后切换到评估模式应正常工作"""
        from models.dann_model import DANNModel

        torch.manual_seed(42)
        model = DANNModel(input_dim=768, num_classes=2, feat_dim=32)

        # 训练一步
        model.train()
        src = torch.randn(8, 768)
        tar = torch.randn(8, 768)
        model.grl.set_alpha(0.5)
        _, _, domain_logits = model(src, tar)
        loss = nn.BCEWithLogitsLoss()(domain_logits,
                                       torch.cat([torch.zeros(8, 1), torch.ones(8, 1)]))
        loss.backward()

        # 切换到评估
        model.eval()
        x = torch.randn(8, 768)
        with torch.no_grad():
            feat, logits = model(x, None)
        assert feat.shape == (8, 32)
        assert logits.shape == (8, 2)
        assert not torch.isnan(logits).any(), "评估模式输出不应为 NaN"

    def test_deepcoral_eval_after_training(self):
        """DeepCORAL 训练后切换到评估模式应正常工作"""
        from models.deepcoral_model import DeepCORALModel, coral_loss

        torch.manual_seed(42)
        model = DeepCORALModel(input_dim=768, num_classes=2, feat_dim=32)

        # 训练一步
        model.train()
        src = torch.randn(8, 768)
        tar = torch.randn(8, 768)
        src_feat, tar_feat, logits = model(src, tar)
        loss = nn.CrossEntropyLoss()(logits, torch.zeros(8, dtype=torch.long)) + \
               coral_loss(src_feat, tar_feat)
        loss.backward()

        # 切换到评估
        model.eval()
        x = torch.randn(8, 768)
        with torch.no_grad():
            feat, logits = model(x, None)
        assert feat.shape == (8, 32)
        assert logits.shape == (8, 2)
        assert not torch.isnan(logits).any(), "评估模式输出不应为 NaN"

    def test_dann_multi_epoch_convergence(self):
        """DANN 多轮训练损失应递减 (基本收敛性测试)"""
        from models.dann_model import DANNModel

        torch.manual_seed(42)
        model = DANNModel(input_dim=768, num_classes=2, feat_dim=32, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.BCEWithLogitsLoss()

        src = torch.randn(32, 768)
        tar = torch.randn(32, 768)
        src_label = torch.randint(0, 2, (32,))

        losses = []
        model.train()
        for _ in range(20):
            model.grl.set_alpha(0.5)
            src_feat, src_logits, domain_logits = model(src, tar)
            cls_loss = criterion(src_logits, src_label)
            domain_labels = torch.cat([torch.zeros(32, 1), torch.ones(32, 1)])
            domain_loss = domain_criterion(domain_logits, domain_labels)
            total_loss = cls_loss + 0.5 * domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        # 最后 5 轮平均损失应小于前 5 轮
        early_avg = np.mean(losses[:5])
        late_avg = np.mean(losses[-5:])
        assert late_avg < early_avg, \
            f"损失应递减: 前5轮均值={early_avg:.4f}, 后5轮均值={late_avg:.4f}"

    def test_deepcoral_multi_epoch_convergence(self):
        """DeepCORAL 多轮训练损失应递减 (基本收敛性测试)"""
        from models.deepcoral_model import DeepCORALModel, coral_loss

        torch.manual_seed(42)
        model = DeepCORALModel(input_dim=768, num_classes=2, feat_dim=32, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        src = torch.randn(32, 768)
        tar = torch.randn(32, 768)
        src_label = torch.randint(0, 2, (32,))

        losses = []
        model.train()
        for _ in range(20):
            src_feat, tar_feat, src_logits = model(src, tar)
            cls_loss = criterion(src_logits, src_label)
            coral = coral_loss(src_feat, tar_feat)
            total_loss = cls_loss + coral

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        early_avg = np.mean(losses[:5])
        late_avg = np.mean(losses[-5:])
        assert late_avg < early_avg, \
            f"损失应递减: 前5轮均值={early_avg:.4f}, 后5轮均值={late_avg:.4f}"


# ========================================================================== #
#  模型参数统计测试                                                            #
# ========================================================================== #

class TestModelStats:
    """测试模型参数量合理性"""

    def test_dann_param_count(self):
        """DANN 模型应有合理的参数量"""
        from models.dann_model import DANNModel
        model = DANNModel(input_dim=768, num_classes=2, feat_dim=32, domain_hidden=1024)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params == trainable_params, "所有参数应可训练"
        assert total_params > 10000, f"参数量过少: {total_params}"
        assert total_params < 10_000_000, f"参数量过多: {total_params}"
        print(f"DANN 参数量: {total_params:,}")

    def test_deepcoral_param_count(self):
        """DeepCORAL 模型应有合理的参数量"""
        from models.deepcoral_model import DeepCORALModel
        model = DeepCORALModel(input_dim=768, num_classes=2, feat_dim=32)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params == trainable_params, "所有参数应可训练"
        assert total_params > 10000, f"参数量过少: {total_params}"
        # DeepCORAL 没有域分类器，应比 DANN 少
        print(f"DeepCORAL 参数量: {total_params:,}")

    def test_deepcoral_smaller_than_dann(self):
        """DeepCORAL 参数量应少于 DANN (因无域分类器)"""
        from models.dann_model import DANNModel
        from models.deepcoral_model import DeepCORALModel

        dann = DANNModel(input_dim=768, num_classes=2, feat_dim=32, domain_hidden=1024)
        coral = DeepCORALModel(input_dim=768, num_classes=2, feat_dim=32)

        dann_params = sum(p.numel() for p in dann.parameters())
        coral_params = sum(p.numel() for p in coral.parameters())

        assert coral_params < dann_params, \
            f"DeepCORAL({coral_params:,}) 参数应少于 DANN({dann_params:,})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
