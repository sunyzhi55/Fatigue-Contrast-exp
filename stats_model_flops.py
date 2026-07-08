"""
统计所有疲劳检测对比实验模型的参数量、MACs、FLOPs

使用 calflops 库计算，输出到 CSV 文件并打印表格。

使用方法:
    pip install calflops
    python stats_model_flops.py

输出:
    - 终端打印格式化表格
    - model_stats.csv (同目录)
"""
import sys
import csv
import importlib.util
import types
from pathlib import Path

import torch
import torch.nn as nn

# ========================================================================== #
#  配置                                                                        #
# ========================================================================== #

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_CSV = PROJECT_ROOT / "model_stats.csv"

# 统一输入规格 (ADF 三通道)
C = 3           # 通道数 (ADF: 空间漂移 + 一阶差分 + 滑动均值)
W = 256         # 序列长度 (window_size)
FLAT = C * W    # 展平维度 (768)
B = 1           # batch_size (单样本统计)

# ========================================================================== #
#  工具函数                                                                    #
# ========================================================================== #

def _load(name):
    """从 models/ 直接加载模块，绕过 __init__.py 的重量级导入"""
    path = MODELS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_mlda():
    """确保 mlda_model 已加载 (DANN/DeepCORAL 依赖它)"""
    if "models.mlda_model" not in sys.modules:
        # 预注册 stub 避免 torchvision/timm 依赖
        stub = types.ModuleType("models")
        stub.__path__ = [str(MODELS_DIR)]
        sys.modules.setdefault("models", stub)
        _load("mlda_model")


class _FewShotWrapper(nn.Module):
    """将 ProtoNet/RelationNet 的 predict() 包装为标准 forward()
    因为这两个模型的 forward() 是训练模式 (需要 labels)，
    而推理路径在 predict() 方法中。
    """

    def __init__(self, model, n_way=2, k_shot=5, n_query=10, input_size=768):
        super().__init__()
        self.model = model
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.input_size = input_size

    def forward(self, query):
        n_support = self.n_way * self.k_shot
        support = torch.zeros(n_support, self.input_size, device=query.device)
        labels = torch.arange(self.n_way, device=query.device).repeat_interleave(self.k_shot)
        probs, preds = self.model.predict(support, labels, query)
        return probs


# ========================================================================== #
#  构建所有模型                                                                #
# ========================================================================== #

def build_all_models():
    """构建所有对比实验模型，返回 {name: (model, args_or_shape, note)}"""
    device = torch.device("cpu")

    # 预加载依赖
    _ensure_mlda()

    models = {}

    # ---- 时序基线 ----
    try:
        mod = _load("lstm")
        models["LSTM"] = (
            mod.LSTMClassifier(
                input_size=C, hidden_size=64, num_layers=2,
                num_classes=2, dropout=0.3, bidirectional=False,
            ).to(device),
            (torch.randn(B, W, C),),
            "(B,W,C)",
        )
    except Exception as e:
        print(f"  [SKIP] LSTM: {e}")

    try:
        mod = _load("transformer_encoder")
        models["Transformer"] = (
            mod.TransformerEncoderClassifier(
                input_size=C, d_model=64, nhead=4, num_layers=2,
                dim_feedforward=128, num_classes=2, dropout=0.3, max_seq_len=W,
            ).to(device),
            (torch.randn(B, W, C),),
            "(B,W,C)",
        )
    except Exception as e:
        print(f"  [SKIP] Transformer: {e}")

    try:
        mod = _load("mamba_model")
        models["Mamba"] = (
            mod.MambaModel(
                input_size=C, d_model=64, n_layer=2, d_conv=4,
                d_state=16, expand=2, num_classes=2, dropout=0.3,
            ).to(device),
            (torch.randn(B, W, C),),
            "(B,W,C)",
        )
    except Exception as e:
        print(f"  [SKIP] Mamba (needs CUDA): {e}")

    try:
        mod = _load("timesnet_model")
        models["TimesNet"] = (
            mod.TimesNetClassifier(
                input_size=C, seq_len=W, d_model=32, d_ff=64,
                num_kernels=6, top_k=3, e_layers=2,
                num_classes=2, dropout=0.1,
            ).to(device),
            (torch.randn(B, W, C),),
            "(B,W,C)",
        )
    except Exception as e:
        print(f"  [SKIP] TimesNet: {e}")

    # ---- 小样本学习 ----
    try:
        mod = _load("protonet")
        pn = mod.ProtoNet(
            input_size=FLAT, hidden_size=64, embedding_size=32,
            num_classes=2, dropout=0.2,
        ).to(device)
        models["ProtoNet"] = (
            _FewShotWrapper(pn, input_size=FLAT).to(device),
            (torch.randn(B * 2 * 10, FLAT),),
            f"(B*{2*10},{FLAT})",
        )
    except Exception as e:
        print(f"  [SKIP] ProtoNet: {e}")

    try:
        mod = _load("relationnet")
        rn = mod.RelationNet(
            input_size=FLAT, hidden_size=64, embedding_size=32,
            relation_size=16, num_classes=2, dropout=0.2,
        ).to(device)
        models["RelationNet"] = (
            _FewShotWrapper(rn, input_size=FLAT).to(device),
            (torch.randn(B * 2 * 10, FLAT),),
            f"(B*{2*10},{FLAT})",
        )
    except Exception as e:
        print(f"  [SKIP] RelationNet: {e}")

    # ---- 域适应 ----
    try:
        _load("mlda_model")
        mod_mlda = sys.modules["mlda_model"]
        models["MLDA"] = (
            mod_mlda.MLDAModel(
                input_dim=FLAT, num_classes=2, feat_dim=32, dropout=0.05,
            ).to(device),
            (torch.randn(B, FLAT),),
            f"(B,{FLAT})",
        )
    except Exception as e:
        print(f"  [SKIP] MLDA: {e}")

    try:
        mod = _load("daeevit_model")
        models["DAEEGViT"] = (
            mod.DAEEGViTModel(
                seq_len=W, patch_size=32, in_channels=C, num_classes=2,
                embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0,
                qkv_bias=True, drop_ratio=0.1, attn_drop_ratio=0.0,
                drop_path_ratio=0.1, mbconv_expand_ratio=4, mbconv_se_ratio=0.25,
            ).to(device),
            (torch.randn(B, C, W),),
            "(B,C,W)",
        )
    except Exception as e:
        print(f"  [SKIP] DAEEGViT: {e}")

    try:
        mod = _load("lamsda_model")
        lamsda_model = mod.LAMSDAModel(
            in_channels=C, seq_len=W, num_classes=2,
            num_sources=5, feature_dim=64, ds_hidden_dim=256,
        ).to(device)
        models["LA-MSDA"] = (
            lamsda_model,
            (torch.randn(B, C, W), torch.tensor(0)),  # x, domain_idx
            "(B,C,W)",
        )
    except Exception as e:
        print(f"  [SKIP] LA-MSDA: {e}")

    try:
        mod = _load("dann_model")
        models["DANN"] = (
            mod.DANNModel(
                input_dim=FLAT, num_classes=2, feat_dim=32,
                dropout=0.05, domain_hidden=1024,
            ).to(device),
            (torch.randn(B, FLAT),),
            f"(B,{FLAT})",
        )
    except Exception as e:
        print(f"  [SKIP] DANN: {e}")

    try:
        mod = _load("deepcoral_model")
        models["DeepCORAL"] = (
            mod.DeepCORALModel(
                input_dim=FLAT, num_classes=2, feat_dim=32, dropout=0.05,
            ).to(device),
            (torch.randn(B, FLAT),),
            f"(B,{FLAT})",
        )
    except Exception as e:
        print(f"  [SKIP] DeepCORAL: {e}")

    # ---- 域泛化 ----
    try:
        mod = _load("interpcnn_model")
        models["InterpretableCNN"] = (
            mod.InterpretableCNN(
                in_channels=C, seq_len=W, num_classes=2,
                n_filters=16, depth_multiplier=2, kernel_size=64, dropout=0.0,
            ).to(device),
            (torch.randn(B, C, W),),
            "(B,C,W)",
        )
    except Exception as e:
        print(f"  [SKIP] InterpretableCNN: {e}")

    try:
        mod = _load("afmcir_model")
        models["AFM-CIR"] = (
            mod.AFMCIRNet(
                in_channels=C, seq_len=W, num_classes=2,
                feat_dim=64, dropout=0.1, adv_hidden=64, kappa=0.8,
            ).to(device),
            (torch.randn(B, C, W),),
            "(B,C,W)",
        )
    except Exception as e:
        print(f"  [SKIP] AFM-CIR: {e}")

    return models


# ========================================================================== #
#  主流程                                                                      #
# ========================================================================== #

def main():
    from calflops import calculate_flops

    print("=" * 72)
    print("  疲劳检测对比实验 — 模型参数量 / MACs / FLOPs 统计")
    print(f"  输入规格: C={C}, W={W}, FLAT={FLAT}")
    print("=" * 72)

    models = build_all_models()
    results = []

    for name, (model, args, note) in models.items():
        model.eval()
        try:
            flops_str, macs_str, params_str = calculate_flops(
                model=model,
                args=list(args),
                output_as_string=True,
                output_precision=2,
                print_results=False,
                print_detailed=False,
            )
            results.append({
                "Model": name,
                "Input": note,
                "Params": params_str,
                "MACs": macs_str,
                "FLOPs": flops_str,
            })
            print(f"  [OK]   {name:20s} | Params: {params_str:>12s} | "
                  f"MACs: {macs_str:>12s} | FLOPs: {flops_str:>12s}")
        except Exception as e:
            results.append({
                "Model": name,
                "Input": note,
                "Params": "ERROR",
                "MACs": "ERROR",
                "FLOPs": str(e)[:60],
            })
            print(f"  [FAIL] {name:20s} | Error: {e}")

    # ---- 打印汇总表格 ----
    print("\n" + "=" * 72)
    header = f"{'Model':<20s} {'Input':<14s} {'Params':>12s} {'MACs':>12s} {'FLOPs':>12s}"
    print(header)
    print("-" * 72)
    for r in results:
        print(f"{r['Model']:<20s} {r['Input']:<14s} "
              f"{r['Params']:>12s} {r['MACs']:>12s} {r['FLOPs']:>12s}")
    print("=" * 72)

    # ---- 写入 CSV ----
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Model", "Input", "Params", "MACs", "FLOPs"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[OK] CSV saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
