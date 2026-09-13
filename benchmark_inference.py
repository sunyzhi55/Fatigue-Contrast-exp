"""
推理时长基准测试

给定一个 JSONL 文件（每行一帧，30fps），计算视频时长，
然后对每个模型测量单窗口推理耗时（100 轮取平均），输出到 CSV。

使用方法:
    python benchmark_inference.py <jsonl_file> [--device cuda] [--rounds 100] [--window_size 256]

输出:
    - 终端打印视频时长 + 各模型推理耗时
    - inference_time.csv (同目录)
"""
import sys
import csv
import json
import time
import importlib.util
import types
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ========================================================================== #
#  配置                                                                        #
# ========================================================================== #

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_CSV = PROJECT_ROOT / "inference_time.csv"

FPS = 30  # 帧率
NUM_CLASSES = 2


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
        stub = types.ModuleType("models")
        stub.__path__ = [str(MODELS_DIR)]
        sys.modules.setdefault("models", stub)
        _load("mlda_model")


def sliding_mean(values: np.ndarray, window_size: int) -> np.ndarray:
    """滑动窗口局部均值"""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("sliding_mean expects a 1-D array")
    if window_size <= 1:
        return values.copy()
    n = len(values)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    idx = np.arange(n)
    starts = np.maximum(0, idx - window_size + 1)
    counts = (idx - starts + 1).astype(np.float32)
    out = (cumsum[1:] - cumsum[starts]) / counts
    return out.astype(np.float32)


def compute_adf(drift: np.ndarray, local_mean_size: int = 16) -> np.ndarray:
    """由一维 drift 序列计算 ADF 三通道特征，返回 (T, 3)"""
    drift = np.asarray(drift, dtype=np.float32)
    if drift.size == 0:
        return drift.reshape(0, 3)
    diff = np.diff(drift, prepend=drift[:1]).astype(np.float32)
    local_mean = sliding_mean(drift, local_mean_size)
    return np.stack([drift, diff, local_mean], axis=-1).astype(np.float32)


# ========================================================================== #
#  JSONL 读取                                                                 #
# ========================================================================== #

def load_jsonl(jsonl_path: str, feature_name: str = "deviation_px_before_calibrate"):
    """读取 JSONL 文件，提取 drift 序列

    Args:
        jsonl_path: JSONL 文件路径
        feature_name: 提取的特征字段名

    Returns:
        drift: np.ndarray, shape (T,)
        num_frames: int
    """
    drifts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            val = frame.get(feature_name, None)
            if val is None:
                # 尝试 GAIPAT 字段
                val = frame.get("deviation_cm", 0.0)
            drifts.append(float(val))

    drift = np.array(drifts, dtype=np.float32)
    return drift, len(drifts)


def build_window(drift: np.ndarray, window_size: int = 256,
                 per_sample_norm: bool = True, local_mean_size: int = 16):
    """从 drift 序列构建一个 ADF 窗口

    Args:
        drift: (T,) 原始 drift 序列
        window_size: 窗口大小
        per_sample_norm: 是否做 Min-Max 归一化
        local_mean_size: sliding mean 窗口

    Returns:
        window_adf: (W, 3) ADF 三通道特征
    """
    # 截取前 window_size 帧（如果不足则补零）
    if len(drift) >= window_size:
        drift = drift[:window_size]
    else:
        drift = np.pad(drift, (0, window_size - len(drift)), mode="constant")

    # per-sample Min-Max normalization
    if per_sample_norm:
        d_min, d_max = drift.min(), drift.max()
        if d_max - d_min > 1e-8:
            drift = (drift - d_min) / (d_max - d_min)
        else:
            drift = np.zeros_like(drift)

    # ADF 3-channel: (W, 3)
    window_adf = compute_adf(drift, local_mean_size)
    return window_adf


# ========================================================================== #
#  推理耗时测量                                                                #
# ========================================================================== #

class _FewShotWrapper(nn.Module):
    """ProtoNet/RelationNet 推理包装：构造 dummy support set，调用 predict()"""

    def __init__(self, model, n_way=2, k_shot=5, input_size=768):
        super().__init__()
        self.model = model
        self.n_way = n_way
        self.k_shot = k_shot
        self.input_size = input_size
        # 预构造固定 support set
        self.register_buffer(
            "support",
            torch.zeros(n_way * k_shot, input_size),
        )
        self.register_buffer(
            "labels",
            torch.arange(n_way).repeat_interleave(k_shot),
        )

    def forward(self, query):
        probs, _ = self.model.predict(self.support, self.labels, query)
        return probs


def benchmark_model(model, input_tensor, forward_fn, device, rounds=100,
                    warmup=10):
    """测量模型推理耗时

    Args:
        model: nn.Module
        input_tensor: 输入 tensor
        forward_fn: callable(model, x) -> output
        device: torch.device
        rounds: 测量轮数
        warmup: 预热轮数

    Returns:
        (mean_ms, std_ms)
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = forward_fn(model, input_tensor)

    # GPU 同步
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(rounds):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = forward_fn(model, input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)  # ms

    timings = np.array(timings)
    return float(timings.mean()), float(timings.std())


# ========================================================================== #
#  构建所有模型                                                                #
# ========================================================================== #

def build_all_models(device, window_size=256):
    """构建所有对比实验模型

    Returns:
        {name: (model, forward_fn, input_shape_note)}
        forward_fn: callable(model, x) -> output
    """
    C = 3
    W = window_size
    FLAT = C * W

    _ensure_mlda()
    models = {}

    # ---- 标准 forward ----
    def _std_forward(model, x):
        return model(x)

    # ---- 推理模式 forward (tar_data=None) ----
    def _da_forward(model, x):
        out = model(x, None)
        # DANN/MLDA 返回 tuple (feat, logits)
        return out[-1] if isinstance(out, tuple) else out

    # ---- LA-MSDA ensemble ----
    def _lamsda_forward(model, x):
        return model.ensemble_predict(x)

    # ---- 时序基线 (B, W, C) ----
    temporal_configs = {
        "LSTM": ("lstm", lambda m: m.LSTMClassifier(
            input_size=C, hidden_size=64, num_layers=2,
            num_classes=NUM_CLASSES, dropout=0.3, bidirectional=False,
        )),
        "Transformer": ("transformer_encoder", lambda m: m.TransformerEncoderClassifier(
            input_size=C, d_model=64, nhead=4, num_layers=2,
            dim_feedforward=128, num_classes=NUM_CLASSES, dropout=0.3, max_seq_len=W,
        )),
        "TimesNet": ("timesnet_model", lambda m: m.TimesNetClassifier(
            input_size=C, seq_len=W, d_model=32, d_ff=64,
            num_kernels=6, top_k=3, e_layers=2,
            num_classes=NUM_CLASSES, dropout=0.1,
        )),
        "STAFNet": ("stafnet_model", lambda m: m.STAFNetClassifier(
            input_size=C, seq_len=W, num_classes=NUM_CLASSES,
            spectral_channels=8, num_bands=5, se_reduction=4,
            temporal_channels=16, gru_hidden=64, gru_layers=1,
            branch_output_dim=2, dropout=0.1,
        )),
    }

    for name, (mod_name, builder) in temporal_configs.items():
        try:
            mod = _load(mod_name)
            models[name] = (builder(mod).to(device), _std_forward, "(B,W,C)")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    # Mamba (可能需要 CUDA)
    try:
        mod = _load("mamba_model")
        models["Mamba"] = (
            mod.MambaModel(
                input_size=C, d_model=64, n_layer=2, d_conv=4,
                d_state=16, expand=2, num_classes=NUM_CLASSES, dropout=0.3,
            ).to(device),
            _std_forward, "(B,W,C)",
        )
    except Exception as e:
        print(f"  [SKIP] Mamba: {e}")

    # ---- 小样本学习 (B, FLAT) ----
    try:
        mod = _load("protonet")
        pn = mod.ProtoNet(
            input_size=FLAT, hidden_size=64, embedding_size=32,
            num_classes=NUM_CLASSES, dropout=0.2,
        ).to(device)
        models["ProtoNet"] = (
            _FewShotWrapper(pn, input_size=FLAT).to(device),
            _std_forward, f"(B,{FLAT})",
        )
    except Exception as e:
        print(f"  [SKIP] ProtoNet: {e}")

    try:
        mod = _load("relationnet")
        rn = mod.RelationNet(
            input_size=FLAT, hidden_size=64, embedding_size=32,
            relation_size=16, num_classes=NUM_CLASSES, dropout=0.2,
        ).to(device)
        models["RelationNet"] = (
            _FewShotWrapper(rn, input_size=FLAT).to(device),
            _std_forward, f"(B,{FLAT})",
        )
    except Exception as e:
        print(f"  [SKIP] RelationNet: {e}")

    # ---- 域适应 - 展平向量 (B, FLAT) ----
    try:
        _load("mlda_model")
        mod_mlda = sys.modules["mlda_model"]
        models["MLDA"] = (
            mod_mlda.MLDAModel(
                input_dim=FLAT, num_classes=NUM_CLASSES, feat_dim=32, dropout=0.05,
            ).to(device),
            _da_forward, f"(B,{FLAT})",
        )
    except Exception as e:
        print(f"  [SKIP] MLDA: {e}")

    try:
        mod = _load("dann_model")
        models["DANN"] = (
            mod.DANNModel(
                input_dim=FLAT, num_classes=NUM_CLASSES, feat_dim=32,
                dropout=0.05, domain_hidden=1024,
            ).to(device),
            _da_forward, f"(B,{FLAT})",
        )
    except Exception as e:
        print(f"  [SKIP] DANN: {e}")

    try:
        mod = _load("deepcoral_model")
        models["DeepCORAL"] = (
            mod.DeepCORALModel(
                input_dim=FLAT, num_classes=NUM_CLASSES, feat_dim=32, dropout=0.05,
            ).to(device),
            _da_forward, f"(B,{FLAT})",
        )
    except Exception as e:
        print(f"  [SKIP] DeepCORAL: {e}")

    # ---- 域适应/泛化 - 时序 (B, C, W) ----
    try:
        mod = _load("daeevit_model")
        models["DAEEGViT"] = (
            mod.DAEEGViTModel(
                seq_len=W, patch_size=32, in_channels=C, num_classes=NUM_CLASSES,
                embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0,
                qkv_bias=True, drop_ratio=0.1, attn_drop_ratio=0.0,
                drop_path_ratio=0.1, mbconv_expand_ratio=4, mbconv_se_ratio=0.25,
            ).to(device),
            _std_forward, "(B,C,W)",
        )
    except Exception as e:
        print(f"  [SKIP] DAEEGViT: {e}")

    try:
        mod = _load("lamsda_model")
        models["LA-MSDA"] = (
            mod.LAMSDAModel(
                in_channels=C, seq_len=W, num_classes=NUM_CLASSES,
                num_sources=5, feature_dim=64, ds_hidden_dim=256,
            ).to(device),
            _lamsda_forward, "(B,C,W)",
        )
    except Exception as e:
        print(f"  [SKIP] LA-MSDA: {e}")

    try:
        mod = _load("interpcnn_model")
        models["InterpretableCNN"] = (
            mod.InterpretableCNN(
                in_channels=C, seq_len=W, num_classes=NUM_CLASSES,
                n_filters=16, depth_multiplier=2, kernel_size=64, dropout=0.0,
            ).to(device),
            _std_forward, "(B,C,W)",
        )
    except Exception as e:
        print(f"  [SKIP] InterpretableCNN: {e}")

    try:
        mod = _load("afmcir_model")
        models["AFM-CIR"] = (
            mod.AFMCIRNet(
                in_channels=C, seq_len=W, num_classes=NUM_CLASSES,
                feat_dim=512, dropout=0.1, adv_hidden=256, kappa=0.8,
            ).to(device),
            _std_forward, "(B,C,W)",
        )
    except Exception as e:
        print(f"  [SKIP] AFM-CIR: {e}")

    return models


def prepare_inputs(window_adf: np.ndarray, device: torch.device):
    """为一个 ADF 窗口准备各模型的输入 tensor

    Args:
        window_adf: (W, 3) ADF 特征

    Returns:
        {shape_note: tensor}
    """
    W, C = window_adf.shape
    FLAT = W * C

    inputs = {}
    # (B, W, C) — 时序模型
    inputs["(B,W,C)"] = torch.from_numpy(
        window_adf[np.newaxis, :, :]  # (1, W, C)
    ).float().to(device)

    # (B, FLAT) — 展平模型
    inputs[f"(B,{FLAT})"] = torch.from_numpy(
        window_adf.reshape(1, -1)  # (1, FLAT)
    ).float().to(device)

    # (B, C, W) — (B,C,W) 模型
    inputs["(B,C,W)"] = torch.from_numpy(
        window_adf.T[np.newaxis, :, :]  # (1, C, W)
    ).float().to(device)

    return inputs


# ========================================================================== #
#  主流程                                                                      #
# ========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="推理时长基准测试")
    parser.add_argument("--jsonl_file", type=str, default="./test.jsonl", help="JSONL 数据文件路径")
    parser.add_argument("--device", type=str, default="cpu",
                        help="计算设备 (cpu / cuda)")
    parser.add_argument("--rounds", type=int, default=100,
                        help="测量轮数 (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="预热轮数 (default: 10)")
    parser.add_argument("--window_size", type=int, default=256,
                        help="窗口大小 (default: 256)")
    parser.add_argument("--feature_name", type=str,
                        default="deviation_px_before_calibrate",
                        help="JSONL 中的特征字段名")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 路径 (default: inference_time.csv)")
    parser.add_argument("--local_mean_size", type=int, default=16,
                        help="sliding mean 窗口大小 (default: 16)")
    parser.add_argument("--per_sample_norm", action="store_true", default=True,
                        help="是否做 per-sample Min-Max 归一化")
    args = parser.parse_args()

    output_csv = Path(args.output) if args.output else OUTPUT_CSV
    device = torch.device(args.device)
    W = args.window_size

    # ---- 1. 读取 JSONL ----
    print("=" * 72)
    print("  推理时长基准测试")
    print("=" * 72)

    jsonl_path = Path(args.jsonl_file)
    if not jsonl_path.exists():
        print(f"[ERROR] 文件不存在: {jsonl_path}")
        sys.exit(1)

    drift, num_frames = load_jsonl(str(jsonl_path), args.feature_name)
    video_duration_sec = num_frames / FPS

    print(f"\n  JSONL 文件:    {jsonl_path.name}")
    print(f"  总帧数:        {num_frames}")
    print(f"  帧率:          {FPS} fps")
    print(f"  视频时长:      {video_duration_sec:.2f} 秒 "
          f"({video_duration_sec / 60:.2f} 分钟)")
    print(f"  窗口大小:      {W} 帧 ({W / FPS:.2f} 秒)")
    print(f"  可用窗口数:    {max(0, num_frames - W + 1)}")
    print(f"  设备:          {device}")
    print(f"  测量轮数:      {args.rounds} (预热 {args.warmup})")
    print()

    # ---- 2. 构建窗口 ----
    window_adf = build_window(
        drift, window_size=W,
        per_sample_norm=args.per_sample_norm,
        local_mean_size=args.local_mean_size,
    )
    print(f"  ADF 窗口形状:  {window_adf.shape}")

    # ---- 3. 准备输入 ----
    inputs = prepare_inputs(window_adf, device)

    # ---- 4. 构建模型并测量 ----
    print(f"\n{'=' * 72}")
    print(f"  {'Model':<20s} {'Input':<12s} {'Mean (ms)':>10s} {'Std (ms)':>10s} "
          f"{'FPS':>8s}")
    print(f"  {'-' * 62}")

    all_models = build_all_models(device, window_size=W)
    results = []

    for name, (model, forward_fn, shape_note) in all_models.items():
        # 找到对应的输入 tensor
        input_tensor = inputs.get(shape_note, None)
        if input_tensor is None:
            print(f"  [SKIP] {name:20s} | 未找到输入格式 {shape_note}")
            continue

        try:
            mean_ms, std_ms = benchmark_model(
                model, input_tensor, forward_fn, device,
                rounds=args.rounds, warmup=args.warmup,
            )
            fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
            results.append({
                "Model": name,
                "Input": shape_note,
                "Mean_ms": f"{mean_ms:.4f}",
                "Std_ms": f"{std_ms:.4f}",
                "FPS": f"{fps:.1f}",
            })
            print(f"  [OK]   {name:<20s} {shape_note:<12s} "
                  f"{mean_ms:>10.4f} {std_ms:>10.4f} {fps:>8.1f}")
        except Exception as e:
            results.append({
                "Model": name,
                "Input": shape_note,
                "Mean_ms": "ERROR",
                "Std_ms": "ERROR",
                "FPS": str(e)[:40],
            })
            print(f"  [FAIL] {name:<20s} | Error: {e}")

    # ---- 5. 汇总 ----
    print(f"\n{'=' * 72}")
    print(f"  汇总")
    print(f"{'=' * 72}")
    header = (f"  {'Model':<20s} {'Input':<12s} "
              f"{'Mean (ms)':>10s} {'Std (ms)':>10s} {'FPS':>8s}")
    print(header)
    print(f"  {'-' * 62}")
    for r in results:
        print(f"  {r['Model']:<20s} {r['Input']:<12s} "
              f"{r['Mean_ms']:>10s} {r['Std_ms']:>10s} {r['FPS']:>8s}")
    print(f"{'=' * 72}")

    # ---- 6. 写入 CSV ----
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Model", "Input", "Mean_ms", "Std_ms", "FPS"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[OK] CSV saved to {output_csv}")
    print(f"     视频时长: {video_duration_sec:.2f}s, "
          f"帧数: {num_frames}, 窗口: {W}")


if __name__ == "__main__":
    main()
