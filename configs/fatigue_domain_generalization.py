"""
疲劳检测域泛化 (Domain Generalization) 对比方法配置
包含以下域泛化方法的实验配置:

使用方法：
    python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline
    python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline

InterpretableCNN 说明:
    - training_type: "dg_interpcnn" 触发域泛化训练循环
    - 纯 ERM 基线: 仅在源域被试上训练，目标域数据完全不参与
    - 参考论文: Cui et al. "EEG-Based Cross-Subject Driver Drowsiness
      Recognition With an Interpretable Convolutional Neural Network."
      IEEE TNNLS, 2022.

AFM-CIR 说明:
    - training_type: "dg_afmcir" 触发因果域泛化训练循环
    - 三阶段: 引导编码器预训练 → AFM 频域增强 → 因果启发训练
    - 核心: 自适应傅里叶 Mixup + FAC 关联因子化 + 对抗掩码
    - 参考论文: Zhu et al. "Causality-Preserving Domain Generalization
      via Adaptive Fourier Mixup for RUL Prediction." IEEE TPAMI, 2026.
"""


DATA_DIR = "/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Datapreprocess_l2cs/Data0620_tf_calibrate"
BATCH_SIZE = 128
EPOCHS = 200
PATIENCE = 200
VAL_STRATEGY = "loso"  # 验证策略：kfold 或 loso
DIFFICULTY = "easy"  # 数据类别（easy / hard）
fatigue_dg_experiments={
    # ====================================================================== #
    #              InterpretableCNN 域泛化基线 (DG, 纯 ERM)                    #
    # ====================================================================== #
    "Fatigue_InterpCNN_baseline": {
        # ---- 模型 ----
        "model_name": "interpcnn",
        "num_classes": 2,
        "checkpoint_path": None,

        # InterpretableCNN 特有参数 (论文默认值)
        "n_filters": 16,              # 通道混合滤波器数 N1 (论文: 16)
        "depth_multiplier": 2,        # 深度可分离乘数 d (论文: 2)
        "kernel_size": 64,            # 时间卷积核大小 (论文: 64)
        "dropout": 0.0,               # 分类器前 Dropout (论文: 0)

        # ---- 数据 ----
        "dataset_name": "FatigueDetection",
        "data_dir": DATA_DIR,

        # 特征与窗口
        "feature_name": "deviation_px_before_calibrate",
        "window_size": 256,
        "stride": 64,
        "use_adf": True,
        "local_mean_size": 16,

        # ---- 数据划分 ----
        "difficulty": DIFFICULTY,
        "test_ids": None,
        "folds": {
            1: {"val_ids": ["01", "05", "14", "19"]},
            2: {"val_ids": ["02", "06", "10", "15"]},
            3: {"val_ids": ["07", "11", "16", "20"]},
            4: {"val_ids": ["03", "08", "12", "17"]},
            5: {"val_ids": ["04", "09", "13", "18"]},
        },

        # ---- 训练 ----
        "training_type": "dg_interpcnn",
        "trainer_name": "TrainerForInterpCNN",
        "loss_fn_name": "NLLLoss",
        "label_smoothing": 0.0,
        "optimizer_name": "AdamW",

        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "k_fold": 5,
        "val_strategy": VAL_STRATEGY,

        # ---- 超参数 ----
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "lr_policy": "onecycle",
        "lr_decay": 0.95,
        "niter": 50,

        # ---- 系统 ----
        "device": "cuda:1",
        "seed": 42,
        "output_dir": "./result",

        # ---- SwanLab（可选） ----
        "use_swanlab": False,
        "swanlab_description": "Fatigue Detection - InterpretableCNN DG Baseline",
        "swanlab_num_samples": 8,
    },

    # ====================================================================== #
    #        AFM-CIR 因果域泛化 (Adaptive Fourier Mixup + CIR)                 #
    # ====================================================================== #
    "Fatigue_AFM_CIR_baseline": {
        # ---- 模型 ----
        "model_name": "afmcir",
        "num_classes": 2,
        "checkpoint_path": None,

        # AFMCIRNet 主模型参数
        "feat_dim": 64,               # 特征维度 N
        "dropout": 0.1,               # Dropout
        "adv_hidden": 64,             # 对抗掩码器隐藏层
        "kappa": 0.8,                 # 优势维度比例 (论文: 0.8)

        # ---- 数据 ----
        "dataset_name": "FatigueDetection",
        "data_dir": DATA_DIR,

        # 特征与窗口
        "feature_name": "deviation_px_before_calibrate",
        "window_size": 256,
        "stride": 64,
        "use_adf": True,
        "local_mean_size": 16,

        # ---- 数据划分 ----
        "difficulty": DIFFICULTY,
        "test_ids": None,
        "folds": {
            1: {"val_ids": ["01", "05", "14", "19"]},
            2: {"val_ids": ["02", "06", "10", "15"]},
            3: {"val_ids": ["07", "11", "16", "20"]},
            4: {"val_ids": ["03", "08", "12", "17"]},
            5: {"val_ids": ["04", "09", "13", "18"]},
        },

        # ---- 训练 ----
        "training_type": "dg_afmcir",
        "trainer_name": "TrainerForAFMCIR",
        "loss_fn_name": "NLLLoss",
        "label_smoothing": 0.0,
        "optimizer_name": "AdamW",

        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "k_fold": 5,
        "val_strategy": VAL_STRATEGY,

        # ---- 超参数 ----
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "lr_policy": "onecycle",
        "lr_decay": 0.95,
        "niter": 50,

        # ---- AFM-CIR 特有超参数 ----
        # Phase 1: 引导编码器预训练
        "guidance_embed_dim": 32,     # 引导嵌入维度
        "guidance_epochs": 50,        # 引导编码器预训练轮数
        "guidance_lr": 1e-3,          # 引导编码器学习率
        "guidance_alpha_adv": 1.0,    # 域对抗损失权重
        "guidance_alpha_rnc": 1.0,    # RNC 对比损失权重
        "guidance_tau": 0.1,          # RNC 温度参数

        # Phase 2: AFM 增强
        "afm_gamma_A": 0.5,           # 振幅混合系数下界 (论文: 0.5)
        "afm_gamma_P": 0.9,           # 相位扰动系数下界 (论文: 0.9)
        "afm_eta": 0.8,               # 相位邻居阈值 (论文: 0.8)

        # Phase 3: CIR 训练
        "cir_tau_fac": 2.0,           # FAC 损失权重 (论文: 2)
        "cir_adv_weight": 0.5,        # 对抗掩码损失权重

        # ---- 系统 ----
        "device": "cuda:7",
        "seed": 42,
        "output_dir": "./result",

        # ---- SwanLab（可选） ----
        "use_swanlab": False,
        "swanlab_description": "Fatigue Detection - AFM-CIR Causal DG",
        "swanlab_num_samples": 8,
    },
}