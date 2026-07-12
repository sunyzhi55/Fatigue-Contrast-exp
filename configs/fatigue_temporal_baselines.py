"""
疲劳检测时序基线方法配置
包含 LSTM、Transformer、Mamba、TimesNet 四种时序分类模型的实验配置。

使用方法:
    python main_fatigue.py --exp_name Fatigue_LSTM_baseline
    python main_fatigue.py --exp_name Fatigue_Transformer_baseline
    python main_fatigue.py --exp_name Fatigue_Mamba_baseline
    python main_fatigue.py --exp_name Fatigue_TimesNet_baseline

配置说明:
    - data_dir: JSONL 数据文件目录（绝对路径）
    - test_ids: 测试集受试者ID列表（可选，None则无独立测试集）
    - folds: K-Fold 配置，每个fold指定验证集受试者ID
    - window_size / stride: 滑动窗口参数
    - feature_name: 使用的特征字段
"""

DATA_DIR = "/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Datapreprocess_l2cs/Data0620_tf_calibrate"
BATCH_SIZE = 128
EPOCHS = 200
PATIENCE = 200
VAL_STRATEGY = "loso"  # 验证策略：kfold 或 loso
DIFFICULTY = "easy"  # 数据类别（easy / hard）
fatigue_temporal_experiments = {

    # ====================================================================== #
    #                    LSTM 时序基线                                        #
    # ====================================================================== #
    "Fatigue_LSTM_baseline": {
        # ---- 模型 ----
        "model_name": "lstm",
        "num_classes": 2,
        "checkpoint_path": None,

        # LSTM 特有参数
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": False,

        # ---- 数据 ----
        "dataset_name": "FatigueDetection",
        "data_dir": DATA_DIR,

        # 特征与窗口
        "feature_name": "deviation_px_before_calibrate",
        "window_size": 256,
        "stride": 64,
        # ADF 三通道特征: 空间漂移 + 一阶时序差分 + 滑动窗口局部均值
        "use_adf": True,
        "local_mean_size": 16,

        # ---- 数据划分 ----
        "difficulty": DIFFICULTY,  # 数据类别（easy / hard）
        # 测试集受试者ID（None 表示无独立测试集）
        "test_ids": None,
        # K-Fold 配置（LOSO 时由脚本自动生成）
        # 每个 fold 指定 val_ids（验证集受试者ID），剩余为训练集
        "folds": {
            1: {"val_ids": ["01", "05", "14", "19"]},
            2: {"val_ids": ["02", "06", "10", "15"]},
            3: {"val_ids": ["07", "11", "16", "20"]},
            4: {"val_ids": ["03", "08", "12", "17"]},
            5: {"val_ids": ["04", "09", "13", "18"]},
        },

        # ---- 训练 ----
        "trainer_name": "TrainerForFatigue",
        "loss_fn_name": "CrossEntropyLoss",
        "label_smoothing": 0.1,
        "optimizer_name": "AdamW",

        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "k_fold": 5,        # fold 数量（与 folds 字典一致）
        "val_strategy": VAL_STRATEGY,  # 验证策略：kfold 或 loso

        # ---- 超参数 ----
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "lr_policy": "onecycle",
        "lr_decay": 0.95,
        "niter": 50,

        # ---- 系统 ----
        "device": "cuda:0",
        "seed": 42,
        "output_dir": "./result",

        # ---- SwanLab（可选） ----
        "use_swanlab": False,
        "swanlab_description": "Fatigue Detection - LSTM Baseline",
        "swanlab_num_samples": 8,
    },

    # ====================================================================== #
    #                    Transformer 时序基线                                 #
    # ====================================================================== #
    "Fatigue_Transformer_baseline": {
        # ---- 模型 ----
        "model_name": "transformer",
        "num_classes": 2,
        "checkpoint_path": None,

        # Transformer 特有参数
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.2,
        "max_seq_len": 500,

        # ---- 数据 ----
        "dataset_name": "FatigueDetection",
        "data_dir": DATA_DIR,

        # 特征与窗口
        "feature_name": "deviation_px_before_calibrate",
        "window_size": 256,
        "stride": 64,
        # ADF 三通道特征: 空间漂移 + 一阶时序差分 + 滑动窗口局部均值
        "use_adf": True,
        "local_mean_size": 16,

        # ---- 数据划分 ----
        "difficulty": DIFFICULTY,  # 数据类别（easy / hard）
        "test_ids": None,
        "folds": {
            1: {"val_ids": ["01", "05", "14", "19"]},
            2: {"val_ids": ["02", "06", "10", "15"]},
            3: {"val_ids": ["07", "11", "16", "20"]},
            4: {"val_ids": ["03", "08", "12", "17"]},
            5: {"val_ids": ["04", "09", "13", "18"]},
        },

        # ---- 训练 ----
        "trainer_name": "TrainerForFatigue",
        "loss_fn_name": "CrossEntropyLoss",
        "label_smoothing": 0.1,
        "optimizer_name": "AdamW",

        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "k_fold": 5,        # fold 数量（与 folds 字典一致）
        "val_strategy": VAL_STRATEGY,  # 验证策略：kfold 或 loso

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

        # ---- SwanLab ----
        "use_swanlab": False,
        "swanlab_description": "Fatigue Detection - Transformer Baseline",
        "swanlab_num_samples": 8,
    },

    # ====================================================================== #
    #                    Mamba 时序基线                                       #
    # ====================================================================== #
    "Fatigue_Mamba_baseline": {
        # ---- 模型 ----
        "model_name": "mamba",
        "num_classes": 2,
        "checkpoint_path": None,

        # Mamba 特有参数（基于 mamba-ssm 包）
        "d_model": 64,
        "n_layer": 2,
        "d_conv": 4,
        "d_state": 16,       # SSM 状态维度
        "expand": 2,         # 内部扩展因子
        "dropout": 0.2,

        # ---- 数据 ----
        "dataset_name": "FatigueDetection",
        "data_dir": DATA_DIR,

        # 特征与窗口
        "feature_name": "deviation_px_before_calibrate",
        "window_size": 256,
        "stride": 64,
        # ADF 三通道特征: 空间漂移 + 一阶时序差分 + 滑动窗口局部均值
        "use_adf": True,
        "local_mean_size": 16,

        # ---- 数据划分 ----
        "difficulty": DIFFICULTY,  # 数据类别（easy / hard）
        "test_ids": None,
        "folds": {
            1: {"val_ids": ["01", "05", "14", "19"]},
            2: {"val_ids": ["02", "06", "10", "15"]},
            3: {"val_ids": ["07", "11", "16", "20"]},
            4: {"val_ids": ["03", "08", "12", "17"]},
            5: {"val_ids": ["04", "09", "13", "18"]},
        },

        # ---- 训练 ----
        "trainer_name": "TrainerForFatigue",
        "loss_fn_name": "CrossEntropyLoss",
        "label_smoothing": 0.1,
        "optimizer_name": "AdamW",

        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "k_fold": 5,        # fold 数量（与 folds 字典一致）
        "val_strategy": VAL_STRATEGY,  # 验证策略：kfold 或 loso
        # ---- 超参数 ----
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "lr_policy": "onecycle",
        "lr_decay": 0.95,
        "niter": 50,

        # ---- 系统 ----
        "device": "cuda:3",
        "seed": 42,
        "output_dir": "./result",

        # ---- SwanLab ----
        "use_swanlab": False,
        "swanlab_description": "Fatigue Detection - Mamba Baseline",
        "swanlab_num_samples": 8,
    },

    # ====================================================================== #
    #                    TimesNet 时序基线 (ICLR 2023)                         #
    # ====================================================================== #
    "Fatigue_TimesNet_baseline": {
        # ---- 模型 ----
        "model_name": "timesnet",
        "num_classes": 2,
        "checkpoint_path": None,

        # TimesNet 特有参数
        "d_model": 32,              # 模型维度 (论文: min(max(2*ceil(log2(C)),32),64))
        "d_ff": 64,                 # Inception 瓶颈层维度
        "num_kernels": 6,           # Inception 核数 (核大小 1,3,5,7,9,11)
        "top_k": 3,                 # FFT top-k 周期数 (论文分类默认 3)
        "e_layers": 2,              # TimesBlock 层数 (论文分类默认 2)
        "dropout": 0.1,             # Dropout

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
        "trainer_name": "TrainerForTimesNet",
        "loss_fn_name": "CrossEntropyLoss",
        "label_smoothing": 0.0,
        "optimizer_name": "AdamW",

        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "k_fold": 5,        # fold 数量, 与下方 folds 字典一致
        "val_strategy": VAL_STRATEGY,  # 验证策略: kfold 或 loso

        # ---- 超参数 ----
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "lr_policy": "onecycle",
        "lr_decay": 0.95,
        "niter": 50,

        # ---- 系统 ----
        "device": "cuda:6",
        "seed": 42,
        "output_dir": "./result",

        # ---- SwanLab ----
        "use_swanlab": False,
        "swanlab_description": "Fatigue Detection - TimesNet Baseline (ICLR 2023)",
        "swanlab_num_samples": 8,
    },
}
