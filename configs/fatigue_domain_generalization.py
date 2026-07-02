"""
疲劳检测域泛化 (Domain Generalization) 对比方法配置
包含以下域泛化方法的实验配置:

使用方法：
    python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline

InterpretableCNN 说明:
    - training_type: "dg_interpcnn" 触发域泛化训练循环
    - 纯 ERM 基线: 仅在源域被试上训练，目标域数据完全不参与
    - 参考论文: Cui et al. "EEG-Based Cross-Subject Driver Drowsiness
      Recognition With an Interpretable Convolutional Neural Network."
      IEEE TNNLS, 2022.
"""


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
        "data_dir": "/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Datapreprocess_l2cs/Data0620_tf_calibrate",

        # 特征与窗口
        "feature_name": "deviation_px_before_calibrate",
        "window_size": 256,
        "stride": 64,
        "use_adf": True,
        "local_mean_size": 16,

        # ---- 数据划分 ----
        "difficulty": "easy",
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

        "batch_size": 128,
        "epochs": 1000,
        "patience": 1000,
        "k_fold": 5,
        "val_strategy": "kfold",

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
        "swanlab_description": "Fatigue Detection - InterpretableCNN DG Baseline",
        "swanlab_num_samples": 8,
    },
}