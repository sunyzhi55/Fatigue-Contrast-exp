"""
疲劳检测域适应 (Domain Adaptation) 对比方法配置
包含 MLDA (Multi-Level Domain Adaptation) 方法的实验配置。

参考论文:
    "Multi-level domain adaptation for improved generalization in
     electroencephalogram-based driver fatigue detection"
    Engineering Applications of Artificial Intelligence 142 (2025)

输入适配说明:
    原论文使用 EEG 差分熵特征 (750维) 作为输入。
    本配置使用注视偏差序列 (deviation sequences) 的展平向量替代 EEG，
    input_dim = window_size * num_channels (ADF三通道: 256*3=768)。

使用方法:
    python main_fatigue.py --exp_name Fatigue_MLDA_baseline

配置说明:
    - training_type: "domain_adapt" 触发 MLDA 域适应训练循环
    - mlda_loss_weight: 域间/域内损失平衡权重 λ (论文 Eq.17)
    - mlda_lambda_center: sigmoid 调度中心 epoch (论文 Eq.18)
"""

fatigue_da_experiments = {

    # ====================================================================== #
    #                    MLDA 多级域适应基线                                  #
    # ====================================================================== #
    "Fatigue_MLDA_baseline": {
        # ---- 模型 ----
        "model_name": "mlda",
        "num_classes": 2,
        "checkpoint_path": None,

        # MLDA 特有参数
        "feat_dim": 32,             # 编码器输出特征维度
        "proj_dim": 32,             # U/V 投影网络维度
        "dropout": 0.05,            # 编码器 Dropout 率

        # ---- 数据 ----
        "dataset_name": "FatigueDetection",
        "data_dir": "/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Datapreprocess_l2cs/Data0620_tf_calibrate",

        # 特征与窗口
        "feature_name": "deviation_px_before_calibrate",
        "window_size": 256,
        "stride": 64,
        # ADF 三通道特征: 空间漂移 + 一阶时序差分 + 滑动窗口局部均值
        "use_adf": True,
        "local_mean_size": 16,

        # ---- 数据划分 ----
        "difficulty": "easy",       # 数据类别（easy / hard）
        "test_ids": None,           # 测试集受试者ID（None 表示无独立测试集）
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
        "training_type": "domain_adapt",
        "trainer_name": "TrainerForMLDA",
        "loss_fn_name": "CrossEntropyLoss",
        "label_smoothing": 0.0,
        "optimizer_name": "SGD",

        "batch_size": 64,
        "epochs": 1000,
        "patience": 1000,
        "k_fold": 5,               # fold 数量（与 folds 字典一致）
        "val_strategy": "kfold",   # 验证策略：kfold 或 loso

        # ---- 超参数 ----
        "lr": 5e-3,
        "weight_decay": 5e-4,
        "lr_policy": "onecycle",
        "lr_decay": 0.95,
        "niter": 50,

        # ---- MLDA 域适应参数 ----
        "mlda_loss_weight": 0.5,   # 域间/域内损失平衡权重 λ
        "mlda_lambda_center": 100, # sigmoid 调度中心 epoch

        # ---- 系统 ----
        "device": "cuda:0",
        "seed": 42,
        "output_dir": "./result",

        # ---- SwanLab（可选） ----
        "use_swanlab": False,
        "swanlab_description": "Fatigue Detection - MLDA Domain Adaptation Baseline",
        "swanlab_num_samples": 8,
    },

    # ====================================================================== #
    #                    DAEEGViT 域适应 Vision Transformer 基线              #
    # ====================================================================== #
    "Fatigue_DAEEGViT_baseline": {
        # ---- 模型 ----
        "model_name": "daeevit",
        "num_classes": 2,
        "checkpoint_path": None,

        # DAEEGViT 特有参数
        "embed_dim": 64,             # 嵌入维度 D
        "depth": 4,                  # Transformer 层数
        "num_heads": 4,              # 注意力头数
        "patch_size": 32,            # Patch 大小 (256/32=8 patches)
        "mlp_ratio": 4.0,            # MLP 隐藏层比率
        "qkv_bias": True,            # QKV 偏置
        "dropout": 0.1,              # Dropout 率
        "attn_drop_ratio": 0.0,      # 注意力 Dropout 率
        "drop_path_ratio": 0.1,      # Stochastic Depth 率
        "mbconv_expand_ratio": 4,    # MBConv 扩展比率
        "mbconv_se_ratio": 0.25,     # MBConv SE 比率
        "representation_size": None, # 表示层维度 (None=不使用)

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
        "training_type": "domain_adapt_vit",
        "trainer_name": "TrainerForDAEEGViT",
        "loss_fn_name": "CrossEntropyLoss",
        "label_smoothing": 0.0,
        "optimizer_name": "AdamW",

        "batch_size": 64,
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

        # ---- 域适应参数 ----
        "mmd_weight": 1.0,           # MMD 损失权重 (论文 Eq.8: L = L_cls + L_mmd)

        # ---- 系统 ----
        "device": "cuda:0",
        "seed": 42,
        "output_dir": "./result",

        # ---- SwanLab（可选） ----
        "use_swanlab": False,
        "swanlab_description": "Fatigue Detection - DAEEGViT Domain Adaptation Baseline",
        "swanlab_num_samples": 8,
    },

    # ====================================================================== #
    #                    LA-MSDA 多源标签域适应基线                            #
    # ====================================================================== #
    "Fatigue_LA_MSDA_baseline": {
        # ---- 模型 ----
        "model_name": "lamsda",
        "num_classes": 2,
        "checkpoint_path": None,

        # LA-MSDA 特有参数
        "num_sources": 5,             # 源域分支数量 (上限，实际=训练受试者数)
        "feature_dim": 64,            # 共享特征提取器输出维度
        "ds_hidden_dim": 256,         # 域特定网络隐藏维度

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
        "training_type": "multi_source_da",
        "trainer_name": "TrainerForLAMSDA",
        "loss_fn_name": "CrossEntropyLoss",
        "label_smoothing": 0.0,
        "optimizer_name": "SGD",

        "batch_size": 128,
        "epochs": 1000,
        "patience": 1000,
        "k_fold": 5,
        "val_strategy": "kfold",

        # ---- 超参数 ----
        "lr": 1e-3,
        "weight_decay": 5e-4,
        "lr_policy": "none",          # 原论文无 LR 调度，使用 sigmoid 损失调度
        "lr_decay": 0.95,
        "niter": 50,

        # ---- LA-MSDA 域适应参数 ----
        "da_warmup_scale": 10.0,      # sigmoid 预热陡峭度 (原论文=10)

        # ---- 系统 ----
        "device": "cuda:0",
        "seed": 42,
        "output_dir": "./result",

        # ---- SwanLab（可选） ----
        "use_swanlab": False,
        "swanlab_description": "Fatigue Detection - LA-MSDA Multi-Source DA Baseline",
        "swanlab_num_samples": 8,
    },
}
