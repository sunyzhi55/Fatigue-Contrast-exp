"""
疲劳检测小样本学习基线方法配置
包含 ProtoNet、RelationNet 两种小样本学习模型的实验配置。

使用方法:
    python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline
    python main_fatigue.py --exp_name Fatigue_RelationNet_baseline

配置说明:
    - 小样本学习使用 Episodic Training（N-way K-shot）
    - n_way: 每个 episode 的类别数（默认2: alert/sleepy）
    - k_shot: 每个类别的支持集样本数
    - n_query: 每个类别的查询集样本数
    - episodes_per_epoch: 每个 epoch 的 episode 数量
"""
DATA_DIR = "/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Datapreprocess_l2cs/Data0620_tf_calibrate"
BATCH_SIZE = 128
EPOCHS = 200
PATIENCE = 200
VAL_STRATEGY = "loso"  # 验证策略：kfold 或 loso
DIFFICULTY = "easy"  # 数据类别（easy / hard）
fatigue_fewshot_experiments = {

    # ====================================================================== #
    #                    ProtoNet 小样本基线                                  #
    # ====================================================================== #
    "Fatigue_ProtoNet_baseline": {
        # ---- 模型 ----
        "model_name": "protonet",
        "num_classes": 2,
        "checkpoint_path": None,

        # ProtoNet 特有参数
        "hidden_size": 64,
        "embedding_size": 32,
        "dropout": 0.2,

        # ---- 数据 ----
        "dataset_name": "FatigueDetection_FewShot",
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

        # ---- 小样本训练参数 ----
        "n_way": 2,             # 类别数（alert/sleepy）
        "k_shot": 5,            # 支持集样本数/类别
        "n_query": 10,          # 查询集样本数/类别
        "episodes_per_epoch": 100,  # 每个 epoch 的 episode 数

        # ---- 训练 ----
        "trainer_name": "TrainerForFewShot",
        "loss_fn_name": "NLLLoss",   # ProtoNet 使用 NLLLoss（由内部计算）
        "optimizer_name": "AdamW",

        "batch_size": 1,        # Episodic training，batch_size=1
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

        # ---- SwanLab ----
        "use_swanlab": False,
        "swanlab_description": "Fatigue Detection - ProtoNet Baseline",
        "swanlab_num_samples": 8,
    },

    # ====================================================================== #
    #                    RelationNet 小样本基线                               #
    # ====================================================================== #
    "Fatigue_RelationNet_baseline": {
        # ---- 模型 ----
        "model_name": "relationnet",
        "num_classes": 2,
        "checkpoint_path": None,

        # RelationNet 特有参数
        "hidden_size": 64,
        "embedding_size": 32,
        "relation_size": 16,
        "dropout": 0.2,

        # ---- 数据 ----
        "dataset_name": "FatigueDetection_FewShot",
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

        # ---- 小样本训练参数 ----
        "n_way": 2,
        "k_shot": 5,
        "n_query": 10,
        "episodes_per_epoch": 100,

        # ---- 训练 ----
        "trainer_name": "TrainerForFewShot",
        "loss_fn_name": "NLLLoss",
        "optimizer_name": "AdamW",

        "batch_size": 1,
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
        "swanlab_description": "Fatigue Detection - RelationNet Baseline",
        "swanlab_num_samples": 8,
    },
}
