
# Run experiments for different models

# Temporal Baselines
# ---------------------------------------------------
# LSTM
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LSTM_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LSTM_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LSTM_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LSTM_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LSTM_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LSTM_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LSTM_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LSTM_loso_hard_0723.out &

# ----------------------------------------------------
# Transformer
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold （请同步更改config）
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_Transformer_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_Transformer_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_Transformer_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_Transformer_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_Transformer_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_Transformer_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_Transformer_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_Transformer_loso_hard_0723.out &

# ----------------------------------------------------
# TimesNet
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_TimesNet_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_TimesNet_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_TimesNet_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_TimesNet_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_TimesNet_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_TimesNet_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_TimesNet_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_TimesNet_loso_hard_0723.out &

# ----------------------------------------------------
# STAFNet (IEEE TIM 2026)
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_STAFNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_STAFNet_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_STAFNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_STAFNet_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_STAFNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_STAFNet_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_STAFNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_STAFNet_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_STAFNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_STAFNet_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_STAFNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_STAFNet_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_STAFNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_STAFNet_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_STAFNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_STAFNet_loso_hard_0723.out &


# ----------------------------------------------------
# Few-Shot Baselines

# ProtoNet
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_ProtoNet_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_ProtoNet_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_ProtoNet_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_ProtoNet_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_ProtoNet_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_ProtoNet_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_ProtoNet_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_ProtoNet_loso_hard_0723.out &


# ----------------------------------------------------
# RelationNet
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_RelationNet_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_RelationNet_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_RelationNet_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_RelationNet_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_RelationNet_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_RelationNet_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_RelationNet_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_RelationNet_loso_hard_0723.out &

# ----------------------------------------------------
# Domain Generalization Baselines
# InterpCNN
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_InterpCNN_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_InterpCNN_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_InterpCNN_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_InterpCNN_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_InterpCNN_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_InterpCNN_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_InterpCNN_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_InterpCNN_loso_hard_0723.out &

# -----------------------------------------------------
# AFM-CIR
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_AFM_CIR_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_AFM_CIR_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_AFM_CIR_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_AFM_CIR_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_AFM_CIR_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_AFM_CIR_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_AFM_CIR_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_AFM_CIR_loso_hard_0723.out &


# Domain Adaptation Baselines
# -----------------------------------------------------
# MLDA
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_MLDA_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_MLDA_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_MLDA_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_MLDA_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_MLDA_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_MLDA_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_MLDA_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_MLDA_loso_hard_0723.out &


# -----------------------------------------------------
# DAEEGViT
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DAEEGViT_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DAEEGViT_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DAEEGViT_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DAEEGViT_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DAEEGViT_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DAEEGViT_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DAEEGViT_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DAEEGViT_loso_hard_0723.out &



# -----------------------------------------------------
# LA-MSDA
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LA_MSDA_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LA_MSDA_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LA_MSDA_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LA_MSDA_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LA_MSDA_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LA_MSDA_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LA_MSDA_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_LA_MSDA_loso_hard_0723.out &


# -----------------------------------------------------
# DANN
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DANN_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DANN_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DANN_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DANN_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DANN_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DANN_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DANN_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DANN_loso_hard_0723.out &


# -----------------------------------------------------
# DeepCORAL
# Train on FatigueGuard, test on GAIPAT (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DeepCORAL_5fold_easy_0722.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DeepCORAL_loso_easy_0722.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DeepCORAL_5fold_hard_0722.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DeepCORAL_loso_hard_0722.out &

# Train on GAIPAT, test on FatigueGuard (easy + hard)
# easy + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DeepCORAL_5fold_easy_0723.out &
# easy + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DeepCORAL_loso_easy_0723.out &
# hard + kfold
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DeepCORAL_5fold_hard_0723.out &
# hard + loso
CUDA_VISIBLE_DEVICES=0 nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode gaipat_to_fatigue --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --per_sample_norm > result_DeepCORAL_loso_hard_0723.out &


