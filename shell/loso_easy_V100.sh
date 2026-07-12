
# Run experiments for different models

# Temporal Baselines
# easy + kfold
nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_LSTM_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_LSTM_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_LSTM_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_LSTM_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_LSTM_loso_hard_0708.out &



# ----------------------------------------------------
# easy + kfold （请同步更改config）
nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_Transformer_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_Transformer_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_Transformer_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_Transformer_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_Transformer_loso_hard_0708.out &

# ----------------------------------------------------
# easy + kfold
nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_TimesNet_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_TimesNet_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_TimesNet_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_TimesNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_TimesNet_loso_hard_0708.out &

# ----------------------------------------------------
# Few-Shot Baselines
# easy + kfold
nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_ProtoNet_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_ProtoNet_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_ProtoNet_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_ProtoNet_loso_hard_0708.out &

# ----------------------------------------------------
# easy + kfold
nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_RelationNet_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_RelationNet_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_RelationNet_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_RelationNet_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_RelationNet_loso_hard_0708.out &

# ----------------------------------------------------
# Domain Generalization Baselines
# easy + kfold
nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_InterpCNN_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_InterpCNN_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_InterpCNN_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_InterpCNN_loso_hard_0708.out &

# -----------------------------------------------------
# easy + kfold
nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_AFM_CIR_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_AFM_CIR_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_AFM_CIR_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_AFM_CIR_loso_hard_0708.out &



# Domain Adaptation Baselines
# -----------------------------------------------------
# easy + kfold
nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_MLDA_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_MLDA_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_MLDA_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_MLDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_MLDA_loso_hard_0708.out &


# -----------------------------------------------------
# easy + kfold
nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DAEEGViT_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DAEEGViT_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DAEEGViT_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DAEEGViT_loso_hard_0708.out &


# -----------------------------------------------------
# easy + kfold
nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_LA_MSDA_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_LA_MSDA_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_LA_MSDA_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_LA_MSDA_loso_hard_0708.out &

# -----------------------------------------------------
# easy + kfold
nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DANN_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DANN_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DANN_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_DANN_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DANN_loso_hard_0708.out &


# -----------------------------------------------------
# easy + kfold
nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DeepCORAL_5fold_easy_0708.out &
# easy + loso
nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DeepCORAL_loso_easy_0708.out &
# hard + kfold
nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DeepCORAL_5fold_hard_0708.out &
# hard + loso
nohup python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline --eval_mode fatigue_to_gaipat --gaipat_dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data --per_sample_norm > result_DeepCORAL_loso_hard_0708.out &