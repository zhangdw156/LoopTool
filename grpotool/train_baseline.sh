export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4
export ROLLOUT_TP_SIZE=1
# export VLLM_ATTENTION_BACKEND=XFORMERS

export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=1
export STRICTMATCH=0
export CORRECTMAX1=1
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0
export RESPONSE_HALF_REWARD=0  
export TOOL_REWARD_VERSION=1 
export ERRORMAX=0


# Change
export USE_KL_LOSS=False
export VAL_BEFORE_TRAIN=True
export CLIP_RATIO_HIGH=0.28
export TRAIN_BATCH_SIZE=256
export VAL_BATCH_SIZE=256
export LR=1e-6
export STEPS_SAVE=10 # To Change
export EPOCHS=4
export HYDRA_FULL_ERROR=1
export GROUP_SIZE=16
export DATA_DIR="/dfs/data/datasets/LoopTool-23k" 
export TRAIN_FILE=$DATA_DIR/train.parquet  # To Change
export VAL_FILE=$DATA_DIR/test.parquet
export BASE_MODEL="/dfs/data/models/Qwen3-4B"
export PROJECT_NAME="LoopTool"
export EXPERIMENT_NAME="modelfactory-qwen-baseline-v1" # e.g., "grpo-qwen2.5-3b"  # To Change

export RAY_TMP_DIR="/dfs/data/tmp/ray"
export RAY_tmp_dir="$RAY_TMP_DIR"
export TMPDIR="$RAY_TMP_DIR"

bash grpo_qwen.sh
