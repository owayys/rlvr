#!/usr/bin/env bash
set -euo pipefail

# RLVR GRPO training script for countdown task with Qwen3 models
# Usage:
#   ./scripts/train.sh 0.6b
#   ./scripts/train.sh 1.7b

MODEL_SIZE=${1:?Usage: ./scripts/train.sh <0.6b|1.7b>}

if [[ "$MODEL_SIZE" != "0.6b" && "$MODEL_SIZE" != "1.7b" ]]; then
    echo "Error: MODEL_SIZE must be 0.6b or 1.7b" >&2
    exit 1
fi

# Set defaults
export N_GPUS=${N_GPUS:-1}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export NNODES=${NNODES:-1}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-${MODEL_SIZE}-Base}
export DATA_DIR=${DATA_DIR:-./data/countdown}
export RLVR_OUT_DIR=${RLVR_OUT_DIR:-./checkpoints}

CONFIG_FILE="configs/countdown_qwen3_${MODEL_SIZE}.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

if [[ ! -f "$DATA_DIR/train.parquet" ]]; then
    echo "Error: Data file not found: $DATA_DIR/train.parquet" >&2
    echo "Run: python scripts/preprocess_countdown.py" >&2
    exit 1
fi

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=.logs
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/countdown-qwen3-${MODEL_SIZE}-${RUN_TIMESTAMP}.log"

echo "Starting RLVR GRPO training..."
echo "Model size: $MODEL_SIZE"
echo "Config: $CONFIG_FILE"
echo "Data dir: $DATA_DIR"
echo "Base model: $BASE_MODEL"
echo "N_GPUS: $N_GPUS"
echo "ROLLOUT_TP_SIZE: $ROLLOUT_TP_SIZE"
echo "Log file: $LOG_FILE"
echo "---"

python -m rlvr.main_grpo \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "---"
echo "Training completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
