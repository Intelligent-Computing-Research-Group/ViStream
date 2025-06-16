#!/usr/bin/env bash

EXP_NAME="non-LCC"         # 按需修改
CFG_PATH="config/${EXP_NAME}.yaml"
SMRY_ROOT="results/summary/${EXP_NAME}"
mkdir -p "$SMRY_ROOT"

QANN_PATH="/home/zilingwei/UniTrack-main/checkpoint-90.pth"
LOG_DIR="/home/zilingwei/UniTrack-main/output"

for t in $(seq 1 16); do
    LOG_FILE="${SMRY_ROOT}/vos_T=${t}.log"

    # 如果该步已有日志文件且大小 >0，则视为已完成并跳过
    if [[ -s "$LOG_FILE" ]]; then
        echo "[SKIP] time_step=${t} —— 已存在 ${LOG_FILE}"
        continue
    fi

    echo "========== Running time_step=${t} =========="
    CUDA_VISIBLE_DEVICES=0 \
    python -u test/test_vos.py \
        --config "$CFG_PATH" \
        --QANNPath "$QANN_PATH" \
        --level 16 \
        --weight_quantization_bit 32 \
        --time_step "$t" \
        --encoding_type analog \
        --log_dir "$LOG_DIR" \
    | tee "$LOG_FILE"

    echo "========== Finished time_step=${t} =========="
done

echo "全部循环已结束（或已跳过已完成步骤）。"
