#!/usr/bin/env bash
# 等待 GPU 空闲后运行 P4 benchmark
set -euo pipefail

NEEDED_FREE_MB=50000  # TP=4 每卡需要 ~12GB，留 50GB 余量
CHECK_GPUS="0 1 2 3 4 5 6 7"
POLL_INTERVAL=60
LOG="/tmp/wait_and_test_p4_$(date +%Y%m%d_%H%M%S).log"

echo "[monitor] Waiting for GPUs to have ${NEEDED_FREE_MB}MiB free each..." | tee "$LOG"
echo "[monitor] Checking every ${POLL_INTERVAL}s" | tee -a "$LOG"

while true; do
    all_ok=true
    status=""
    for gpu_id in $CHECK_GPUS; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i "$gpu_id" 2>/dev/null | tr -d ' MiB')
        status="${status} GPU${gpu_id}=${free}MiB"
        if [[ -z "$free" ]] || (( free < NEEDED_FREE_MB )); then
            all_ok=false
        fi
    done

    if $all_ok; then
        echo "[monitor] All GPUs ready!${status}" | tee -a "$LOG"
        break
    fi

    echo "[monitor]${status} - waiting ${POLL_INTERVAL}s..." | tee -a "$LOG"
    sleep "$POLL_INTERVAL"
done

cd /home/honglin/meta-infer
source /home/honglin/.venv/bin/activate

echo "[bench] Running P4 benchmark (DeepSeek-V2-Lite, TP=4)..." | tee -a "$LOG"
SKIP_VLLM=1 CUDA_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 ROUNDS=25 STEPS=32 REQUEST_RATE=4 MAX_CONCURRENCY=1 \
  bash run_compare_metainfer_vllm.sh dsv2 2>&1 | tee -a "$LOG"

echo "[bench] Done." | tee -a "$LOG"
echo "[bench] Log: $LOG"
