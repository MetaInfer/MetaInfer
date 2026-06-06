#!/usr/bin/env bash
# agent-engine TP=4 API benchmark（修复版）
#
# 与 bench_agent.sh 的区别：
#   1. 服务日志不丢 /dev/null，输出到文件便于调试
#   2. 使用 trap 确保 Ctrl+C 和被 kill 时子进程全部清理
#   3. pkill -9 -f "openai_tp_server.py" 匹配 python 子进程（非 torchrun）
#   4. 健康检查有超时，超时后自动失败退出
#   5. 可配 NCCL timeout，避免 NCCL 挂死
#
# 用法：
#   # 默认 TP=4，GPU=0,1,2,3
#   bash /home/honglin/inference-agent-system/bench_api.sh
#
#   # 自定义参数
#   CUDA_DEVICES=0,1 PORT=9002 ROUNDS=20 STEPS="16 64" \
#     bash /home/honglin/inference-agent-system/bench_api.sh
#
#   # 单步压测（只测 decode=32）
#   STEPS=32 bash /home/honglin/inference-agent-system/bench_api.sh
#
# 环境变量：
#   CUDA_DEVICES       GPU 设备号，默认 0,1,2,3
#   PORT               服务端口，默认 9001
#   ROUNDS             每个 step 的请求数，默认 10
#   STEPS              空格分隔的输出长度列表，默认 "4 32"
#   REQUEST_RATE       请求速率，默认 4
#   MODEL_DIR          模型目录，默认 /home/honglin/models/qwen/Qwen3-8B
#   RESULT_DIR         结果目录，默认 /home/honglin/meta-infer/tmpuhorputs
#   BACKEND            推理后端，默认 qwen_tp
#   TORCH_DISTRIBUTED_TIMEOUT  NCCL 超时秒数，默认 60
set -euo pipefail

# ---- 配置 ----
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
export PATH="/home/honglin/miniconda3/envs/meta/bin:${PATH}"
export PYTHONPATH="/home/honglin/inference-agent-system:${PYTHONPATH:-}"
export META_INFER_CUDA_GRAPH=0
export META_INFER_LOG_RANK0_ONLY=1
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
# NCCL 超时：避免 worker 在 broadcast_object_list 上永久阻塞
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-60}"

MODEL_DIR="${MODEL_DIR:-/home/honglin/models/qwen/Qwen3-8B}"
BACKEND="${BACKEND:-qwen_tp}"
PORT="${PORT:-9001}"
ROUNDS="${ROUNDS:-10}"
REQUEST_RATE="${REQUEST_RATE:-4}"
STEPS="${STEPS:-4 32}"
BENCH_SCRIPT="/home/honglin/meta-infer/ref_projects/vllm/benchmarks/benchmark_serving_structured_output.py"
RESULT_DIR="${RESULT_DIR:-/home/honglin/meta-infer/tmpuhorputs}"
LOG_DIR="/home/honglin/inference-agent-system/tmp_logs"

TIMESTAMP=$(date +%m%d_%H%M)
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/server_${TIMESTAMP}.log"

# ---- 清理函数 ----
cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    # Step 1: 优雅关闭 rank0 server
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Sending SIGTERM to server PID ${SERVER_PID}..."
        kill "${SERVER_PID}" 2>/dev/null || true
        sleep 3
    fi
    # Step 2: 强制杀掉所有 openai_tp_server.py 进程（含非 rank0 worker）
    if pkill -9 -f "openai_tp_server.py" 2>/dev/null; then
        echo "Force-killed remaining openai_tp_server.py processes."
    fi
    sleep 1
    echo "Cleanup done."
}
trap cleanup EXIT INT TERM

# ---- 启动服务 ----
echo "=== agent-engine TP=4 benchmark ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES}, PORT: ${PORT}, ROUNDS: ${ROUNDS}"
echo "STEPS: ${STEPS}, BACKEND: ${BACKEND}"
echo "Server log: ${SERVER_LOG}"

GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
torchrun --nproc_per_node="${GPUS}" --master_port=$((29500+RANDOM%1000)) \
  /home/honglin/inference-agent-system/openai_tp_server.py \
  --model-dir "${MODEL_DIR}" --backend "${BACKEND}" --port "${PORT}" \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# ---- 等待服务就绪 ----
echo "Waiting for server to be ready..."
HEALTH_OK=0
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        HEALTH_OK=1
        break
    fi
    # 检查 server 是否还活着
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Server process died! Check log: ${SERVER_LOG}"
        tail -30 "${SERVER_LOG}"
        exit 1
    fi
    sleep 2
done

if [[ "${HEALTH_OK}" -eq 0 ]]; then
    echo "ERROR: Server failed to become healthy after 120s!"
    echo "Last 30 lines of server log:"
    tail -30 "${SERVER_LOG}"
    exit 1
fi

# ---- 快速连通性测试（streaming） ----
echo "=== Quick streaming smoke test ==="
SMOKE_OK=0
if curl -s --max-time 15 -X POST "http://127.0.0.1:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"你好","max_tokens":4,"temperature":0,"stream":true}' \
    | grep -q "data: \[DONE\]" 2>/dev/null; then
    echo "Smoke test PASSED"
    SMOKE_OK=1
else
    echo "WARNING: Smoke test FAILED (streaming may have issues)"
fi

# ---- 运行 benchmark ----
for S in ${STEPS}; do
    OUTFILE="${RESULT_DIR}/agent_api_s${S}_${TIMESTAMP}.json"
    echo ""
    echo "--- STEPS=${S} ---"
    python "${BENCH_SCRIPT}" \
        --backend openai --base-url "http://127.0.0.1:${PORT}" --endpoint /v1/completions \
        --model "${MODEL_DIR}" --tokenizer "${MODEL_DIR}" --trust-remote-code \
        --num-prompts "${ROUNDS}" --output-len "${S}" \
        --request-rate "${REQUEST_RATE}" --max-concurrency 1 \
        --percentile-metrics ttft,tpot --metric-percentiles 99 \
        --no-structured-output --dataset zh-default \
        --save-results --result-dir "${RESULT_DIR}" \
        --result-filename "$(basename "${OUTFILE}")" 2>&1 | grep -E "Output token|Mean TTFT|Mean TPOT|Request rate|======"
    echo ""
done

echo "=== Benchmark complete ==="
echo "Results: ${RESULT_DIR}/"
echo "Server log: ${SERVER_LOG}"
