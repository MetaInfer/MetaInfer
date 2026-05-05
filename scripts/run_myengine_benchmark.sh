#!/usr/bin/env bash
# 压测 meta-infer OpenAI 服务（需先启动 scripts/start_tp_infer_service.sh）
#
# 环境变量（可选）:
#   PORT / HOST           服务地址，默认 9000 / 127.0.0.1
#   TP_SIZE               仅用于展示当前实验配置（默认 4）
#   CUDA_VISIBLE_DEVICES  与服务端一致时便于复现实验
#   ROUNDS                请求条数，对应 benchmark 的 --num-prompts（默认 25）
#   STEPS                 每请求最大输出 token 数，对应 --output-len / decode 步数上限（默认 32）
#   REQUEST_RATE          默认 4
#   MAX_CONCURRENCY       默认 1（TP 服务建议串行，避免 collective 乱序）
#   DATASET               zh-default | sharegpt-json（默认 zh-default，即 5 句 prompt 循环）
#   SHAREGPT_JSON         当 DATASET=sharegpt-json 时必填，ShareGPT 格式 JSON 路径
#   RESULT_DIR / RESULT_NAME
#   PERCENTILE_METRICS    默认 ttft,tpot,e2el
#   METRIC_PERCENTILES    默认 99
#
# 不指定数据集文件时: 使用内置 5 条中文句（与 tests 一致），无需 SHAREGPT。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_SCRIPT="${ROOT_DIR}/ref_projects/vllm/benchmarks/benchmark_serving_structured_output.py"

QWEN_MODEL="/home/honglin/models/qwen/Qwen3-8B"
DSV2_MODEL="/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat"

MODEL_KEY="${1:-dsv2}" # dsv2 | qwen
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-9000}"
HOST="${HOST:-127.0.0.1}"
ROUNDS="${ROUNDS:-25}"
STEPS="${STEPS:-32}"
NUM_PROMPTS="${NUM_PROMPTS:-$ROUNDS}"
OUTPUT_LEN="${OUTPUT_LEN:-$STEPS}"
REQUEST_RATE="${REQUEST_RATE:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
DATASET="${DATASET:-zh-default}"
SHAREGPT_JSON="${SHAREGPT_JSON:-}"
RESULT_DIR="${RESULT_DIR:-/home/honglin/meta-infer/tests}"
PERCENTILE_METRICS="${PERCENTILE_METRICS:-ttft,tpot,e2el}"
METRIC_PERCENTILES="${METRIC_PERCENTILES:-99}"

case "$MODEL_KEY" in
dsv2)
  MODEL_DIR="$DSV2_MODEL"
  RESULT_NAME="${RESULT_NAME:-myengine_deepseekv2_tp4.json}"
  ;;
qwen)
  MODEL_DIR="$QWEN_MODEL"
  RESULT_NAME="${RESULT_NAME:-myengine_qwen_tp4.json}"
  ;;
*)
  echo "Unknown model key: $MODEL_KEY (expected: dsv2 or qwen)"
  exit 1
  ;;
esac

if [[ ! -f "$BENCH_SCRIPT" ]]; then
  echo "Benchmark script not found: $BENCH_SCRIPT"
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR"
  exit 1
fi

if [[ "$DATASET" == "sharegpt-json" ]]; then
  if [[ -z "$SHAREGPT_JSON" || ! -f "$SHAREGPT_JSON" ]]; then
    echo "DATASET=sharegpt-json 需要有效 SHAREGPT_JSON 文件路径"
    exit 1
  fi
fi

mkdir -p "$RESULT_DIR"

echo "[bench] model=$MODEL_KEY rounds(num_prompts)=$NUM_PROMPTS steps(output_len)=$OUTPUT_LEN"
echo "[bench] dataset=$DATASET service=http://${HOST}:${PORT}"
echo "[bench] tp_size=$TP_SIZE cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<inherit>}"

echo "[bench] checking service health at http://${HOST}:${PORT}/health"
python - <<PY
import json, sys, urllib.request
url = "http://${HOST}:${PORT}/health"
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        print("[bench] health:", r.status, r.read().decode("utf-8"))
except Exception as e:
    print(f"[bench] service health check failed: {e}")
    sys.exit(1)
PY

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

BENCH_EXTRA=()
if [[ "$DATASET" == "sharegpt-json" ]]; then
  BENCH_EXTRA=(--dataset sharegpt-json --sharegpt-json "$SHAREGPT_JSON")
else
  BENCH_EXTRA=(--dataset zh-default)
fi

echo "[bench] running benchmark..."
python "$BENCH_SCRIPT" \
  --backend openai \
  --base-url "http://${HOST}:${PORT}" \
  --endpoint /v1/completions \
  --model "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR" \
  --trust-remote-code \
  --num-prompts "$NUM_PROMPTS" \
  --output-len "$OUTPUT_LEN" \
  --request-rate "$REQUEST_RATE" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --percentile-metrics "$PERCENTILE_METRICS" \
  --metric-percentiles "$METRIC_PERCENTILES" \
  "${BENCH_EXTRA[@]}" \
  --no-structured-output \
  --save-results \
  --result-dir "$RESULT_DIR" \
  --result-filename "$RESULT_NAME"

echo "[bench] done: ${RESULT_DIR}/${RESULT_NAME}"
