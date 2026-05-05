#!/usr/bin/env bash
# 顺序跑 vLLM TP 与 meta-infer TP 的 serving benchmark，并输出对比表（同一 GPU、同一参数）。
# 用法:
#   bash scripts/run_compare_metainfer_vllm.sh dsv2
# 环境变量（与 scripts/run_myengine_benchmark.sh 对齐）:
#   TP_SIZE, CUDA_VISIBLE_DEVICES
#   ROUNDS, STEPS, REQUEST_RATE, MAX_CONCURRENCY
#   DATASET (zh-default | sharegpt-json), SHAREGPT_JSON
#   VLLM_PORT (默认 8001), MY_PORT (默认 9000)
#   VLLM_BIN: vLLM 可执行，默认 vllm（需已在 PATH）
#   MODEL_DIR: 覆盖模型路径（可选）
#   SKIP_VLLM=1 只跑 meta-infer；SKIP_MYENGINE=1 只跑 vLLM
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_SCRIPT="${ROOT_DIR}/ref_projects/vllm/benchmarks/benchmark_serving_structured_output.py"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}"

DSV2_MODEL="/data/xinference/cache/deepseek-v2-chat-pytorch-16b"
QWEN_MODEL="/data/xinference/cache/Qwen3-8B"

MODEL_KEY="${1:-dsv2}"
TP_SIZE="${TP_SIZE:-4}"
build_cuda_visible_devices() {
  local n="$1"
  local out=""
  local i
  for ((i = 0; i < n; i++)); do
    if [[ -z "$out" ]]; then
      out="$i"
    else
      out="${out},${i}"
    fi
  done
  echo "$out"
}
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(build_cuda_visible_devices "$TP_SIZE")}"
export CUDA_VISIBLE_DEVICES

ROUNDS="${ROUNDS:-25}"
STEPS="${STEPS:-32}"
REQUEST_RATE="${REQUEST_RATE:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
DATASET="${DATASET:-zh-default}"
SHAREGPT_JSON="${SHAREGPT_JSON:-}"
VLLM_PORT="${VLLM_PORT:-8001}"
MY_PORT="${MY_PORT:-9000}"
VLLM_BIN="${VLLM_BIN:-vllm}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/tests}"
PERCENTILE_METRICS="${PERCENTILE_METRICS:-ttft,tpot,e2el}"
METRIC_PERCENTILES="${METRIC_PERCENTILES:-99}"

case "$MODEL_KEY" in
dsv2) MODEL_DIR="${MODEL_DIR:-$DSV2_MODEL}" ;;
qwen) MODEL_DIR="${MODEL_DIR:-$QWEN_MODEL}" ;;
*) echo "usage: $0 [dsv2|qwen]"; exit 1 ;;
esac

if [[ ! -d "$MODEL_DIR" || ! -f "$BENCH_SCRIPT" ]]; then
  echo "Missing model dir or benchmark script: MODEL_DIR=$MODEL_DIR"
  exit 1
fi

bench_supports_dataset() {
  local ds="$1"
  python - "$BENCH_SCRIPT" "$ds" <<'PY'
import ast, pathlib, re, sys
script = pathlib.Path(sys.argv[1])
target = sys.argv[2]
text = script.read_text(encoding="utf-8")
# Parse --dataset choices statically to avoid importing benchmark deps.
m = re.search(r'--dataset".*?choices\s*=\s*(\[[^\]]*\])', text, re.S)
if not m:
    sys.exit(2)
try:
    choices = ast.literal_eval(m.group(1))
except Exception:
    sys.exit(2)
sys.exit(0 if target in choices else 1)
PY
}

if [[ "$DATASET" != "zh-default" && "$DATASET" != "sharegpt-json" ]]; then
  echo "DATASET 仅支持 zh-default | sharegpt-json，当前: $DATASET"
  exit 1
fi

if ! bench_supports_dataset "$DATASET"; then
  echo "当前 benchmark 脚本不支持 DATASET=$DATASET，请先升级/同步:"
  echo "  $BENCH_SCRIPT"
  exit 1
fi

if [[ "$DATASET" == "sharegpt-json" && (-z "$SHAREGPT_JSON" || ! -f "$SHAREGPT_JSON") ]]; then
  echo "DATASET=sharegpt-json 需要 SHAREGPT_JSON 指向有效文件"
  exit 1
fi

mkdir -p "$RESULT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_VLLM="${RESULT_DIR}/vllm_${MODEL_KEY}_tp${TP_SIZE}_${TS}.json"
OUT_MY="${RESULT_DIR}/myengine_${MODEL_KEY}_tp${TP_SIZE}_${TS}.json"
SUMMARY="${RESULT_DIR}/compare_${MODEL_KEY}_tp${TP_SIZE}_${TS}.txt"

BENCH_COMMON=(
  --model "$MODEL_DIR"
  --tokenizer "$MODEL_DIR"
  --trust-remote-code
  --num-prompts "$ROUNDS"
  --output-len "$STEPS"
  --request-rate "$REQUEST_RATE"
  --max-concurrency "$MAX_CONCURRENCY"
  --percentile-metrics "$PERCENTILE_METRICS"
  --metric-percentiles "$METRIC_PERCENTILES"
  --no-structured-output
  --save-results
  --result-dir "$RESULT_DIR"
)
if [[ "$DATASET" == "sharegpt-json" ]]; then
  BENCH_DS=(--dataset sharegpt-json --sharegpt-json "$SHAREGPT_JSON")
else
  BENCH_DS=(--dataset zh-default)
fi

wait_http() {
  local name="$1" ; shift
  local max="${1:-120}"; shift
  local urls=("$@")
  local i=0
  while (( i < max )); do
    for url in "${urls[@]}"; do
      if python - <<PY
import urllib.request, sys
try:
    urllib.request.urlopen("$url", timeout=2)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
      then
        echo "[ok] $name (probe $url)"
        return 0
      fi
    done
    sleep 2
    ((i+=2)) || true
  done
  echo "[fail] $name not ready, tried: ${urls[*]}"
  return 1
}

kill_group() {
  local pidfile="$1"
  if [[ -f "$pidfile" ]]; then
    local p
    p="$(cat "$pidfile" || true)"
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null || true
      wait "$p" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
}

run_bench() {
  local backend="$1" base_url="$2" outname="$3"
  python "$BENCH_SCRIPT" \
    --backend "$backend" \
    --base-url "$base_url" \
    --endpoint /v1/completions \
    "${BENCH_COMMON[@]}" \
    "${BENCH_DS[@]}" \
    --result-filename "$(basename "$outname")"
}

extract_metrics_py() {
  local fpath="$1"
  python - "$fpath" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
def g(k, default=None):
    return d.get(k, default)
keys = [
    "request_throughput", "output_throughput",
    "p99_ttft_ms", "p99_tpot_ms", "p99_e2el_ms",
    "completed", "duration",
]
out = {k: g(k) for k in keys}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
}

VLLM_PIDF="${RESULT_DIR}/.vllm_serve_pid_${VLLM_PORT}"
MY_PIDF="${RESULT_DIR}/.metainfer_serve_pid_${MY_PORT}"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== compare: model=$MODEL_KEY tp=$TP_SIZE rounds=$ROUNDS steps=$STEPS dataset=$DATASET ==="
echo "vLLM -> http://127.0.0.1:${VLLM_PORT}  |  meta-infer -> http://127.0.0.1:${MY_PORT}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

if [[ "${SKIP_VLLM:-0}" != "1" ]]; then
  kill_group "$VLLM_PIDF"
  echo "[vllm] starting..."
  # shellcheck disable=SC2086
  nohup "$VLLM_BIN" serve "$MODEL_DIR" \
    --tensor-parallel-size "$TP_SIZE" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    >"${RESULT_DIR}/vllm_serve_${TS}.log" 2>&1 &
  echo $! > "$VLLM_PIDF"
  wait_http "vllm" 300 \
    "http://127.0.0.1:${VLLM_PORT}/health" \
    "http://127.0.0.1:${VLLM_PORT}/v1/models" || { echo "vLLM 启动失败，见 ${RESULT_DIR}/vllm_serve_${TS}.log"; exit 1; }
  run_bench vllm "http://127.0.0.1:${VLLM_PORT}" "$OUT_VLLM"
  kill_group "$VLLM_PIDF"
  sleep 3
else
  echo "[skip] vLLM"
fi

if [[ "${SKIP_MYENGINE:-0}" != "1" ]]; then
  kill_group "$MY_PIDF"
  echo "[myengine] starting torchrun openai_tp_server..."
  cd "$ROOT_DIR"
  export MAX_NEW_TOKENS_CAP="${MAX_NEW_TOKENS_CAP:-$(( STEPS + 32 ))}"
  # shellcheck disable=SC2086
  nohup torchrun --nproc_per_node="$TP_SIZE" openai_tp_server.py \
    --model-dir "$MODEL_DIR" \
    --backend tp \
    --host 0.0.0.0 \
    --port "$MY_PORT" \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --max-new-tokens-cap "$MAX_NEW_TOKENS_CAP" \
    >"${RESULT_DIR}/metainfer_serve_${TS}.log" 2>&1 &
  echo $! > "$MY_PIDF"
  wait_http "meta-infer" 600 "http://127.0.0.1:${MY_PORT}/health" || { echo "meta-infer 启动失败，见 ${RESULT_DIR}/metainfer_serve_${TS}.log"; exit 1; }
  run_bench openai "http://127.0.0.1:${MY_PORT}" "$OUT_MY"
  kill_group "$MY_PIDF"
else
  echo "[skip] meta-infer"
fi

# 从最新 result 文件取基名（脚本已按 result-filename 写入 RESULT_DIR）
{
  echo "========== 对比表 (${TS}) =========="
  echo "ROUNDS=$ROUNDS STEPS=$STEPS RPS=$REQUEST_RATE max_conc=$MAX_CONCURRENCY"
  echo
  if [[ -f "$OUT_VLLM" && "${SKIP_VLLM:-0}" != "1" ]]; then
    echo "--- vLLM ---"
    extract_metrics_py "$OUT_VLLM" || true
  fi
  if [[ -f "$OUT_MY" && "${SKIP_MYENGINE:-0}" != "1" ]]; then
    echo "--- meta-infer ---"
    extract_metrics_py "$OUT_MY" || true
  fi
} | tee "$SUMMARY"

echo
echo "详细 JSON: $OUT_VLLM $OUT_MY"
echo "摘要: $SUMMARY"
