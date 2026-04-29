#!/usr/bin/env bash
# 启动 meta-infer TP + OpenAI 兼容 HTTP 服务
#
# 若不指定 DATASET_PATH（或设为空）: 不检查数据集文件，压测请用 run_myengine_benchmark.sh 的 zh-default。
# 环境变量: TP_SIZE, PORT, HOST, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, BACKEND,
#           MAX_NEW_TOKENS_CAP（应 >= 压测 STEPS）
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 可选本地 ShareGPT: /data/dataset/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json
QWEN_MODEL="/home/honglin/models/qwen/Qwen3-8B"
DSV2_MODEL="/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat"

MODEL_KEY="${1:-qwen}"  # qwen | dsv2
TP_SIZE="${TP_SIZE:-4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# 默认不依赖数据集文件；需要时可: DATASET_PATH=$DEFAULT_DATASET 或显式传路径
DATASET_PATH="${DATASET_PATH:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
BACKEND="${BACKEND:-tp}" # tp | qwen_tp | deepseek_tp | hf
MAX_NEW_TOKENS_CAP="${MAX_NEW_TOKENS_CAP:-512}"

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

case "$MODEL_KEY" in
qwen)
  MODEL_DIR="$QWEN_MODEL"
  ;;
dsv2)
  MODEL_DIR="$DSV2_MODEL"
  ;;
*)
  echo "Unknown model key: $MODEL_KEY (expected: qwen or dsv2)"
  exit 1
  ;;
esac

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR"
  exit 1
fi

if [[ -n "${DATASET_PATH}" && ! -f "$DATASET_PATH" ]]; then
  echo "Dataset file not found: $DATASET_PATH (set DATASET_PATH= to skip)"
  exit 1
fi

export META_INFER_MODEL_DIR="$MODEL_DIR"
if [[ -n "${DATASET_PATH}" ]]; then
  export META_INFER_DATASET_PATH="$DATASET_PATH"
else
  unset META_INFER_DATASET_PATH 2>/dev/null || true
fi
export META_INFER_LOG_RANK0_ONLY="${META_INFER_LOG_RANK0_ONLY:-1}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(build_cuda_visible_devices "$TP_SIZE")"
fi
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "[meta-infer] model_key=$MODEL_KEY"
echo "[meta-infer] model_dir=$MODEL_DIR"
echo "[meta-infer] tp_size=$TP_SIZE"
echo "[meta-infer] host=$HOST port=$PORT"
echo "[meta-infer] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[meta-infer] dataset_path=${DATASET_PATH:-<none>}"
echo "[meta-infer] backend=$BACKEND max_new_tokens_cap=$MAX_NEW_TOKENS_CAP"
echo "[meta-infer] TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo
echo "[meta-infer] starting OpenAI-compatible TP service..."

cd "$ROOT_DIR"
torchrun --nproc_per_node="$TP_SIZE" openai_tp_server.py \
  --model-dir "$MODEL_DIR" \
  --backend "$BACKEND" \
  --host "$HOST" \
  --port "$PORT" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-new-tokens-cap "$MAX_NEW_TOKENS_CAP"
