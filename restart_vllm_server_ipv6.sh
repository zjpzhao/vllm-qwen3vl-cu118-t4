#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/Qwen3-VL-Embedding-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-Embedding-2B}"
PORT="${PORT:-8000}"
REPO_DIR="${REPO_DIR:-/root/vllm-qwen3vl-cu118-t4}"
LOG_FILE="${LOG_FILE:-${REPO_DIR}/vllm_server.log}"
PID_FILE="${PID_FILE:-${REPO_DIR}/vllm_server.pid}"
IMAGE_LIMIT="${IMAGE_LIMIT:-1}"
VIDEO_LIMIT="${VIDEO_LIMIT:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MM_LIMITS="{\"image\":${IMAGE_LIMIT},\"video\":${VIDEO_LIMIT}}"

if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
  echo "ERROR: PORT must be an integer between 1 and 65535." >&2
  exit 1
fi
if [[ ! "${IMAGE_LIMIT}" =~ ^[0-9]+$ || ! "${VIDEO_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: IMAGE_LIMIT and VIDEO_LIMIT must be non-negative integers." >&2
  exit 1
fi
for setting in MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS; do
  value="${!setting}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${setting} must be a positive integer." >&2
    exit 1
  fi
done
if ((MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN)); then
  echo "ERROR: MAX_NUM_BATCHED_TOKENS must be >= MAX_MODEL_LEN because chunked prefill is disabled." >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate vllm-t4-cu118-torch271 first." >&2
  exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm is not available in the active environment." >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

cd "${REPO_DIR}"

listener_pid() {
  ss -H -lntp "sport = :${PORT}" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p; t found; b; :found q'
}

current_pid="$(listener_pid || true)"
if [[ -n "${current_pid}" ]]; then
  echo "Stopping PID ${current_pid} on port ${PORT}..."
  kill -TERM "${current_pid}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    remaining_listener="$(listener_pid || true)"
    process_state="$(ps -o stat= -p "${current_pid}" 2>/dev/null \
      | tr -d '[:space:]' || true)"
    if [[ -z "${remaining_listener}" || "${process_state}" == Z* ]]; then
      break
    fi
    sleep 1
  done
  remaining_listener="$(listener_pid || true)"
  if [[ -n "${remaining_listener}" ]]; then
    echo "ERROR: port ${PORT} is still listened on by PID ${remaining_listener}." >&2
    exit 1
  fi
  process_state="$(ps -o stat= -p "${current_pid}" 2>/dev/null \
    | tr -d '[:space:]' || true)"
  if [[ "${process_state}" == Z* ]]; then
    echo "Listener stopped; PID ${current_pid} is a harmless zombie awaiting parent reap."
  else
    echo "Listener on port ${PORT} stopped."
  fi
fi

unset CUDA_HOME
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_T4_XFORMERS_CONTIGUOUS_PREFILL=1
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
export TRITON_CACHE_DIR=/tmp/triton-cache-cu118-sm75-xformers
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/compat:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${TRITON_CACHE_DIR}"

echo "Starting ${SERVED_MODEL_NAME} on [::]:${PORT}..."
nohup vllm serve "${MODEL_PATH}" \
  --host :: \
  --port "${PORT}" \
  --disable-uvicorn-access-log \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --runner pooling \
  --convert embed \
  --dtype half \
  --kv-cache-dtype auto \
  --enforce-eager \
  -O0 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.80 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt "${MM_LIMITS}" \
  --trust-remote-code \
  >"${LOG_FILE}" 2>&1 &

server_pid=$!
echo "${server_pid}" >"${PID_FILE}"
sleep 2
if ! kill -0 "${server_pid}" 2>/dev/null; then
  echo "ERROR: vLLM exited during startup; inspect ${LOG_FILE}." >&2
  tail -n 80 "${LOG_FILE}" >&2 || true
  exit 1
fi

echo "Started PID ${server_pid}; log: ${LOG_FILE}"
echo "Multimodal limits: ${MM_LIMITS}"
echo "Scheduler: max_model_len=${MAX_MODEL_LEN}, max_num_seqs=${MAX_NUM_SEQS}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "Wait for model loading, then run:"
echo "  curl --noproxy '*' -g http://[::1]:${PORT}/health"
echo "  ss -lntp | grep ':${PORT}'"
