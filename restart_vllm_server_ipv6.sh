#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/Qwen3-VL-Embedding-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-Embedding-2B}"
PORT="${PORT:-8000}"
REPO_DIR="${REPO_DIR:-/root/vllm-qwen3vl-cu118-t4}"
LOG_FILE="${LOG_FILE:-${REPO_DIR}/vllm_server.log}"
PID_FILE="${PID_FILE:-${REPO_DIR}/vllm_server.pid}"

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

current_pid="$(
  ss -H -lntp "sport = :${PORT}" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p; t found; b; :found q'
)"
if [[ -n "${current_pid}" ]]; then
  echo "Stopping PID ${current_pid} on port ${PORT}..."
  kill -TERM "${current_pid}"
  for _ in $(seq 1 30); do
    if ! kill -0 "${current_pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "${current_pid}" 2>/dev/null; then
    echo "ERROR: PID ${current_pid} did not stop after 30 seconds." >&2
    exit 1
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
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
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
echo "Wait for model loading, then run:"
echo "  curl --noproxy '*' -g http://[::1]:${PORT}/health"
echo "  ss -lntp | grep ':${PORT}'"
