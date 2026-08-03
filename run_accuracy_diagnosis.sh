#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
MODEL_PATH="${MODEL_PATH:-/root/Qwen3-VL-Embedding-2B}"
RUNS_DIR="${RUNS_DIR:-${REPO_DIR}/accuracy_runs}"
PORT="${PORT:-8000}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate vllm-t4-cu118-torch271 first." >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -e "${RUNS_DIR}/latest" ]]; then
  echo "ERROR: accuracy_runs/latest is missing; run both accuracy stages first." >&2
  exit 1
fi

RUN_DIR="${RUN_DIR:-$(readlink -f "${RUNS_DIR}/latest")}"
REFERENCE="${RUN_DIR}/precision_transformers.json"
CANDIDATE="${RUN_DIR}/precision_vllm.json"
if [[ ! -s "${REFERENCE}" || ! -s "${CANDIDATE}" ]]; then
  echo "ERROR: saved accuracy vectors are incomplete in ${RUN_DIR}." >&2
  exit 1
fi

unset CUDA_HOME
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
export TRITON_CACHE_DIR=/tmp/triton-cache-cu118-sm75-diagnosis
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/compat:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${TRITON_CACHE_DIR}"

exec > >(tee "${RUN_DIR}/accuracy_diagnosis.log") 2>&1
echo "RUN_DIR=${RUN_DIR}"
if [[ -s "${RUN_DIR}/diagnose_mrope.json" ]] && python - \
  "${RUN_DIR}/diagnose_mrope.json" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["passed"] else 1)
PY
then
  echo "Stage 1/2: reusing the existing passing MRoPE result"
else
  echo "Stage 1/2: MRoPE Triton versus pure PyTorch"
  if ! python "${REPO_DIR}/diagnose_accuracy_mismatch.py" mrope \
    --model "${MODEL_PATH}" \
    --output "${RUN_DIR}/diagnose_mrope.json" \
    2>&1 | tee "${RUN_DIR}/diagnose_mrope.log"; then
    echo "STOP: MRoPE failed. The server remains running for inspection."
    exit 2
  fi
fi

listener_pid() {
  ss -H -lntp "sport = :${PORT}" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p; t found; b; :found q'
}

listener_pid="$(listener_pid || true)"
if [[ -n "${listener_pid}" ]]; then
  echo "Stopping vLLM PID ${listener_pid} before loading Transformers..."
  kill -TERM "${listener_pid}"
  for _ in $(seq 1 60); do
    remaining_listener="$(listener_pid || true)"
    process_state="$(ps -o stat= -p "${listener_pid}" 2>/dev/null \
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
  process_state="$(ps -o stat= -p "${listener_pid}" 2>/dev/null \
    | tr -d '[:space:]' || true)"
  if [[ "${process_state}" == Z* ]]; then
    echo "Listener is gone; PID ${listener_pid} is a harmless zombie awaiting parent reap."
  else
    echo "Listener on port ${PORT} stopped."
  fi
  sleep 3
fi

echo "Stage 2/2: scan Transformers token hidden states"
python "${REPO_DIR}/diagnose_accuracy_mismatch.py" hidden-scan \
  --model "${MODEL_PATH}" \
  --reference "${REFERENCE}" \
  --candidate "${CANDIDATE}" \
  --output "${RUN_DIR}/diagnose_hidden_scan.json" \
  2>&1 | tee "${RUN_DIR}/diagnose_hidden_scan.log"

echo "DONE: ${RUN_DIR}"
echo "Send back these two files:"
echo "  ${RUN_DIR}/diagnose_mrope.json"
echo "  ${RUN_DIR}/diagnose_hidden_scan.json"
