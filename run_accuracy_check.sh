#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
MODEL_PATH="${MODEL_PATH:-/root/Qwen3-VL-Embedding-2B}"
IMAGE_PATH="${IMAGE_PATH:-${REPO_DIR}/accuracy_inputs/qwen_vl_demo.jpeg}"
RUNS_DIR="${RUNS_DIR:-${REPO_DIR}/accuracy_runs}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-Embedding-2B}"

usage() {
  cat <<'EOF'
Usage:
  ./run_accuracy_check.sh transformers
  ./run_accuracy_check.sh vllm

Optional environment variables:
  MODEL_PATH, IMAGE_PATH, RUN_DIR, RUNS_DIR, PORT, SERVED_MODEL_NAME,
  T4_EXECUTION_MODE, CUDAGRAPH_CAPTURE_SIZES_JSON

The transformers stage stops vLLM, creates a timestamped RUN_DIR, and writes
the reference vectors. The vllm stage reuses accuracy_runs/latest, starts the
visual service, writes candidate vectors, and produces the comparison report.
EOF
}

require_common_inputs() {
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: activate vllm-t4-cu118-torch271 first." >&2
    exit 1
  fi
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
  fi
  if [[ ! -s "${IMAGE_PATH}" ]]; then
    echo "ERROR: test image does not exist or is empty: ${IMAGE_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${MODEL_PATH}/scripts/qwen3_vl_embedding.py" ]]; then
    echo "ERROR: official Transformers wrapper is missing:" >&2
    echo "       ${MODEL_PATH}/scripts/qwen3_vl_embedding.py" >&2
    exit 1
  fi
  if [[ ! -f "${REPO_DIR}/compare_vllm_transformers.py" ]]; then
    echo "ERROR: comparison script is missing from ${REPO_DIR}." >&2
    exit 1
  fi
}

listener_pid() {
  ss -H -lntp "sport = :${PORT}" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p; t found; b; :found q'
}

stop_vllm() {
  local pid
  pid="$(listener_pid)"
  if [[ -z "${pid}" ]]; then
    echo "No listener on port ${PORT}."
    return
  fi

  echo "Stopping PID ${pid} on port ${PORT}..."
  kill -TERM "${pid}"
  for _ in $(seq 1 60); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Stopped PID ${pid}."
      return
    fi
    sleep 1
  done
  echo "ERROR: PID ${pid} did not stop after 60 seconds." >&2
  exit 1
}

prepare_runtime_environment() {
  unset CUDA_HOME
  export LD_LIBRARY_PATH="/usr/local/cuda-11.8/compat:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
}

run_transformers() {
  require_common_inputs
  local run_dir
  run_dir="${RUN_DIR:-${RUNS_DIR}/$(date +%Y%m%d_%H%M%S)}"
  mkdir -p "${run_dir}" "${RUNS_DIR}"
  run_dir="$(cd "${run_dir}" && pwd)"
  ln -sfn "$(basename "${run_dir}")" "${RUNS_DIR}/latest"

  exec > >(tee "${run_dir}/transformers_stage.log") 2>&1
  echo "RUN_DIR=${run_dir}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "IMAGE_PATH=${IMAGE_PATH}"

  stop_vllm
  prepare_runtime_environment
  nvidia-smi | tee "${run_dir}/nvidia_smi_before_transformers.log"

  python "${REPO_DIR}/compare_vllm_transformers.py" reference \
    --model "${MODEL_PATH}" \
    --image "${IMAGE_PATH}" \
    --max-length 2048 \
    --output "${run_dir}/precision_transformers.json" \
    2>&1 | tee "${run_dir}/precision_transformers.log"

  test -s "${run_dir}/precision_transformers.json"
  echo "TRANSFORMERS PASS: ${run_dir}/precision_transformers.json"
  echo "Next command: ./run_accuracy_check.sh vllm"
}

resolve_run_dir() {
  if [[ -n "${RUN_DIR:-}" ]]; then
    cd "${RUN_DIR}" && pwd
    return
  fi
  if [[ ! -e "${RUNS_DIR}/latest" ]]; then
    echo "ERROR: no latest accuracy run; execute the transformers stage first." >&2
    exit 1
  fi
  readlink -f "${RUNS_DIR}/latest"
}

wait_for_server() {
  local health_url="http://[::1]:${PORT}/health"
  for _ in $(seq 1 120); do
    if curl -fsS --noproxy '*' -g "${health_url}" >/dev/null; then
      echo "vLLM health check passed: ${health_url}"
      return
    fi
    sleep 2
  done
  echo "ERROR: vLLM did not become healthy within 240 seconds." >&2
  exit 1
}

run_vllm_and_compare() {
  require_common_inputs
  local run_dir
  run_dir="$(resolve_run_dir)"
  if [[ ! -s "${run_dir}/precision_transformers.json" ]]; then
    echo "ERROR: Transformers reference is missing from ${run_dir}." >&2
    exit 1
  fi

  exec > >(tee "${run_dir}/vllm_stage.log") 2>&1
  echo "RUN_DIR=${run_dir}"
  echo "T4_EXECUTION_MODE=${T4_EXECUTION_MODE:-eager}"
  prepare_runtime_environment

  IMAGE_LIMIT=1 \
  VIDEO_LIMIT=0 \
  MODEL_PATH="${MODEL_PATH}" \
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  PORT="${PORT}" \
  REPO_DIR="${REPO_DIR}" \
  LOG_FILE="${run_dir}/vllm_server.log" \
  PID_FILE="${run_dir}/vllm_server.pid" \
    "${REPO_DIR}/restart_vllm_server_ipv6.sh"

  wait_for_server
  python "${REPO_DIR}/compare_vllm_transformers.py" vllm \
    --endpoint "http://[::1]:${PORT}/v1/embeddings" \
    --model-name "${SERVED_MODEL_NAME}" \
    --image "${IMAGE_PATH}" \
    --output "${run_dir}/precision_vllm.json" \
    2>&1 | tee "${run_dir}/precision_vllm.log"

  python "${REPO_DIR}/compare_vllm_transformers.py" compare \
    --reference "${run_dir}/precision_transformers.json" \
    --candidate "${run_dir}/precision_vllm.json" \
    --report "${run_dir}/precision_report.json" \
    2>&1 | tee "${run_dir}/precision_compare.log"

  echo "ACCURACY PASS: ${run_dir}/precision_report.json"
  find "${run_dir}" -maxdepth 1 -type f -printf '%f\n' | sort
}

case "${STAGE}" in
  transformers)
    run_transformers
    ;;
  vllm)
    run_vllm_and_compare
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
