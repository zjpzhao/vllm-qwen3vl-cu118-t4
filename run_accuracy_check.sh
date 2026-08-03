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
OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
ACCURACY_CLEANUP_DONE=0
ACCURACY_CLEANUP_STATUS=0
ACCURACY_PID_FILE=""

usage() {
  cat <<'EOF'
Usage:
  ./run_accuracy_check.sh transformers
  ./run_accuracy_check.sh vllm

Optional environment variables:
  MODEL_PATH, IMAGE_PATH, RUN_DIR, RUNS_DIR, PORT, SERVED_MODEL_NAME,
  OMP_NUM_THREADS, T4_EXECUTION_MODE, CUDAGRAPH_CAPTURE_SIZES_JSON

The transformers stage stops vLLM, creates a timestamped RUN_DIR, and writes
the reference vectors. The vllm stage reuses accuracy_runs/latest, starts the
visual service, writes candidate vectors, and produces the comparison report.
EOF
}

require_common_inputs() {
  if [[ ! "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: OMP_NUM_THREADS must be a positive integer." >&2
    exit 1
  fi
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

cleanup_accuracy_vllm() {
  if ((ACCURACY_CLEANUP_DONE)); then
    return
  fi
  ACCURACY_CLEANUP_DONE=1

  echo "Stopping all vLLM processes created or left by the accuracy run..."
  local tracked_pid=""
  if [[ -s "${ACCURACY_PID_FILE}" ]]; then
    read -r tracked_pid <"${ACCURACY_PID_FILE}" || true
  fi
  if [[ "${tracked_pid}" =~ ^[0-9]+$ ]]; then
    kill -TERM "${tracked_pid}" 2>/dev/null || true
  fi
  pkill -TERM -f \
    'vllm\.entrypoints\.openai\.api_server|EngineCore|multiprocessing\.spawn.*vllm' \
    2>/dev/null || true
  sleep 3
  if [[ "${tracked_pid}" =~ ^[0-9]+$ ]]; then
    kill -KILL "${tracked_pid}" 2>/dev/null || true
  fi
  pkill -KILL -f \
    'vllm\.entrypoints\.openai\.api_server|EngineCore|multiprocessing\.spawn.*vllm' \
    2>/dev/null || true

  if [[ -n "${ACCURACY_PID_FILE}" ]]; then
    rm -f -- "${ACCURACY_PID_FILE}"
  fi

  local remaining_pid
  remaining_pid="$(listener_pid || true)"
  if [[ -n "${remaining_pid}" ]]; then
    kill -KILL "${remaining_pid}" 2>/dev/null || true
    sleep 1
    remaining_pid="$(listener_pid || true)"
  fi
  if [[ -n "${remaining_pid}" ]]; then
    echo "ERROR: vLLM cleanup failed; PID ${remaining_pid} still listens on port ${PORT}." >&2
    ACCURACY_CLEANUP_STATUS=1
  else
    echo "vLLM cleanup passed: no listener remains on port ${PORT}."
  fi
}

prepare_runtime_environment() {
  unset CUDA_HOME
  export OMP_NUM_THREADS
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
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

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
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  ACCURACY_PID_FILE="${run_dir}/vllm_server.pid"
  trap cleanup_accuracy_vllm EXIT
  trap 'exit 130' INT TERM
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

  local compare_status=0
  if python "${REPO_DIR}/compare_vllm_transformers.py" compare \
      --reference "${run_dir}/precision_transformers.json" \
      --candidate "${run_dir}/precision_vllm.json" \
      --report "${run_dir}/precision_report.json" \
      2>&1 | tee "${run_dir}/precision_compare.log"; then
    compare_status=0
  else
    compare_status=$?
  fi

  cleanup_accuracy_vllm
  trap - EXIT INT TERM
  find "${run_dir}" -maxdepth 1 -type f -printf '%f\n' | sort

  python - "${run_dir}/precision_report.json" \
    "${T4_EXECUTION_MODE:-eager}" "${ACCURACY_CLEANUP_STATUS}" <<'PY'
import json
import sys

report_path, execution_mode, cleanup_status_raw = sys.argv[1:]
cleanup_passed = cleanup_status_raw == "0"
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)

summary = report["summary"]
thresholds = report["thresholds"]
passed = bool(report["passed"])
overall_passed = passed and cleanup_passed

print()
print("=" * 72)
print("最终测试结论：" + ("PASS" if overall_passed else "FAIL"))
print("精度判定：" + ("PASS" if passed else "FAIL"))
print(f"执行模式：{execution_mode}")
print("服务清理：" + ("PASS（vLLM 已退出）" if cleanup_passed else "FAIL"))
print(
    "最低同输入 cosine："
    f"{summary['min_same_input_cosine']:.10f} "
    f"(要求 >= {thresholds['min_cosine']:.10f})"
)
print(
    "Pairwise similarity MAE："
    f"{summary['pairwise_similarity_mae']:.10f} "
    f"(要求 <= {thresholds['max_similarity_mae']:.10f})"
)
print(
    "检索 Top-1 一致率："
    f"{summary['retrieval_top1_agreement'] * 100:.2f}%"
)
if passed and cleanup_passed:
    print(
        "结论：vLLM 与 Transformers 精度对齐，当前执行模式未发现可观测的精度回归，"
        "且测试服务已完全退出，可以继续稳定性和性能验证。"
    )
elif passed:
    print("结论：精度对齐，但 vLLM 服务清理失败；本轮整体测试不通过。")
else:
    failures = "; ".join(report.get("failures") or ["unknown failure"])
    print(f"结论：精度验收未通过，不应进入性能验证；失败原因：{failures}")
print(f"报告：{report_path}")
print("=" * 72)
PY

  if ((compare_status != 0)); then
    return "${compare_status}"
  fi
  if ((ACCURACY_CLEANUP_STATUS != 0)); then
    return 1
  fi
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
