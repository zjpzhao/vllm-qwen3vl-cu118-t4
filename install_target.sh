#!/usr/bin/env bash
set -euo pipefail

RELEASE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WHEELHOUSE="${RELEASE_DIR}/wheelhouse"
CONSTRAINTS="${RELEASE_DIR}/constraints-t4-cu118.txt"

if [[ $(uname -m) != "x86_64" ]]; then
  echo "ERROR: this release only supports x86_64." >&2
  exit 1
fi

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"ERROR: CPython 3.10 is required, got {sys.version.split()[0]}")
PY

GLIBC_VERSION=$(getconf GNU_LIBC_VERSION)
GLIBC_VERSION=${GLIBC_VERSION##* }
if [[ $(printf '%s\n' "2.28" "${GLIBC_VERSION}" | sort -V | head -n 1) != "2.28" ]]; then
  echo "ERROR: glibc >= 2.28 is required by these locally built wheels; got ${GLIBC_VERSION}." >&2
  exit 1
fi

SHM_KIB=$(df -Pk /dev/shm | awk 'NR == 2 {print $2}')
if (( SHM_KIB < 8 * 1024 * 1024 )); then
  echo "WARNING: /dev/shm is smaller than 8 GiB; vLLM workers may fail at runtime." >&2
  echo "         Ask the administrator to enlarge /dev/shm before serving." >&2
fi

case ":${LD_LIBRARY_PATH:-}:" in
  *lib/stubs*|*/stubs:*|*cuda-12.9/compat*|*cuda-12*/compat*)
    echo "ERROR: remove CUDA toolkit stubs and CUDA 12 compat paths from LD_LIBRARY_PATH." >&2
    exit 1
    ;;
esac

if [[ -f "${RELEASE_DIR}/SHA256SUMS" ]]; then
  (cd "${RELEASE_DIR}" && sha256sum -c SHA256SUMS)
fi

python -m pip install --no-index --find-links "${WHEELHOUSE}" \
  --constraint "${CONSTRAINTS}" \
  "torch==2.7.1+cu118" "torchvision==0.22.1+cu118"

python -m pip install --no-index --find-links "${WHEELHOUSE}" \
  --constraint "${CONSTRAINTS}" \
  "transformers==4.57.3" "huggingface-hub==0.34.3" \
  "qwen-vl-utils==0.0.14" "av==13.1.0"

python -m pip install --no-index --no-deps \
  "${WHEELHOUSE}/xformers-0.0.31-cp39-abi3-linux_x86_64.whl"

python -m pip install --no-index --find-links "${WHEELHOUSE}" \
  --constraint "${CONSTRAINTS}" \
  "${WHEELHOUSE}/vllm-0.11.0+torch271.cu118-cp310-cp310-linux_x86_64.whl"

python "${RELEASE_DIR}/apply_t4_xformers_hotfix.py"

if python -m pip show cupy-cuda12x >/dev/null 2>&1; then
  echo "ERROR: cupy-cuda12x must not be installed in this CUDA 11.8 environment." >&2
  exit 1
fi

python -m pip check

echo
echo "Installation complete. Before serving, run:"
echo "  export VLLM_USE_V1=1"
echo "  export VLLM_ATTENTION_BACKEND=XFORMERS"
echo "  export VLLM_T4_XFORMERS_CONTIGUOUS_PREFILL=1"
echo "  export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas"
echo "  export TRITON_CACHE_DIR=/tmp/triton-cache-cu118-sm75-xformers"
echo "  python ${RELEASE_DIR}/verify_target.py"
echo "  python ${RELEASE_DIR}/verify_qwen3vl_embedding.py --model /root/Qwen3-VL-Embedding-2B"
echo "For full multimodal validation, append: --image /path/to/test.jpg"
