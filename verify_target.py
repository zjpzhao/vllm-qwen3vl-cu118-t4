#!/usr/bin/env python3
import importlib
import importlib.metadata
import json
import os
import pathlib
import platform
import site
import subprocess
import sys

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")

if sys.version_info[:2] != (3, 10):
    raise RuntimeError(f"CPython 3.10 is required, got {sys.version.split()[0]}")

glibc_name, glibc_version = platform.libc_ver()
if glibc_name != "glibc" or tuple(map(int, glibc_version.split(".")[:2])) < (2, 28):
    raise RuntimeError(f"glibc >= 2.28 is required, got {glibc_name} {glibc_version}")

bad_ld_paths = [
    p
    for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if p and ("/stubs" in p or ("cuda-12" in p and "/compat" in p))
]
if bad_ld_paths:
    raise RuntimeError(f"Forbidden CUDA library paths: {bad_ld_paths}")

try:
    importlib.metadata.distribution("cupy-cuda12x")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise RuntimeError("cupy-cuda12x is installed; remove it from this CUDA 11.8 environment")

import torch
import torchvision
import transformers
import xformers
import vllm

assert torch.__version__ == "2.7.1+cu118", torch.__version__
assert torchvision.__version__ == "0.22.1+cu118", torchvision.__version__
assert xformers.__version__ == "0.0.31", xformers.__version__
assert transformers.__version__ == "4.57.3", transformers.__version__
assert importlib.metadata.version("huggingface-hub") == "0.34.3"
assert importlib.metadata.version("qwen-vl-utils") == "0.0.14"
assert importlib.metadata.version("av") == "13.1.0"
assert importlib.metadata.version("vllm") == "0.11.0+torch271.cu118"
# setup.py appends the CUDA suffix to wheel metadata after the package version
# module is generated, so the runtime constant intentionally omits `.cu118`.
assert vllm.__version__ == "0.11.0+torch271", vllm.__version__
assert torch.version.cuda == "11.8", torch.version.cuda

native_modules = [
    "vllm._C",
    "vllm._moe_C",
    "vllm.cumem_allocator",
    "vllm.vllm_flash_attn._vllm_fa2_C",
]
loaded = [importlib.import_module(name) for name in native_modules]

# ldd does not inherit the search paths PyTorch adds while importing its
# extension modules.  Add the libraries shipped by the installed Torch/CUDA
# wheels explicitly so this check reflects the target environment.
runtime_library_dirs = [pathlib.Path(torch.__file__).parent / "lib"]
for site_dir in site.getsitepackages():
    runtime_library_dirs.extend(pathlib.Path(site_dir).glob("nvidia/*/lib"))
ldd_env = os.environ.copy()
ldd_env["LD_LIBRARY_PATH"] = ":".join(
    [str(path) for path in runtime_library_dirs if path.is_dir()]
    + ([ldd_env["LD_LIBRARY_PATH"]] if ldd_env.get("LD_LIBRARY_PATH") else [])
)

for module in loaded:
    result = subprocess.run(
        ["ldd", module.__file__],
        check=True,
        text=True,
        capture_output=True,
        env=ldd_env,
    )
    print(f"\nldd {module.__name__}:\n{result.stdout}")
    lowered = result.stdout.lower()
    if "not found" in lowered:
        raise RuntimeError(f"Unresolved library in {module.__name__}")
    if "/stubs" in lowered or ("cuda-12" in lowered and "/compat" in lowered):
        raise RuntimeError(f"Wrong CUDA driver library resolved for {module.__name__}")

cpp_info = pathlib.Path(xformers.__file__).with_name("cpp_lib.json")
build_info = json.loads(cpp_info.read_text())
assert build_info["version"]["cuda"] == 1108, build_info
assert build_info["version"]["torch"] == "2.7.1+cu118", build_info
assert build_info["env"]["TORCH_CUDA_ARCH_LIST"] == "7.5", build_info

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; check the NVIDIA driver or cuda-compat-11-8 setup")

assert torch.cuda.get_device_capability(0) == (7, 5), torch.cuda.get_device_capability(0)
assert "T4" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
print("CUDA tensor sum:", x.sum().item())

boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
print("torchvision NMS:", torchvision.ops.nms(boxes, scores, 0.5))

from xformers.ops import memory_efficient_attention

q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)
y = memory_efficient_attention(q, q, q)
print("XFormers attention:", y.shape, y.dtype, y.device)

print(
    "PASS:",
    {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": torchvision.__version__,
        "xformers": xformers.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "attention_backend": os.environ["VLLM_ATTENTION_BACKEND"],
    },
)
