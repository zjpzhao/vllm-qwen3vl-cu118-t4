# Qwen3-VL Embedding：vLLM 0.11.0 / T4 / CUDA 11.8 / glibc 2.28 发布包

这是面向 `x86_64 + CPython 3.10 + glibc >= 2.28 + NVIDIA T4 (SM75)` 的定制构建，包含 PyTorch 2.7.1+cu118、XFormers 0.0.31、裁剪后的 vLLM 0.11.0 wheel、Qwen3-VL embedding/pooling 回移、离线依赖、源码补丁、安装脚本和验证脚本。

该构建只保证 FP16、`--kv-cache-dtype auto` 和 XFormers attention；不支持 FP8/DeepGEMM、DeepSeek FP8 MLA、FA3、Ray CGraph/Pipeline Parallel，以及编译时跳过的 Hopper/Blackwell 等高架构内核。它不是通用 CUDA wheel。

## 目标机安装

目标机不需要 nvcc 或完整 CUDA Toolkit。进入完整发布包目录后执行：

```bash
conda env create -f environment.yml
conda activate vllm-t4-cu118-torch271
./install_target.sh

export VLLM_ATTENTION_BACKEND=XFORMERS
python verify_target.py
python verify_qwen3vl_embedding.py \
  --model /root/Qwen3-VL-Embedding-2B
```

如需同时验证视觉输入，追加 `--image /path/to/test.jpg`。验证脚本要求输出维度为 2048、L2 norm 约为 1，并分别覆盖纯文本和可选图片 embedding。

启动时保持相同环境，例如：

```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
vllm serve /path/to/Qwen3-VL-model \
  --dtype half \
  --kv-cache-dtype auto \
  --tensor-parallel-size 1
```

模型权重不包含在本发布包中。

## R450.191.01 与 libcuda 边界

R450.191.01 达到 CUDA 11.x minor-version compatibility 的最低驱动分支，但旧驱动的 PTX JIT 能力仍可能使部分依赖在目标机报错，因此必须在真实 T4 上运行 `verify_target.py` 和一次最小模型推理。

优先使用目标机系统驱动提供的 `libcuda.so.1`。若验证失败，首选升级驱动；无法升级时，由管理员为数据中心 GPU 安装 `cuda-compat-11-8`，再将它放到搜索路径最前面：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/compat:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
```

务必遵守以下两条：

- 禁止使用或复制编译机的 `/usr/local/cuda-12.9/compat/libcuda.so.1`；它不属于此 CUDA 11.8 目标环境。
- 禁止把任何 CUDA Toolkit 的 `lib/stubs` 加入运行时 `LD_LIBRARY_PATH`；stub 只用于链接，不能驱动 GPU。

可用下面的命令确认 Python 实际加载的驱动库路径：

```bash
python - <<'PY'
import ctypes

ctypes.CDLL("libcuda.so.1")
paths = sorted({line.rsplit(None, 1)[-1]
                for line in open("/proc/self/maps")
                if "libcuda.so" in line})
print("\n".join(paths))
assert not any("cuda-12.9/compat" in path or "/stubs/" in path
               for path in paths), paths
PY
```

## 复现源码裁剪

补丁以干净的 vLLM `v0.11.0` 源码为基线：

```bash
git checkout v0.11.0
git apply --check patches/vllm-0.11.0-t4-cu118-torch271.patch
git apply patches/vllm-0.11.0-t4-cu118-torch271.patch
```

补丁记录 CUDA 11.8/T4 与 Qwen3-VL embedding 所需的最小改动：CUDA 11 编译标志、FP8/DeepGEMM 路径裁剪、CUDA 12 MoE 算子 stub、KV cache 兼容转换、嵌套语言模型 generation head 清理，以及从运行依赖中移除官方 Torch/XFormers 固定版本并禁用 Ray CGraph extra。

## Git 仓库与 GitHub Release

Git 仓库只提交 README、文档、脚本、requirements、日志和 `patches/`。不要提交 `wheelhouse/`、完整 `.tar.gz` 或分卷文件，也不要使用 Git LFS 存放这些二进制；它们应作为 GitHub Release assets 发布。

仓库的 `.gitignore` 至少应包含：

```gitignore
wheelhouse/
*.tar.gz
*.tar.gz.part-*
SHA256SUMS
```

首次提交时应显式选择文件，避免误把 wheel 加入 Git：

```bash
git add README.md environment.yml constraints-t4-cu118.txt \
  install_target.sh verify_target.py verify_qwen3vl_embedding.py \
  requirements patches logs \
  'vLLM 0.11.0 在 T4 CUDA 11.8 上的源码编译与内核裁剪.md' \
  .gitignore
git status --short
git commit -m 'Release vLLM 0.11.0 for T4 CUDA 11.8'
git push origin HEAD
```

在 `git status` 中确认没有 `wheelhouse/` 或分卷文件后再 push。

### 生成小于 2 GiB 的 Release 分卷

假设完整材料目录为 `/path/to/user-storage/tools/vllm-qwen3vl-cu118-t4`：

```bash
cd /path/to/user-storage/tools
tar --exclude='.git' -cf vllm-qwen3vl-cu118-t4-offline.tar \
  vllm-qwen3vl-cu118-t4
sha256sum vllm-qwen3vl-cu118-t4-offline.tar \
  > vllm-qwen3vl-cu118-t4-offline.tar.sha256
split -b 1900M -d -a 3 \
  vllm-qwen3vl-cu118-t4-offline.tar \
  vllm-qwen3vl-cu118-t4-offline.tar.part-
sha256sum vllm-qwen3vl-cu118-t4-offline.tar.part-* > SHA256SUMS
```

`1900M` 低于 GitHub Release 单 asset 的 2 GiB 硬限制。只上传 `part-*`、`SHA256SUMS` 和整包校验文件：

```bash
gh release create v0.11.0-t4-cu118-torch271 \
  --title 'vLLM 0.11.0 for T4 / CUDA 11.8 / Torch 2.7.1' \
  --notes-file README.md
gh release upload v0.11.0-t4-cu118-torch271 \
  /path/to/user-storage/tools/vllm-qwen3vl-cu118-t4-offline.tar.part-* \
  /path/to/user-storage/tools/SHA256SUMS \
  /path/to/user-storage/tools/vllm-qwen3vl-cu118-t4-offline.tar.sha256
```

### 下载后重组

所有分卷下载到同一目录后，先逐卷校验，再按固定数字后缀顺序重组并校验整包：

```bash
sha256sum -c SHA256SUMS
cat vllm-qwen3vl-cu118-t4-offline.tar.part-* \
  > vllm-qwen3vl-cu118-t4-offline.tar
sha256sum -c vllm-qwen3vl-cu118-t4-offline.tar.sha256
tar -xf vllm-qwen3vl-cu118-t4-offline.tar
```

然后进入解压目录，按“目标机安装”一节执行。
