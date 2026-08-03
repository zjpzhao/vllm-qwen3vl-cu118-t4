[GitHub 仓库：https://github.com/zjpzhao/vllm-qwen3vl-cu118-t4](https://github.com/zjpzhao/vllm-qwen3vl-cu118-t4)

# 面向 NVIDIA T4 / CUDA 11.8 / glibc 2.28 的 vLLM 0.11.0 Qwen3-VL Embedding 回移与内核裁剪

## 修改与裁剪摘要

本项目以 vLLM 0.11.0 为基线，为 NVIDIA T4（SM75）、CUDA 11.8、glibc
2.28 和 PyTorch 2.7.1+cu118 建立了独立构建与运行边界，主要修改如下：

- **固定构建工具链：** 使用 nvcc 11.8.89、GCC/G++ 11.4、glibc 2.28
  sysroot 和 `TORCH_CUDA_ARCH_LIST=7.5`，避免误用编译机 CUDA 12.9，并只生成
  T4 的 `sm_75` 设备代码。
- **恢复 CUDA 11 编译能力：** 重新启用 half、half2 和 BF16 运算符/转换，增加
  `VLLM_CUDA11_COMPAT` 条件，只隔离 CUDA 11.8 无法实例化的实现。
- **显式裁剪 DeepGEMM BF16/FP8 路径：** 不编译 DeepGEMM BF16/FP8 辅助函数
  和 `silu_mul_fp8_quant_deep_gemm_kernel`；保留算子注册和明确报错入口，维持
  Python/C++ ABI 完整。
- **裁剪 CUDA 12-only MoE 实现：** 不移植 `moe_permute`、
  `moe_unpermute`、`shuffle_rows` 的 CUDA 12 kernel，在 CUDA 11 分支保留
  与 operator schema 一致的报错 stub，使 `_moe_C` 可以正常加载。
- **修复并保留通用 SM75 内核：** 修复 KV Cache BF16 赋值歧义和 MoE stub
  签名，保留普通 FP16 SiLU/GELU、`act_and_mul_quant_kernel`、PagedAttention、
  KV Cache、MoE grouped-topk 和 CUTLASS C2x 路径。
- **按 SM75 自动跳过高架构内核：** 不构建 FlashMLA、FA2/FA3 CUDA kernel、
  Marlin、Machete、NVFP4、W4A8、CUTLASS C3x SM90/SM100/SM120 以及
  Hopper/Blackwell 专用实现。
- **回移 Qwen3-VL embedding/pooling：** 将后续版本的最小通用 adapter 行为
  回移到 0.11.0，使 `Qwen3VLForConditionalGeneration` 可转换为
  `Qwen3VLForEmbedding`，并清理生成 head、过滤 `lm_head.*` 权重。
- **启用视觉 embedding 与精度对照：** 服务默认允许每请求 1 张图片（视频关闭），
  保留视觉编码器；新增同输入 Transformers/vLLM 两阶段对照，检查逐向量误差、
  相似度矩阵和检索 Top-1 一致性。
- **绕开 T4 Triton Attention 阻断：** 增加可恢复的
  `apply_t4_xformers_hotfix.py`，对纯 prefill embedding 使用预编译的 xFormers
  CUTLASS contiguous attention，避开 SM75 上失败的 Triton Unified/Flex
  Attention；该路径要求关闭 prefix caching 和 chunked prefill。
- **清理 CUDA 12 传递依赖与动态链接：** 移除 `ray[cgraph]` 引入的
  `cupy-cuda12x`，修复相对 RUNPATH，确保 wheel 不捆绑 CUDA 12 compat、
  toolkit stub 或构建机绝对路径。

最终交付只保证 FP16、KV Cache dtype `auto`、XFormers attention 和
pooling/embedding 纯 prefill，不支持上述裁剪内核、FP8/DeepGEMM、文本生成
decode、prefix caching 或 chunked prefill。文本 Embedding API 已在真实 T4、
R450.191.01、glibc 2.28 与 `cuda-compat-11-8` 环境验证通过；视觉输入和
Transformers 对照需按第 6.6 节在目标机执行并保存报告。

## 结论

vLLM 0.11.0 的部分 DeepGEMM BF16/FP8 模板依赖 CUDA 12 才满足的接口、编译宏组合和更高 GPU 架构能力，无法在 CUDA 11.8 + SM75 下原样编译。本项目将源码调整为一个**仅面向 NVIDIA T4（SM75）、glibc 2.28、以 FP16 为受支持运行边界的 CUDA 11.8 wheel**：恢复 CUDA 11 的 half/BF16 编译能力，裁剪不兼容的 DeepGEMM 实现，为 CUDA 12-only MoE 算子保留 ABI stub，修复 CUDA driver 链接，并把 vLLM 后续版本的通用多模态 embedding/pooling 适配方式回移到 0.11.0，使 `Qwen3VLForConditionalGeneration` 能转换为 `Qwen3VLForEmbedding`。

当前已使用稳定版 PyTorch 2.7.1+cu118 完成 vLLM 与 XFormers wheel 构建、安装、原生扩展导入、相对 RUNPATH、`auditwheel show` 和 Qwen3-VL pooling 路由回归测试，并已在真实 T4、R450.191.01、glibc 2.28 与 `cuda-compat-11-8` 环境完成离线推理和 OpenAI Embedding API 端到端验证。

## 1. 目标与边界

| 项目 | 固定值 |
|---|---|
| vLLM | `v0.11.0`，commit `b8b302cde434df8c9289a2b465406b47ebab1c2d` |
| 目标 GPU | NVIDIA T4，计算能力 7.5（`sm_75`） |
| CUDA | nvcc 11.8.89 |
| Host 编译器 | Conda GCC/G++ 11.4，`sysroot_linux-64=2.28` |
| Python | 3.10.20 |
| 目标系统 ABI | x86_64，glibc `>=2.28` |
| PyTorch | `2.7.1+cu118`（CUDA 11.8 的最新官方稳定 wheel 系列） |
| torchvision | `0.22.1+cu118` |
| XFormers | `0.0.31`，源码构建，CUDA 11.8 / PyTorch 2.7.1 / SM75 |
| Transformers | `4.57.3`；Qwen-VL utils `0.0.14` |
| 运行边界 | FP16 权重，KV Cache dtype `auto` |

这里的“FP16 wheel”表示受支持的运行配置，不表示二进制中不存在任何 FP8 模板或符号。

## 2. 实际改动

### 2.1 固定 CUDA 11.8 与 SM75 工具链

构建工具链固定为：

```bash
export BUILD_ENV=/path/to/user-storage/miniconda3/envs/vllm-cu118-torch271-sysroot
export CUDA_HOME=/tmp/cuda-11.8-sysroot-view
export PATH="$BUILD_ENV/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export CC="$BUILD_ENV/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$BUILD_ENV/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST="7.5"
export MAX_JOBS=8
export NVCC_THREADS=1
export CMAKE_BUILD_TYPE=Release
export SETUPTOOLS_SCM_PRETEND_VERSION=0.11.0+torch271
export VLLM_TARGET_DEVICE=cuda
```

关键点：

- `CUDA_HOME` 必须指向隔离的 nvcc 11.8 view，避免 `setup.py` 误用编译机的 CUDA 12.9；`CUDA::cuda_driver` 链接到该 view 的 toolkit stub。
- Host compiler 和 sysroot 固定为 GCC/G++ 11.4 + glibc 2.28，避免把编译机更高版本的 GLIBC/GLIBCXX 写入 wheel。
- `TORCH_CUDA_ARCH_LIST=7.5` 保证 wheel 只生成 T4 的 `sm_75` 设备代码。
- `MAX_JOBS=8`、`NVCC_THREADS=1` 是本机实测安全值。当前编译会话受 32 GiB cgroup 限制，16 worker 会令多个 NVCC 被 SIGKILL（exit 137）并可能连带杀掉连接进程；使用 Release 模式和 8 worker 后完整构建通过。
- `SETUPTOOLS_SCM_PRETEND_VERSION=0.11.0+torch271` 明确区分定制 ABI，最终 metadata 为 `0.11.0+torch271.cu118`。

### 2.2 固定本地依赖

为避免 CMake 在构建过程中联网，构建过程使用 vLLM 0.11.0 对应的固定源码：

| 依赖 | 版本/commit | 环境变量 |
|---|---|---|
| CUTLASS | v4.0.0 | `VLLM_CUTLASS_SRC_DIR` |
| FlashMLA | `5f65b85703c7ed75fda01e06495077caad207c3f` | `FLASH_MLA_SRC_DIR` |
| vllm-flash-attention | `ee4d25bd84e0cbc7e0b9b9685085fd5db2dcb62a` | `VLLM_FLASH_ATTN_SRC_DIR` |

构建 Python 环境使用 CPython 3.10，并从 PyTorch 官方 cu118 索引安装稳定版组合：

```bash
conda create -n vllm-cu118-torch271 python=3.10 pip -y
conda activate vllm-cu118-torch271
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

PyTorch 2.7.1 是仍提供官方 cu118 wheel 的最新稳定 PyTorch 系列；PyTorch 2.8 的稳定 Linux wheel 已不提供 cu118。vLLM 0.11.0 上游原本固定 PyTorch 2.8，因此 CMake 会打印 `2.8.0 expected` 警告；本交付物是已经实际重编并验证原生扩展的 PyTorch 2.7.1 定制分支，不是 vLLM 官方二进制组合。

XFormers 使用官方 v0.0.31 源码重新构建：其 wheel metadata 为 `xformers==0.0.31`，`cpp_lib.json` 记录 CUDA `1108`、PyTorch `2.7.1+cu118` 与 `TORCH_CUDA_ARCH_LIST=7.5`，`_C.so` 的 cubin 静态检查只包含 `sm_75`。T4 应使用 XFormers CUTLASS attention 路径。

### 2.3 增加 CUDA 11 兼容编译条件

在 `CMakeLists.txt` 中使用 FindCUDA 实际提供的 `CUDA_VERSION` 判断 CUDA 11，并追加：

```cmake
if(VLLM_GPU_LANG STREQUAL "CUDA" AND CUDA_VERSION VERSION_LESS 12.0)
  list(APPEND VLLM_GPU_FLAGS
    "-U__CUDA_NO_HALF_OPERATORS__"
    "-U__CUDA_NO_HALF_CONVERSIONS__"
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"
    "-U__CUDA_NO_HALF2_OPERATORS__"
    "-DVLLM_CUDA11_COMPAT=1")
endif()
```

作用：

- 恢复 PyTorch 默认关闭的 CUDA half、half2 和 BF16 运算符/转换。
- 让普通 FP16/BF16 模板能够在 nvcc 11.8 下实例化。
- 用 `VLLM_CUDA11_COMPAT` 精确隔离无法兼容的实现，而不是删除整个 FP8 模块。
- 必须检查 `CUDA_VERSION`；当前 FindCUDA 构建路径不会可靠设置 `CMAKE_CUDA_COMPILER_VERSION`。

### 2.4 修复 CUDA driver 链接

构建节点的系统 driver 文件不可用于真实 GPU 验证。在 CUDA 11 兼容分支中，`CUDA::cuda_driver` **链接期**使用 CUDA 11.8 toolkit 自带的 `lib/stubs/libcuda.so`，确保扩展只保留通用 SONAME：

```text
DT_NEEDED: libcuda.so.1
```

toolkit stub 仅用于链接，不能用于导入或运行；部署到 T4 后必须由系统加载真实 NVIDIA driver，必要时加载 `cuda-compat-11-8`。

### 2.5 源码与算子 ABI 调整

| 文件/内核 | 处理方式 | 最终状态 |
|---|---|---|
| `csrc/cache_kernels.cu`：`concat_and_cache_ds_mla_kernel` | 将 `uint8_t` 量化值显式经 `float` 转为 `cache_t`，消除 CUDA 11 BF16 赋值歧义 | 修复后保留 |
| `indexer_k_quant_and_cache_kernel` | 通过恢复 BF16 conversions 编译，不裁剪源码 | 保留 |
| `csrc/quantization/activation_kernels.cu`：DeepGEMM 辅助函数与 `silu_mul_fp8_quant_deep_gemm_kernel` | 用 `VLLM_CUDA11_COMPAT` 排除 CUDA 11 不兼容实现 | 显式裁剪 |
| `silu_mul_fp8_quant_deep_gemm_cuda` | 保留函数和算子注册；CUDA 11 下调用时明确报错 | ABI 保留 |
| `act_and_mul_quant_kernel` | 保持在兼容条件块之外 | 保留 |
| `csrc/moe/grouped_topk_kernels.cu` | 通过恢复 half/half2 运算符和转换完成编译，不改源码 | 保留 |
| `csrc/moe/moe_permute_unpermute_op.cu`：`moe_permute` | 同步 CUDA 11 stub 与 operator schema 的参数签名 | ABI 修复 |
| `moe_unpermute`、`shuffle_rows` | 不移植 CUDA 12 kernel；为 CUDA 11 补齐明确报错的 stub | ABI 保留，不支持执行 |

保留 ABI stub 的目的是让 `_moe_C` 和 Python 算子注册正常加载；它不代表 CUDA 11.8 获得了 CUDA 12-only MoE kernel。

### 2.6 回移 Qwen3-VL embedding/pooling 适配

vLLM 0.11.0 已有 pooling 基础设施，但 Qwen3-VL 的生成模型同时在顶层和嵌套 `language_model` 中保留 `lm_head`/`logits_processor`，直接转换会残留生成路径。此次在 `vllm/model_executor/models/adapters.py` 中补齐嵌套清理逻辑，并沿用上游后续版本的通用模型转换方式：

- `Qwen3VLForConditionalGeneration` 在 `convert_type="embed"`、pooling runner 下转换为 `Qwen3VLForEmbedding`。
- 保留视觉编码器和多模态接口，只移除生成 head 与 logits processor。
- 加载 checkpoint 时跳过 `lm_head.*`，其余语言模型和视觉权重正常加载。
- 三个 CPU 回归测试覆盖顶层/嵌套模块清理、生成 head 权重过滤和真实 Qwen3-VL registry 路由；均已通过。

这不是把整个 vLLM 0.14 合并进 0.11.0，而是只回移 Qwen3-VL embedding 所需的最小通用 adapter 行为，降低与 CUDA 11.8 fork 的冲突面。

### 2.7 删除 CUDA 12 的传递依赖

当前 `ray[cgraph]` 会拉取 `cupy-cuda12x`。单张 T4 不使用 Ray CGraph/流水并行，因此本 T4 fork 将 CUDA requirements 改为基础 `ray>=2.48.0`；最终 wheel metadata 和离线 wheelhouse 均不含 `cupy-cuda12x`、CUDA 12 runtime 或 CUDA 12 compat 库。若以后需要流水并行，应另行设计多卡 CUDA 11 依赖，而不能把 `cupy-cuda12x` 混入本环境。

## 3. 最终内核边界

### 3.1 显式裁剪或禁用

- DeepGEMM BF16/FP8 辅助函数。
- `silu_mul_fp8_quant_deep_gemm_kernel`。
- CUDA 12-only 的 `moe_permute`、`moe_unpermute`、`shuffle_rows` 实现；仅保留 ABI stub。

### 3.2 修复后保留

- 普通 FP16 SiLU/GELU。
- `act_and_mul_quant_kernel`。
- 普通 KV Cache 与 PagedAttention v1/v2。
- `concat_and_cache_ds_mla_kernel`、`indexer_k_quant_and_cache_kernel` 的可编译模板。
- MoE grouped-topk。
- `_C`、`_moe_C`、`cumem_allocator`。
- SM75 CUTLASS C2x 路径。

### 3.3 由 CMake 根据 SM75 自动跳过

- FlashMLA。
- FlashAttention 2/3 CUDA kernels。
- Marlin、Marlin MoE、AllSpark。
- CUTLASS C3x SM90/SM100/SM120、Sparse C3x、CUTLASS MLA。
- NVFP4、W4A8、Machete、HadaCore。
- Hopper/Blackwell grouped MoE 与 SM100 blockwise kernels。

`_vllm_fa2_C` 扩展可以加载，但本构建没有可用于 SM75 的 FA2 CUDA kernel；T4 运行时应使用 XFormers 或已验证的 fallback attention backend。

## 4. 最终构建命令

```bash
cd /path/to/user-storage/tools/vllm
bash scripts/build_t4_cu118_glibc228.sh
```

脚本固定 `sysroot_linux-64=2.28` 构建环境、隔离 CUDA 11.8 view、`TORCH_CUDA_ARCH_LIST=7.5`、`MAX_JOBS=8`、`NVCC_THREADS=1` 和 `CMAKE_BUILD_TYPE=Release`，并把完整输出 `tee` 到 `/path/to/user-storage/tools/vllm/build.log`。如果构建被外部中止，保留 `build/` 后直接重跑脚本，Ninja 会增量续编；不要再次删除缓存，也不要在 32 GiB cgroup 中提高到 16 worker。

链接完成后必须修复 Conda compiler 注入的绝对 RPATH，再用 `wheel pack` 重新生成 RECORD。最终四个扩展只允许以下相对 RUNPATH：

```text
vllm/*.so: $ORIGIN/../torch/lib:$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cuda_nvrtc/lib
vllm/vllm_flash_attn/*.so: $ORIGIN/../../torch/lib:$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cuda_nvrtc/lib
```

## 5. 产物与验证结果

| 检查项 | 结果 |
|---|---|
| Wheel | `dist-torch271/vllm-0.11.0+torch271.cu118-cp310-cp310-linux_x86_64.whl` |
| 大小 | 约 21 MiB（Release，无调试符号） |
| SHA256 | `9a46ed2d8c27a4025297b421bf6639d586642d41f84cbd767dde4ff85ca699dc` |
| XFormers wheel | `xformers-0.0.31-cp39-abi3-linux_x86_64.whl`；SHA256 `5cdbd3a11f836c079e9c50a56ea20a4b4d3d188b29834318a3bd3fdbb6bb91ad` |
| Wheel 完整性 | `python -m zipfile -t` 通过 |
| Metadata | `0.11.0+torch271.cu118`；`ray>=2.48.0`，无 `ray[cgraph]`/CuPy CUDA 12 |
| 原生扩展 | `vllm._C`、`vllm._moe_C`、`vllm.cumem_allocator`、`vllm.vllm_flash_attn._vllm_fa2_C` 均成功导入 |
| CUDA 动态依赖 | 包含 `libcudart.so.11.0` 和 `libcuda.so.1` |
| CUDA 架构 | vLLM `_C`、`_moe_C` 与 XFormers `_C.so` 均只包含 `sm_75`；内置 FA2 不提供 T4 可用路径，运行时必须选择 XFormers |
| PyTorch/torchvision 架构 | `libtorch_cuda.so` 和 `torchvision/_C.so` 均确认包含 `sm_75` |
| RPATH/RUNPATH | 已移除构建 Conda 环境绝对路径；仅保留指向同一环境 `torch/lib`、`nvidia/cuda_runtime/lib`、`nvidia/cuda_nvrtc/lib` 的 `$ORIGIN` 相对 RUNPATH |
| `auditwheel show` | 未发现捆绑的 CUDA 12、CuPy CUDA 12 或绝对路径；预期外部依赖为 `libtorch*`、`libc10*`、`libcudart.so.11.0`、`libnvrtc.so.11.2` 和 `libcuda.so.1` |
| 平台兼容性 | wheel 实际标签为 `linux_x86_64`，`auditwheel` 判定系统符号下限为 `manylinux_2_24_x86_64`；目标机 glibc 2.28 满足要求 |
| Qwen3-VL pooling | 三个 adapter/registry CPU 回归测试通过；`Qwen3VLForConditionalGeneration` 正确路由为 `Qwen3VLForEmbedding` |
| `pip check` | 在 `torch==2.7.1+cu118`、`torchvision==0.22.1+cu118`、`xformers==0.0.31` 的完整运行环境中为 `No broken requirements found` |
| 真实 T4 推理 | 已通过：T4 / R450.191.01 / `cuda-compat-11-8`，FP16、XFormers CUTLASS contiguous prefill；2 条文本离线 embedding 与 OpenAI API 均成功 |
| Embedding API | `/health` 为 HTTP 200，`/v1/models` 正确注册模型；`/v1/embeddings` 返回 2048 维向量，L2 norm `1.0000000199780135`，输入 35 tokens |

构建节点上的**链接**使用 CUDA 11.8 toolkit driver stub；扩展导入时动态链接器实际可能解析到该节点的 CUDA 12.9 compat，因此构建机测试本身只证明 wheel、ELF 与算子 ABI 完整。上述 R450/T4 结论来自目标机独立验收；当前已经验证文本 embedding/pooling 的完整 prefill 与 API 链路，图片输入仍需按本节给出的多模态配置单独验收。本 fork 不用于文本生成/decode。

`readelf -d` 已确认扩展只有通用的 `NEEDED: libcuda.so.1`，wheel 内没有打包任何 `libcuda`；相对 RUNPATH 也不包含 driver 路径。构建链接使用 CUDA 11.8 toolkit stub 生成通用 SONAME，目标 T4 上必须解析到真实 NVIDIA driver 或 `/usr/local/cuda-11.8/compat/libcuda.so.1`，严禁使用编译机的 CUDA 12.9 compat 或 toolkit `lib/stubs`。

## 6. 目标 T4 的部署与验收

### 6.1 先检查目标机，不要直接安装

```bash
uname -m
ldd --version | head -n 1
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total \
  --format=csv,noheader
python3.10 -V
```

本 wheel 的硬性条件如下：

| 项目 | 要求 |
|---|---|
| GPU | NVIDIA T4，计算能力 `7.5` |
| CPU 架构 | `x86_64` |
| Python | CPython 3.10；wheel 标签为 `cp310-cp310` |
| glibc | `>=2.28`；本 wheel 的 auditwheel 系统符号下限为 manylinux_2_24，目标 Debian 10 glibc 2.28 已覆盖 |
| CUDA 用户态 | `torch==2.7.1+cu118`、`torchvision==0.22.1+cu118`、`xformers==0.0.31` |
| NVIDIA driver | 当前为 `450.191.01`；它满足 CUDA 11.x minor compatibility 的最低版本，但已是 EOL 分支，处理方式见下一节 |

目标机使用预编译 wheel 时不需要安装 `nvcc` 或完整 CUDA Toolkit。`nvidia-smi` 顶部显示的“CUDA Version”只是驱动可支持的最高 CUDA 版本，不代表当前 PyTorch wheel 的运行时版本；安装后必须确认 `torch.version.cuda == "11.8"`。

### 6.2 驱动是 450.191.01 时怎么办

`450.191.01` 高于 NVIDIA [CUDA 11.x minor-version compatibility 规则](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)要求的 Linux driver `450.80.02`，所以它**不是硬性版本不满足**；纯 `sm_75` cubin 有机会直接运行。但 NVIDIA 同时明确限制：依赖较新 driver 能力的调用会返回 `cudaErrorCallRequiresNewerDriver`，包含新版本 PTX 的程序可能无法由旧 driver 完成 JIT。

本项目自己的 vLLM 扩展已包含 `sm_75` cubin，但 PyTorch、Triton 和 XFormers 仍可能在运行时使用 PTX/JIT，因此 `450.191.01` 不能只做静态版本判断，处理顺序如下：

1. **生产首选：升级 NVIDIA driver。** CUDA 11.8 GA 的配套 Linux driver 为 [`520.61.05`](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/)，实际应选择不低于该版本且仍在维护的数据中心驱动分支，并继续使用 cu118 wheel。升级 driver 不等于安装 CUDA 12，也不会改变 `torch.version.cuda == "11.8"`。
2. **不能升级 driver：** T4 属于数据中心 GPU；NVIDIA 的 [CUDA 11.8 历史支持说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-09.html)允许 `450.51` 或更高的 R450 使用 driver compatibility package，因此 `450.191.01` 满足这个历史门槛。可由管理员按 NVIDIA [CUDA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html)说明安装 `cuda-compat-11-8`，并把兼容库放到运行时搜索路径首位。但当前 NVIDIA 表格已不再列出 R450，并明确将未列出的 EOL 分支视为不受支持目标，所以这只能作为遗留环境的应急验证方案，不能视为当前生产支持承诺。
3. **既不能升级也不能安装兼容包：** 可以按 CUDA 11 minor compatibility 做实验性实测，但不能作为生产兼容性承诺；一旦出现 PTX/JIT 或 driver API 错误，应停止绕过并升级 driver。

对当前 wheel 和重新编译的判断如下：

| 问题 | 结论 |
|---|---|
| 当前 wheel 能否使用 | **有可能，先用当前 wheel 测试，不需要因 `450.191.01` 先重编**；是否可用以 6.5、6.6 节全链路测试为准 |
| 加载 `cuda-compat-11-8` 后测试通过 | 当前 wheel 可以继续使用，无需重编 vLLM |
| 报 PyTorch/vLLM ABI 或缺少 `sm_75` | 对应组件版本或架构不匹配；按本文 wheelhouse 固定版本，必要时重编该组件 |
| 报 PTX/JIT/driver API 错误 | **只按本文方法重新编译 vLLM 没有用**；源码裁剪解决的是 CUDA 11.8 编译问题，不会改变 R450 的 driver/JIT 能力 |
| 本文方法还能否编译 | 可以；CUDA 11.8 + `TORCH_CUDA_ARCH_LIST=7.5` 仍能生成相同的 SM75 wheel，但“能编译”不等于“能在 R450 上完整运行” |

如果目标机既不能升级 driver，也不能使用 compatibility package，要真正规避 R450 的 PTX/JIT 限制，就需要把 PyTorch、Triton、torchvision、XFormers 和 vLLM 整套依赖改成适配旧 driver 的工具链并消除运行时 PTX/JIT；这已经超出本文的 vLLM 内核裁剪范围，也不建议作为 vLLM 0.11.0 的交付方案。

管理员先为目标机配置 NVIDIA CUDA 11.8 软件源，再按系统选择其中一条安装命令：

```bash
# Ubuntu / Debian
sudo apt install cuda-compat-11-8

# RHEL / Rocky / CentOS 系
sudo dnf install cuda-compat-11-8
```

若当前软件源已经下架 CUDA 11.8 包，应使用组织留存或 NVIDIA 官方归档中的原始包，不要从不可信镜像复制 `libcuda.so`。安装后的典型环境变量为：

```bash
TORCH_LIB=$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/compat:$CONDA_PREFIX/lib:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
```

`/usr/local/cuda-11.8/compat` 必须来自真实的 `cuda-compat-11-8` 包。**禁止**把 CUDA Toolkit 的 `lib/stubs` 加入目标机 `LD_LIBRARY_PATH`；stub 只用于链接，不能驱动真实 GPU。

若遇到以下任一错误，说明 R450 路径没有通过验收，应升级 driver，而不是继续增加环境变量绕过：

- `cudaErrorCallRequiresNewerDriver`。
- `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` 或其他 PTX JIT 错误。
- Triton/XFormers 编译 kernel 失败。
- `libcuda.so.1` 来自 toolkit stub，或 CUDA 初始化失败。

### 6.3 准备离线 wheelhouse

vLLM wheel 外部依赖目标环境中的 `libtorch*`、`libc10*` 和 CUDA 11 runtime，因此不能只复制一个 vLLM wheel。本次已经准备完整离线目录 `release-t4-cu118-torch271/`，包含：

- 143 个 wheel、约 3.1 GiB，包括官方 `torch==2.7.1+cu118`、`torchvision==0.22.1+cu118` 及全部 CUDA 11 用户态依赖。
- 定制 `xformers==0.0.31` SM75 wheel 与 `vllm==0.11.0+torch271.cu118` wheel。
- 普通 vLLM Python 运行依赖、约束文件、安装脚本和真实 T4 验证脚本。
- 源码裁剪文档、编译日志和 SHA256 清单。

wheelhouse 文件名扫描确认没有 `cupy-cuda12x`、CUDA 12 runtime、`libcuda` 或 compat 库。不要额外安装最新版/通用 PyPI XFormers，也不要安装 torchaudio；这两者都可能使 pip 替换已固定的 PyTorch/CUDA ABI。

最终材料整理到：

```text
/path/to/user-storage/tools/vllm-qwen3vl-cu118-t4/
```

### 6.4 在目标机安装

```bash
cd /path/to/vllm-qwen3vl-cu118-t4

conda env create -f environment.yml
conda activate vllm-t4-cu118-torch271
./install_target.sh
```

安装脚本会先检查 `x86_64`、CPython 3.10、glibc 2.28、SHA256 和 `LD_LIBRARY_PATH`，再完全离线安装固定 wheel，并拒绝 `cupy-cuda12x`。正常 driver 路径不需要人为加入系统 CUDA Toolkit：

```bash
unset CUDA_HOME
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_T4_XFORMERS_CONTIGUOUS_PREFILL=1
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
export TRITON_CACHE_DIR=/tmp/triton-cache-cu118-sm75-xformers
python verify_target.py
```

`install_target.sh` 会自动运行 `apply_t4_xformers_hotfix.py`。若目标机已经安装
过旧版材料，无需重装或重编 wheel，只需在更新发布仓库后执行：

```bash
python apply_t4_xformers_hotfix.py
python verify_qwen3vl_embedding.py \
  --model /root/Qwen3-VL-Embedding-2B \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80
```

该补丁解决两条 SM75 运行时 JIT 阻断：XFormers V1 prefill 会进入
`triton_unified_attention.py`，FlexAttention 则会由 TorchInductor 生成 Triton
kernel；两者在 Torch 2.7.1 配套 Triton 的 `ConvertTritonGPUToLLVM` 阶段均会报
`Unsupported conversion from f16 to f16`。补丁仅针对 Qwen3-VL
pooling/embedding 的纯 prefill，把当前连续 Q/K/V 交给预编译的 xFormers
CUTLASS kernel，并要求关闭 prefix caching 和 chunked prefill。回滚命令为：

```bash
python apply_t4_xformers_hotfix.py --restore
```

若采用 R450 + `cuda-compat-11-8`，则在执行验证与服务前把 `/usr/local/cuda-11.8/compat` 放在最前面。不得出现 `/usr/local/cuda-12.9/compat`、其他 CUDA 12 compat 路径或任何 toolkit `lib/stubs`。此时 `nvidia-smi` 仍可能显示 R450 对应的 CUDA 能力，这不代表 compat 库未生效；应以真实 CUDA、XFormers 和 vLLM 测试为准。

### 6.5 安装后必须通过的验收

首先检查 PyTorch、T4、vLLM 扩展和 torchvision 算子：

```bash
python - <<'PY'
import torch
import torchvision
import vllm._C
import vllm._moe_C
import vllm.cumem_allocator
import vllm.vllm_flash_attn._vllm_fa2_C

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

assert torch.version.cuda == "11.8"
assert torch.__version__ == "2.7.1+cu118"
assert torchvision.__version__ == "0.22.1+cu118"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (7, 5)

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
print("CUDA tensor sum:", x.sum().item())

boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
print("torchvision NMS:", torchvision.ops.nms(boxes, scores, 0.5))
PY
```

再检查 XFormers 的构建信息和真实注意力计算：

```bash
python -m xformers.info

python - <<'PY'
import torch
from xformers.ops import memory_efficient_attention

q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)
y = memory_efficient_attention(q, q, q)
print(y.shape, y.dtype, y.device)
PY
```

`xformers.info` 应显示 CUDA 11.8/`1108`、匹配的 PyTorch 版本，并至少有一个适用于 T4 的 memory-efficient attention operator 可用。最后检查动态库和 Python 依赖：

```bash
ldd "$(python -c 'import vllm._C as m; print(m.__file__)')" \
  | grep -E 'not found|cuda|torch|c10'
python -m pip check
```

`ldd` 不得出现 `not found`，不得解析到任何 toolkit `lib/stubs` 或 CUDA 12 compat 路径；`pip check` 必须输出 `No broken requirements found`。离线包中的 `verify_target.py` 已自动覆盖上述版本、动态库、T4、NMS 与 XFormers 注意力检查。

### 6.6 最小化启动并逐步放量

该 wheel 只承诺 `--dtype half --kv-cache-dtype auto`，不承诺 FP8、DeepGEMM、DeepSeek FP8 MLA、FlashMLA、FA3、Hopper/Blackwell kernel 及 SM75 构建时跳过的量化后端。

先用较小模型或能装入 T4 16 GiB 的 checkpoint 完成链路测试；8B 模型仅 FP16 权重就接近 16 GiB，尚未计入 KV Cache、激活和 CUDA 上下文，通常会 OOM。

```bash
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_T4_XFORMERS_CONTIGUOUS_PREFILL=1
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
export TRITON_CACHE_DIR=/tmp/triton-cache-cu118-sm75-xformers

vllm serve /path/to/Qwen3-VL-model \
  --host :: \
  --port 8000 \
  --served-model-name Qwen3-VL-Embedding-2B \
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
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --trust-remote-code
```

这里使用 `--host ::` 是为了让服务监听 IPv6。若启动后
`ss -lntp | grep ':8000'` 只显示 `0.0.0.0:8000`，则该 socket 仅接受 IPv4，
通过 `http://[IPv6]:8000` 访问会失败。可在激活目标 Conda 环境后使用发布目录的
重启脚本：

```bash
cd /root/vllm-qwen3vl-cu118-t4
chmod +x restart_vllm_server_ipv6.sh
./restart_vllm_server_ipv6.sh
```

脚本仅终止当前占用 8000 端口的进程，等待其正常退出后以本文完整参数后台启动；
PID 和日志分别保存为 `vllm_server.pid`、`vllm_server.log`。

日志必须出现 `Using XFormers backend on V1 engine` 和支持任务
`['encode', 'embed']`。T4 上关于 FA2 需要 compute capability >= 8 的信息是
后端探测结果，只要随后明确使用 XFormers 就不影响运行。

启动后先检查服务和模型列表：

```bash
curl -f http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
curl --noproxy '*' -g http://[::1]:8000/health
```

从开发机请求目标机 IPv6 时，应清除环境代理再测试：

```bash
TARGET_IPV6='<TARGET_IPV6>'
env \
  -u http_proxy -u https_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY \
  curl -f -g "http://[${TARGET_IPV6}]:8000/health"
```

若未绕过代理，HTML 格式的 521/502 是代理响应而非 vLLM 响应；这类失败请求的
低延迟和高吞吐量不能用于性能结论。Python `requests`/`httpx` 客户端应分别设置
`Session.trust_env = False` 或 `trust_env=False`。

再按发布目录 `README.md` 的 `/v1/embeddings` 请求完成文本验收。成功判据为：
模型名正确、向量维度 2048、全部数值有限、L2 norm 与 1.0 的差小于 0.02。
本次真实目标机结果中 norm 约为 `1.0` 且 `completion_tokens=0`；embedding
不生成文本，因此 completion tokens 为 0 正常。chat embedding 请求必须显式传入
`add_special_tokens: true`，以和官方 `Qwen3VLEmbedder` 一样在模板末尾追加
`<|endoftext|>` 并对它执行 LAST pooling。省略时 vLLM 默认值为 false，请求会少
1 个 token，并池化前一个换行 token，不能与官方 embedding 对齐。

当前命令默认允许每个请求携带 1 张图片并关闭视频；无图片请求仍只执行文本路径。
视觉请求使用 chat embedding 的 `image_url` 内容块，建议传 data URI，避免服务端
无法访问客户端本地路径。图片链路通过前不得把文本验收结论扩大为完整多模态验收。

精度验证采用两阶段执行，避免 Transformers 与 vLLM 两份 2B 模型同时占用 T4：

```bash
conda activate vllm-t4-cu118-torch271
cd /root/vllm-qwen3vl-cu118-t4
./run_accuracy_check.sh transformers
./run_accuracy_check.sh vllm
```

第一条命令停止现有 vLLM、建立时间戳目录并生成 Transformers 基准；第二条命令
复用 `accuracy_runs/latest`，启动允许单图片输入的服务，生成 vLLM 向量并比较。
如需自定义路径，设置 `MODEL_PATH`、`IMAGE_PATH` 或 `RUN_DIR`。等价展开命令如下：

```bash
cd /root/vllm-qwen3vl-cu118-t4
export MODEL=/root/Qwen3-VL-Embedding-2B
export IMAGE=/root/test.jpg
export RUN_DIR="$PWD/accuracy_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
set -o pipefail
echo "accuracy outputs: $RUN_DIR"

PID="$(ss -H -lntp 'sport = :8000' 2>/dev/null \
  | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)"
if [ -n "$PID" ]; then
  kill -TERM "$PID"
  for _ in $(seq 1 60); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
fi
python compare_vllm_transformers.py reference \
  --model "$MODEL" \
  --image "$IMAGE" \
  --output "$RUN_DIR/precision_transformers.json" \
  2>&1 | tee "$RUN_DIR/precision_transformers.log"

IMAGE_LIMIT=1 VIDEO_LIMIT=0 \
LOG_FILE="$RUN_DIR/vllm_server.log" \
PID_FILE="$RUN_DIR/vllm_server.pid" \
./restart_vllm_server_ipv6.sh
until curl -fsS --noproxy '*' -g http://[::1]:8000/health; do sleep 2; done
python compare_vllm_transformers.py vllm \
  --endpoint http://[::1]:8000/v1/embeddings \
  --image "$IMAGE" \
  --output "$RUN_DIR/precision_vllm.json" \
  2>&1 | tee "$RUN_DIR/precision_vllm.log"
python compare_vllm_transformers.py compare \
  --reference "$RUN_DIR/precision_transformers.json" \
  --candidate "$RUN_DIR/precision_vllm.json" \
  --report "$RUN_DIR/precision_report.json" \
  2>&1 | tee "$RUN_DIR/precision_compare.log"

find "$RUN_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
```

脚本强制对齐 FP16、instruction、chat template、LAST pooling、L2 normalize 和
图片字节，默认要求相同输入最小余弦不低于 0.995、pairwise similarity MAE 不高于
0.02、检索 Top-1 完全一致。每次运行的向量 JSON、比较报告、推理日志和 vLLM
服务日志统一保存在 `accuracy_runs/<时间戳>/`，该目录不会提交到 Git。阈值用于工程回归，不替代在业务标注集上计算 Recall@K、
MRR/nDCG；首次结果应保存为后续固定基线。文本链路通过后，可逐步提高
`--max-num-seqs`、上下文长度和显存利用率；prefix caching 与 chunked prefill
在此热补丁中仍不能启用。

### 6.7 常见阻断条件

| 现象 | 处理 |
|---|---|
| glibc `< 2.28` | 当前交付目标之外；需在对应更老的 manylinux/sysroot 环境重编并重新审计全部二进制依赖 |
| Python 不是 3.10 | 新建 Python 3.10 环境 |
| R450 出现 PTX/JIT/driver API 错误 | 升级到不低于 R520 且仍在维护的分支；若已装 compat 包仍失败，也不要继续绕过 |
| `libcuda.so.1` 缺失 | 修复真实 NVIDIA driver 或 compat 路径，绝不能使用 toolkit stub |
| `libcudart.so.11.0` 缺失 | 安装/恢复 cu118 用户态 runtime，并检查环境库路径 |
| XFormers operator 不可用 | 使用同一 PyTorch/CUDA ABI 和 `sm_75` 重新源码构建 |
| `Unsupported conversion from f16 to f16` | 确认已应用 `apply_t4_xformers_hotfix.py`、后端为 XFORMERS，并关闭 prefix caching/chunked prefill |
| `no kernel image is available` | 检查相关 torch/vision/XFormers wheel 是否包含 `sm_75` |
| T4 OOM | 换更小 checkpoint，降低上下文长度、并发和显存利用率 |
| IPv6 请求返回代理 HTML 521/502 | 服务使用 `--host ::`，开发机清除代理或设置客户端 `trust_env=False`；先以 `/health` 的 HTTP 200 验证直连 |

## 一句话汇报

面向 T4 / CUDA 11.8 / glibc 2.28，vLLM 0.11.0 的部分 DeepGEMM BF16/FP8 与 CUDA 12-only MoE 内核无法原样编译，我已裁剪不兼容实现并保留 ABI stub、回移 Qwen3-VL 文本/视觉 embedding/pooling 适配，并用 xFormers CUTLASS contiguous prefill 热补丁避开 SM75 不兼容的 Triton Attention；服务现默认开放单图片输入，并提供与 Transformers 的同输入精度对照脚本，具体裁剪内核、部署和验收步骤参考本文。
