# Qwen3-VL Embedding：vLLM 0.11.0 / T4 / CUDA 11.8 / glibc 2.28 发布包

这是面向 `x86_64 + CPython 3.10 + glibc >= 2.28 + NVIDIA T4 (SM75)` 的定制构建，包含 PyTorch 2.7.1+cu118、XFormers 0.0.31、裁剪后的 vLLM 0.11.0 wheel、Qwen3-VL 文本/视觉 embedding/pooling 回移、离线依赖、源码补丁、安装脚本和验证脚本。

该构建只保证 FP16、`--kv-cache-dtype auto` 和 XFormers attention；不支持 FP8/DeepGEMM、DeepSeek FP8 MLA、FA3、Ray CGraph/Pipeline Parallel，以及编译时跳过的 Hopper/Blackwell 等高架构内核。它不是通用 CUDA wheel。

详细资料：

- [Transformers/vLLM 精度对齐测试、问题定位与最终结果](Qwen3-VL-Embedding-Transformers-vLLM-精度对齐测试.md)
- [源码编译、回移和内核裁剪说明](vLLM%200.11.0%20在%20T4%20CUDA%2011.8%20上的源码编译与内核裁剪.md)

## 目标机安装

目标机不需要 nvcc 或完整 CUDA Toolkit。进入完整发布包目录后执行：

```bash
conda env create -f environment.yml
conda activate vllm-t4-cu118-torch271
./install_target.sh

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_T4_XFORMERS_CONTIGUOUS_PREFILL=1
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
export TRITON_CACHE_DIR=/tmp/triton-cache-cu118-sm75-xformers
python verify_target.py
python verify_qwen3vl_embedding.py \
  --model /root/Qwen3-VL-Embedding-2B
```

如需同时验证视觉输入，追加 `--image /path/to/test.jpg`。验证脚本要求输出维度为 2048、L2 norm 约为 1，并分别覆盖纯文本和可选图片 embedding。服务默认允许每个请求携带 1 张图片，视频关闭；全模态启动命令见下一节。

## 启动 Embedding 后端

以下命令已经在真实 T4 / R450.191.01 / CUDA 11.8 compatibility package
环境中通过验证。热补丁只支持 pooling/embedding 的纯 prefill，因此 prefix
caching 和 chunked prefill 必须保持关闭。

```bash
conda activate vllm-t4-cu118-torch271
cd /root/vllm-qwen3vl-cu118-t4

unset CUDA_HOME
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_T4_XFORMERS_CONTIGUOUS_PREFILL=1
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
export TRITON_CACHE_DIR=/tmp/triton-cache-cu118-sm75-xformers
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/compat:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}

mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p logs
set -o pipefail

vllm serve /root/Qwen3-VL-Embedding-2B \
  --host :: \
  --port 8000 \
  --disable-uvicorn-access-log \
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
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --trust-remote-code \
  2>&1 | tee logs/vllm_server.log
```

`--host ::` 用于监听 IPv6；若 `ss -lntp | grep ':8000'` 仅显示
`0.0.0.0:8000`，则通过 `http://[IPv6]:8000` 访问一定失败。已经按旧参数启动时，
在激活目标 Conda 环境后使用仓库脚本安全重启：

```bash
conda activate vllm-t4-cu118-torch271
cd /root/vllm-qwen3vl-cu118-t4
chmod +x restart_vllm_server_ipv6.sh
./restart_vllm_server_ipv6.sh
```

脚本会向当前监听 8000 端口的进程发送 `SIGTERM`，最多等待 30 秒，然后使用
上述完整参数在后台启动服务；PID 写入 `vllm_server.pid`，输出写入
`logs/vllm_server.log`。脚本会自动创建日志目录；可用 `LOG_DIR` 修改默认日志
目录，或用 `LOG_FILE` 指定完整日志路径。如模型或端口不同，可在命令前设置 `MODEL_PATH`、
`SERVED_MODEL_NAME`、`PORT`、`MAX_MODEL_LEN`、`MAX_NUM_SEQS`、
`MAX_NUM_BATCHED_TOKENS`、`T4_EXECUTION_MODE` 和
`CUDAGRAPH_CAPTURE_SIZES_JSON`。服务默认导出 `OMP_NUM_THREADS=16`，也可在
命令前覆盖。脚本默认允许每个调度 iteration
合批 8 条序列，并把总 token budget 设为 4096；旧版的
`--max-num-seqs 1` 会完全禁止请求级合批，客户端增加并发只会形成等待队列。
高吞吐启动默认关闭 Uvicorn 的逐请求 access log，避免日志写盘成为前端瓶颈；
vLLM 的周期统计和 `/metrics` 仍保留。

### T4 CUDA Graph 实验模式

默认 `T4_EXECUTION_MODE=eager` 继续使用已经完成精度验证的
`--enforce-eager -O0`。下一阶段可单独开启 piecewise CUDA Graph：

```bash
T4_EXECUTION_MODE=cudagraph ./restart_vllm_server_ipv6.sh
```

该模式使用 vLLM 0.11.0 的 level 3 做 Dynamo 分段，但显式设置
`use_inductor=false`，因此不会让 TorchInductor 为 SM75 生成此前失败的 Triton
FP16 kernel；`vllm.unified_attention_with_output` 仍是分段边界，XFormers CUTLASS
attention 在 CUDA Graph 外执行，非 attention 子图使用 piecewise CUDA Graph。
pooling 模型不使用 full CUDA Graph。默认捕获大小为
`[32,64,128,256,512,1024,2048]`，超过最大捕获大小的 batch 自动回退到 eager；
需要覆盖时传入不带空格的 JSON，并确保每个值不超过 token budget：

```bash
T4_EXECUTION_MODE=cudagraph \
CUDAGRAPH_CAPTURE_SIZES_JSON='[64,128,256,512,1024]' \
./restart_vllm_server_ipv6.sh
```

这是可回退的实验入口，不代表已获得性能提升。启动日志应显示
`cudagraph_mode: 1`、`use_inductor: false`、CUDA Graph capture 完成且服务健康；
随后必须用相同模式重新运行精度检查：

```bash
T4_EXECUTION_MODE=cudagraph ./run_accuracy_check.sh
```

精度脚本在详细 JSON 和产物列表之后还会打印最终 PASS/FAIL 结论、执行模式、三项
关键指标及报告路径；比较失败时会先打印失败原因，再返回非零退出码。无论正常
完成、比较失败还是中途异常，vLLM 阶段都会先 TERM、等待 3 秒，再 KILL 残留的
API Server、EngineCore 和 vLLM spawn 进程；正常路径确认端口释放后才打印最终结论。

若捕获失败、OOM 或精度不通过，执行
`T4_EXECUTION_MODE=eager ./restart_vllm_server_ipv6.sh` 即可回到已验证路径。

如需同时开放文本、图片和视频，保持同一 IPv6 监听配置并覆盖多模态限额：

```bash
conda activate vllm-t4-cu118-torch271
cd /root/vllm-qwen3vl-cu118-t4
PORT=8000 IMAGE_LIMIT=1 VIDEO_LIMIT=1 \
MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=4096 \
./restart_vllm_server_ipv6.sh
```

文本不需要单独的限额开关。上述命令允许每个请求最多 1 张图片和 1 个视频；本项目
当前精度报告已覆盖文本、图片和图片加文本，尚未把视频纳入固定精度用例。
显式设置 `PORT=8000` 可避免 shell 中遗留的其他 `PORT` 值导致启动到错误端口。

启动日志必须包含 `Using XFormers backend on V1 engine` 和
`Supported_tasks: ['encode', 'embed']`。T4 不支持 FA2，因此
`FA2 is only supported on devices with compute capability >= 8` 是后端探测信息，
只要随后选择 XFormers 就不影响服务。模型权重不包含在本发布包中。

如果图片请求返回 `At most 0 image(s) may be provided`，说明当前运行
实例是按 `image=0` 启动的，与模型是否包含视觉编码器无关。用上述
全模态命令重启，并确认脚本输出
`Multimodal limits: {"image":1,"video":1}`。

### 吞吐量与合批调优

单张 T4 保持 `--tensor-parallel-size 1`；数据并行需要额外 GPU，不应在同一张
T4 上启动多个模型副本。先以默认的 `MAX_NUM_SEQS=8`、
`MAX_NUM_BATCHED_TOKENS=4096` 压测；视觉请求较长且显存仍有余量时，可尝试：

```bash
PORT=8000 IMAGE_LIMIT=1 VIDEO_LIMIT=1 \
MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=8192 \
./restart_vllm_server_ipv6.sh
```

若 OOM，先把 token budget 降回 4096；若 `vllm:num_requests_running` 长期达到 8
且仍有等待请求，再尝试 `MAX_NUM_SEQS=16`。使用下面的命令在压测期间确认是否
真正形成合批：

```bash
watch -n 0.2 \
  "curl --noproxy '*' -s -g http://[::1]:8000/metrics | \
   grep -E 'vllm:num_requests_(running|waiting)'"
nvidia-smi dmon -s pucm -d 1
```

只有在调度队列基本为空、GPU 仍空闲且 CPU 图片解码/HTTP 前端饱和时，才考虑
单独测试 `--api-server-count 2`。它只扩展 API 前端，不会增加单张 GPU 的模型执行
并行度，且需要单独验证多前端进程的停止和重启行为。
当前 T4 XFormers 热补丁仍要求 prefix caching 与 chunked prefill 保持关闭。

## API 验收

在另一个终端检查健康状态和模型注册：

```bash
curl -f http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
curl --noproxy '*' -g http://[::1]:8000/health
```

从开发机通过目标 IPv6 地址访问时，必须绕过本机 HTTP/HTTPS/SOCKS 代理；否则
代理可能快速返回 HTML 格式的 `521 Web Server Is Down`，该响应不是 vLLM 生成的：

```bash
# 替换为目标机的实际 IPv6，不要将真实地址提交到仓库。
TARGET_IPV6='<TARGET_IPV6>'
env \
  -u http_proxy -u https_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY \
  curl -f -g "http://[${TARGET_IPV6}]:8000/health"
```

Python 客户端还应使用 `requests.Session().trust_env = False` 或
`httpx.Client(trust_env=False)`。测速前必须先确认健康检查为 HTTP 200；521/502 等
失败响应的耗时和吞吐量不能作为模型性能数据。

发送与离线验证脚本相同模板的文本 embedding 请求：

```bash
curl -s http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-VL-Embedding-2B",
    "messages": [
      {
        "role": "system",
        "content": [{
          "type": "text",
          "text": "Retrieve images or text relevant to the user query."
        }]
      },
      {
        "role": "user",
        "content": [{
          "type": "text",
          "text": "A woman playing with her dog on a beach at sunset."
        }]
      }
    ],
    "add_generation_prompt": true,
    "add_special_tokens": true,
    "encoding_format": "float",
    "normalize": true
  }' -o embedding_response.json
```

检查输出维度、有限值和 L2 norm：

```bash
python - <<'PY'
import json
import math

with open("embedding_response.json") as f:
    response = json.load(f)
if "error" in response:
    raise RuntimeError(response["error"])

vector = response["data"][0]["embedding"]
norm = math.sqrt(sum(x * x for x in vector))
print("model:", response["model"])
print("dimension:", len(vector))
print("norm:", norm)
print("usage:", response["usage"])
assert len(vector) == 2048
assert all(math.isfinite(x) for x in vector)
assert abs(norm - 1.0) < 0.02
print("PASS")
PY
```

真实目标机实测得到 `dimension: 2048`、norm 约为 `1.0` 并输出 `PASS`。
Embedding 不生成文本，
`completion_tokens: 0` 属正常结果。当前服务默认开启视觉分支，每个请求最多
1 张图片；没有图片的请求仍只执行文本路径。

`add_special_tokens: true` 是本模型使用 LAST pooling 时的必要参数。官方
`Qwen3VLEmbedder` 会在 chat template 后追加 `<|endoftext|>`，而 vLLM 的 chat
embedding 协议默认不额外添加 special token。若省略该参数，请求会少 1 个 token，
并错误地池化前一个换行 token；该换行 token 的前向结果虽然与 Transformers
对应位置一致，但不是官方 embedding。

发送图片时使用 chat embedding 的 `image_url` 内容块。下面示例将本地图片编码
成 data URI，避免服务进程访问不到客户端文件路径：

```bash
IMAGE=/path/to/test.jpg
python - "$IMAGE" <<'PY' > visual_request.json
import base64
import json
import mimetypes
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
uri = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
print(json.dumps({
    "model": "Qwen3-VL-Embedding-2B",
    "messages": [
        {"role": "system", "content": [{"type": "text", "text":
          "Retrieve images or text relevant to the user's query."}]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": "Describe this image for retrieval."},
        ]},
    ],
    "add_generation_prompt": True,
    "add_special_tokens": True,
    "encoding_format": "float",
    "normalize": True,
}))
PY

curl -s http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  --data-binary @visual_request.json -o visual_response.json
```

## 与 Transformers 的精度对比

不要只比较一两个相似度分数；应对完全相同的输入保存两套原始 2048 维向量，
同时检查逐样本余弦/MAE/L2、全量 pairwise similarity matrix 误差和 query-to-document
Top-1 一致率。`compare_vllm_transformers.py` 已固定两边均为 FP16、相同 instruction
与 chat template、LAST pooling 和 L2 normalize，并通过图片 SHA256 指纹防止两阶段
误用不同图片。

T4 只有 16 GiB，建议先停止服务，生成 Transformers 基准；再启动 vLLM 视觉服务
生成候选向量并比较：

推荐使用单命令包装脚本。一次执行会先停止已有服务并运行 Transformers，再启动
视觉 vLLM、生成候选、自动比较、打印最终结论并关闭测试服务；所有产物写入同一个
时间戳目录，`accuracy_runs/latest` 指向该目录：

```bash
conda activate vllm-t4-cu118-torch271
cd /root/vllm-qwen3vl-cu118-t4
./run_accuracy_check.sh
```

如模型或图片路径不同，可分别设置 `MODEL_PATH` 和 `IMAGE_PATH`。下面是脚本内部
执行逻辑的等价展开命令：

```bash
conda activate vllm-t4-cu118-torch271
cd /root/vllm-qwen3vl-cu118-t4
export MODEL=/root/Qwen3-VL-Embedding-2B
export IMAGE=/root/test.jpg
export RUN_DIR="$PWD/accuracy_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
set -o pipefail
echo "accuracy outputs: $RUN_DIR"

# 1. 释放 T4 显存；如 PID 文件不存在，可用 ss -lntp 检查实际监听进程。
PID="$(ss -H -lntp 'sport = :8000' 2>/dev/null \
  | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)"
if [ -n "$PID" ]; then
  kill -TERM "$PID"
  for _ in $(seq 1 60); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
fi

# 2. 官方 Transformers wrapper，模型目录默认应包含 scripts/qwen3_vl_embedding.py。
python compare_vllm_transformers.py reference \
  --model "$MODEL" \
  --image "$IMAGE" \
  --output "$RUN_DIR/precision_transformers.json" \
  2>&1 | tee "$RUN_DIR/precision_transformers.log"

# 3. 重新启动已开启图片输入的 vLLM 服务并等待 /health 返回 200。
IMAGE_LIMIT=1 VIDEO_LIMIT=0 \
LOG_FILE="$RUN_DIR/vllm_server.log" \
PID_FILE="$RUN_DIR/vllm_server.pid" \
./restart_vllm_server_ipv6.sh
until curl -fsS --noproxy '*' -g http://[::1]:8000/health; do sleep 2; done

# 4. 对相同文本与同一图片调用 vLLM，并执行验收。
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

默认验收线为：每个相同输入的最小余弦 `>=0.995`、pairwise similarity MAE
`<=0.02`、检索 Top-1 100% 一致。这是工程回归阈值，不是模型质量基准；首次实测
还应保存报告并人工查看每个 case，之后再以同一数据集的首个通过结果建立固定基线。
一次运行的 JSON、Transformers/vLLM/比较日志和服务日志统一保存在带时间戳的
`accuracy_runs/<YYYYmmdd_HHMMSS>/`，该目录已被 Git 忽略。
若只验文本可省略两个生成向量命令中的 `--image`。官方实现当前声明的推荐环境与
本交付环境并不完全相同，因此本脚本明确使用目标机已有的 Transformers 4.57.3、
PyTorch 2.7.1+cu118 和 FP16，以隔离 dtype 与运行环境差异。

真实 T4 最终报告已通过：最小同输入 cosine `0.9998480503`、平均 cosine
`0.9999490930`、pairwise similarity MAE `0.0007551337`、最大相似度误差
`0.0027555053`，检索 Top-1 100% 一致，`failures=[]`。首次失败、内核排查、
EOS/LAST pooling 根因和逐用例结果见
[独立精度测试文档](Qwen3-VL-Embedding-Transformers-vLLM-精度对齐测试.md)。

`install_target.sh` 会自动调用 `apply_t4_xformers_hotfix.py`，将 embedding
纯 prefill 改为连续 Q/K/V 的 xFormers CUTLASS attention，从而避开 SM75 上
无法编译的 Triton Unified/Flex Attention。已经完成旧版安装时可单独执行：

```bash
python apply_t4_xformers_hotfix.py
```

优化版热补丁会识别并升级旧版：它先从 `.pre-t4-hotfix` 恢复原始
`xformers.py`，再应用新补丁，因此不需要重装或重新编译 wheel。新路径把序列长度
校验和 block-diagonal bias 构造移到 CPU metadata builder，每个 batch 只执行一次，
语言层复用同一 bias；在确认请求为无 decode、无历史 KV 的纯 prefill 后，还会跳过
后续不会读取的 paged KV cache 写入。应用后仍必须重新运行精度检查与同负载性能
基准，不能把静态验证当作真实 T4 性能结论。

需要回滚时执行 `python apply_t4_xformers_hotfix.py --restore`。该热补丁仅适用
于 pooling/embedding 的纯 prefill；必须禁用 prefix caching 和 chunked prefill，
不用于文本生成/decode。

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

## GitHub 仓库与 Release

Git 仓库使用 `https://github.com/zjpzhao/vllm-qwen3vl-cu118-t4.git`，只提交 README、文档、脚本、requirements 和 `patches/`。不要提交构建日志、`wheelhouse/`、完整 `.tar.gz` 或分卷文件，也不要使用 Git LFS 存放这些二进制；它们应通过 GitHub Release assets 发布。

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
  compare_vllm_transformers.py run_accuracy_check.sh \
  diagnose_accuracy_mismatch.py run_accuracy_diagnosis.sh \
  apply_t4_xformers_hotfix.py restart_vllm_server_ipv6.sh \
  requirements patches logs \
  'Qwen3-VL-Embedding-Transformers-vLLM-精度对齐测试.md' \
  'vLLM 0.11.0 在 T4 CUDA 11.8 上的源码编译与内核裁剪.md' \
  .gitignore
git status --short
git commit -m 'Release vLLM 0.11.0 for T4 CUDA 11.8'
git push origin HEAD
```

在 `git status` 中确认没有 `wheelhouse/` 或分卷文件后再 push。

### 生成小于 2 GiB 的 GitHub Release 分卷

假设完整材料目录为 `/path/to/workspace/vllm-qwen3vl-cu118-t4`：

```bash
cd /path/to/workspace
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
  /path/to/workspace/vllm-qwen3vl-cu118-t4-offline.tar.part-* \
  /path/to/workspace/SHA256SUMS \
  /path/to/workspace/vllm-qwen3vl-cu118-t4-offline.tar.sha256
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
