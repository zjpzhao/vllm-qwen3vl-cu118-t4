# Qwen3-VL Embedding：Transformers 与 vLLM 精度对齐测试

本文记录 Qwen3-VL-Embedding-2B 在 NVIDIA T4、CUDA 11.8、glibc 2.28
目标机上，将官方 Transformers 实现与定制 vLLM 0.11.0 实现进行同输入精度
对照、定位首次差异并完成修复的全过程。

最终结论：文本和图片共 6 个用例全部通过；最小同输入余弦为
`0.9998480503`，pairwise similarity MAE 为 `0.0007551337`，检索 Top-1
完全一致。首次失败不是 CUDA、XFormers、MRoPE、权重映射或模型前向错误，而是
vLLM chat embedding 请求未显式启用 `add_special_tokens`，导致 LAST pooling
少处理末尾 `<|endoftext|>`。

## 1. 测试环境

| 项目 | 实测值 |
|---|---|
| 操作系统 | Debian GNU/Linux 10 (buster) |
| glibc | 2.28 |
| GPU | Tesla T4，SM75，15109 MiB |
| NVIDIA driver | 450.191.01，配合 `cuda-compat-11-8` |
| CUDA 用户态 | 11.8 |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu118 |
| Transformers | 4.57.3 |
| XFormers | 0.0.31，CUDA 11.8 / SM75 源码构建 |
| vLLM | 0.11.0+torch271，Qwen3-VL embedding/pooling 回移版本 |
| 模型 | `/root/Qwen3-VL-Embedding-2B` |
| dtype | FP16 |
| pooling | LAST + L2 normalize |
| vLLM attention | XFormers CUTLASS contiguous prefill |

T4 只有约 16 GiB 显存，Transformers 和 vLLM 不能同时常驻。因此测试采用两阶段
执行：先停止服务并生成 Transformers 基准，再卸载 Transformers、启动 vLLM
视觉服务并生成候选向量。

## 2. 测试用例与通过阈值

`compare_vllm_transformers.py` 固定相同 instruction、chat template、FP16、
LAST pooling、L2 normalize、图片字节和视觉 resize 上下限，覆盖：

- 两个 query 文本：海滩与狗、城市夜景；
- 两个 document 文本：与两个 query 分别对应；
- 一个纯图片 document；
- 一个图片加文本 document。

测试不只比较某一对向量，而是同时检查：

- 每个同输入 2048 维向量的 cosine、MAE、最大绝对误差和 L2 distance；
- 全部向量 pairwise similarity matrix 的 MAE 与最大误差；
- 两个 query 对全部 document 的 Top-1 检索结果。

固定验收线为：

| 指标 | 阈值 |
|---|---|
| 最小同输入 cosine | `>= 0.995` |
| pairwise similarity MAE | `<= 0.02` |
| retrieval Top-1 agreement | 100% |

## 3. 测试文件与结果目录

相关脚本：

- `run_accuracy_check.sh`：分阶段运行 Transformers、vLLM 和最终比较；
- `compare_vllm_transformers.py`：生成两套向量并计算精度指标；
- `run_accuracy_diagnosis.sh`：首次对照失败后的自动定位入口；
- `diagnose_accuracy_mismatch.py`：MRoPE 对照和 Transformers hidden-state 扫描；
- `apply_t4_xformers_hotfix.py`：T4 XFormers contiguous prefill 热补丁。

每轮结果保存到独立目录：

```text
accuracy_runs/<YYYYmmdd_HHMMSS>/
```

`accuracy_runs/latest` 指向最近一轮。本次最终通过结果位于目标机：

```text
/root/vllm-qwen3vl-cu118-t4/accuracy_runs/20260803_115417
```

主要产物：

| 文件 | 内容 |
|---|---|
| `precision_transformers.json` | 官方 Transformers 6 个原始向量 |
| `precision_vllm.json` | vLLM 6 个原始向量 |
| `precision_report.json` | 最终逐样本、矩阵和检索对照报告 |
| `precision_*.log` | 两阶段推理与比较日志 |
| `vllm_server.log` | 本轮 vLLM 服务日志 |
| `diagnose_mrope.json` | Triton MRoPE 与纯 PyTorch 对照 |
| `diagnose_hidden_scan.json` | vLLM 向量与 Transformers 各 token hidden state 对照 |
| `xformers_attention_accuracy.log` | XFormers attention 数值对照日志 |

这些目录已加入 `.gitignore`，不会把 2048 维向量和运行日志提交到仓库。

## 4. 可复现执行命令

准备官方示例图片：

```bash
conda activate vllm-t4-cu118-torch271
cd /root/vllm-qwen3vl-cu118-t4

mkdir -p accuracy_inputs
curl -fL --retry 10 \
  -o accuracy_inputs/qwen_vl_demo.jpeg \
  'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
```

先运行官方 Transformers。该阶段会停止 8000 端口上的 vLLM，并建立新的时间戳
目录：

```bash
./run_accuracy_check.sh transformers
```

再启动允许图片输入的 vLLM，生成候选向量并自动比较：

```bash
./run_accuracy_check.sh vllm
```

查看最终结论：

```bash
RUN_DIR=$(readlink -f accuracy_runs/latest)
python - "$RUN_DIR/precision_report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
print(json.dumps({
    "summary": report["summary"],
    "passed": report["passed"],
    "failures": report["failures"],
}, indent=2))
PY
```

## 5. 首次失败结果

首次 vLLM 请求没有传 `add_special_tokens: true`。虽然所有向量都经过 L2
归一化，检索 Top-1 也恰好一致，但绝对向量和相似度矩阵明显不对齐：

| 指标 | 首次结果 |
|---|---:|
| 最小同输入 cosine | 0.4323738021 |
| 平均同输入 cosine | 0.5076619949 |
| pairwise similarity MAE | 0.1738329822 |
| pairwise similarity max error | 0.4766273030 |
| retrieval Top-1 agreement | 1.0 |
| 最终判定 | FAIL |

该结果远大于 FP16 正常舍入误差，不能通过降低阈值解决。Top-1 一致只能说明这组
小样本的粗粒度语义排序碰巧相同，不能证明 embedding 实现正确。

## 6. 分层定位过程

### 6.1 XFormers attention

首先将热补丁使用的 XFormers CUTLASS GQA causal attention 与 FP32 手工参考实现
比较：

```text
cosine:        1.0
mean_abs_error: 6.667379057e-05
max_abs_error:  0.0011031628
PASS
```

因此 XFormers contiguous prefill 的注意力布局、GQA 展开、causal mask 和 scale
均数值对齐。

### 6.2 Triton MRoPE

执行：

```bash
./run_accuracy_diagnosis.sh
```

脚本分别用纯文本相同 T/H/W position 和不同 T/H/W position，对比 Triton
MRoPE 与纯 PyTorch 实现。query/key cosine 均为 `0.99999994–1.0`，平均绝对
误差约 `5.35e-05–6.03e-05`，判定通过。因此 MRoPE 不是主因。

### 6.3 Transformers 逐 token hidden-state 扫描

MRoPE 通过后，定位脚本停止 vLLM，加载官方 Transformers 模型，把已保存的 vLLM
向量与 Transformers 每个有效 token 的归一化 hidden state 比较。

两个文本用例都得到同样结论：

| 用例 | Transformers tokens | vLLM tokens | vLLM 最匹配位置 | cosine |
|---|---:|---:|---|---:|
| `q_beach_dog` | 37 | 36 | index 35，换行 token `198` | 0.9999805689 |
| `q_city_night` | 32 | 31 | index 30，换行 token `198` | 0.9999837875 |

Transformers 的真实 LAST token 是随后 index 36/31 的 `<|endoftext|>`，token ID
`151643`。vLLM 少了这个 token，因此错误地对前一个换行 token 做 LAST pooling。
同时，保存的 Transformers 向量与重新计算的 LAST hidden state cosine 接近
`0.99999998`，排除了基准生成脚本自身不一致。

这个证据还说明模型前向和权重加载实际正确：vLLM 向量与 Transformers 同位置
换行 token 的 cosine 已达到约 `0.99998`。

## 7. 根因与修复

官方 `Qwen3VLEmbedder` 先生成 chat template，再调用 `Qwen3VLProcessor`；处理器
默认启用 special tokens，从而在末尾追加 `<|endoftext|>`。vLLM
`EmbeddingChatRequest` 的 `add_special_tokens` 默认值则是 false，因为多数生成模型
的 chat template 已自行包含特殊 token。

对这个采用 LAST pooling 的 embedding 模型，两者差异会改变被池化的 token。修复
不需要重编 wheel，只需保证 chat embedding 请求显式包含：

```json
{
  "add_generation_prompt": true,
  "add_special_tokens": true
}
```

仓库已在以下位置固化该约束：

- `compare_vllm_transformers.py` 的文本和图片 API 请求；
- `verify_qwen3vl_embedding.py` 的离线 token IDs；
- README 的文本和图片请求示例；
- 构建说明中的生产请求边界。

## 8. 修复后的最终结果

修复后复用同一 Transformers 基准，只重新执行 vLLM 阶段：

```bash
./run_accuracy_check.sh vllm
```

逐用例结果：

| 用例 | cosine | mean abs error | max abs error | L2 distance |
|---|---:|---:|---:|---:|
| `q_beach_dog` | 0.9999905221 | 0.0000758467 | 0.0005015731 | 0.0043598545 |
| `q_city_night` | 0.9999879746 | 0.0000843198 | 0.0004866917 | 0.0049067218 |
| `d_beach_dog` | 0.9999918335 | 0.0000701156 | 0.0004531392 | 0.0040438964 |
| `d_city_night` | 0.9999920434 | 0.0000691550 | 0.0003616437 | 0.0039901135 |
| `d_image_only` | 0.9998841344 | 0.0002666012 | 0.0013574436 | 0.0152279594 |
| `d_image_text` | 0.9998480503 | 0.0003088227 | 0.0013656374 | 0.0174321164 |

汇总：

| 指标 | 最终结果 | 阈值 | 判定 |
|---|---:|---:|---|
| 最小同输入 cosine | 0.9998480503 | >= 0.995 | PASS |
| 平均同输入 cosine | 0.9999490930 | — | PASS |
| pairwise similarity MAE | 0.0007551337 | <= 0.02 | PASS |
| pairwise similarity max error | 0.0027555053 | — | PASS |
| retrieval Top-1 agreement | 1.0 | 1.0 | PASS |

`passed=true`，`failures=[]`。图片路径的误差略高于纯文本，但仍远小于工程阈值，
属于 FP16、视觉预处理和不同执行内核的正常数值差异。

## 9. IPv6 全模态服务

文本始终可用；图片和视频分别由 `IMAGE_LIMIT`、`VIDEO_LIMIT` 控制。以下命令让
服务监听 IPv6 `[::]:8000`，并允许每个请求最多 1 张图片和 1 个视频：

```bash
conda activate vllm-t4-cu118-torch271
cd /root/vllm-qwen3vl-cu118-t4

IMAGE_LIMIT=1 VIDEO_LIMIT=1 \
  ./restart_vllm_server_ipv6.sh
```

检查监听和健康状态：

```bash
ss -lntp | grep ':8000'
curl --noproxy '*' -f -g http://[::1]:8000/health
curl --noproxy '*' -s -g http://[::1]:8000/v1/models | python -m json.tool
```

目标机监听应显示 `[::]:8000` 或 `*:8000`，而不是只有 `0.0.0.0:8000`。
从开发机访问时要绕过环境代理：

```bash
TARGET_IPV6='<TARGET_IPV6>'
env \
  -u http_proxy -u https_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY \
  curl -f -g "http://[${TARGET_IPV6}]:8000/health"
```

本文的最终精度套件已完整验证文本、纯图片和图片加文本。上述命令同时开放视频
入口，但本轮 6 用例报告未包含视频；生产接入视频前，应另加固定视频样本，使用
同一 Transformers/vLLM 两阶段方法建立视频精度和显存基线。
