# T4 XFormers 热补丁与 Eager 优化部署说明

GitHub 仓库：<https://github.com/zjpzhao/vllm-qwen3vl-cu118-t4>

本文单独记录 Qwen3-VL-Embedding-2B 在 NVIDIA T4（SM75）、CUDA 11.8、
PyTorch 2.7.1 和 vLLM 0.11.0 环境中的热补丁性能优化、CUDA Graph/编译实验结果、
当前生产运行模式、目标机部署和精度验收方法。

## 1. 最终结论

当前生产可用方案是：

```text
vLLM V1 scheduler
+ PyTorch eager / vLLM O0
+ FP16
+ XFormers CUTLASS contiguous-prefill attention
+ Triton MRoPE SM75 kernel
+ OMP_NUM_THREADS=16
```

CUDA Graph 和 vLLM/TorchInductor 编译优化均已在真实 T4 上验证失败，不能作为
当前构建的生产模式：

1. TorchInductor/Flex Attention 在 SM75 lowering 阶段报
   `Unsupported conversion from f16 to f16`、`Unsupported rounding mode` 和
   `PassManager::run failed`；
2. `use_inductor=false` 的 piecewise CUDA Graph 仍需 TorchDynamo 捕获计算图，
   在动态 GEMM dispatch 处报
   `non-function or method super: _disabled_torch_function_impl`，尚未进入 graph
   capture 就退出。

因此生产启动必须显式或默认使用 `T4_EXECUTION_MODE=eager`。这不是普通
Transformers 单请求模式：vLLM V1 的 API、调度、请求级合批和显存管理仍然生效。

## 2. 分支与提交

性能优化保存在同一个远端分支：

```text
perf/t4-xformers-prefill
```

主要功能提交：

| 提交 | 内容 |
|---|---|
| `d2ce390` | 将 T4 XFormers prefill 热补丁由逐层重复处理优化为 batch 级元数据复用 |
| `0eedc94` | 增加可回退的 piecewise CUDA Graph 实验入口；真实 T4 后续验证失败 |
| `18c43c9` | 默认设置 `OMP_NUM_THREADS=16`，增加明确的最终精度结论 |
| `eb55889` | 精度测试结束时强制退出 vLLM 并确认端口释放 |
| `36d44e3` | 将 Transformers、vLLM、比较和清理合并为一个命令 |

第一阶段热补丁优化已经推送远端，不需要创建额外分支。

## 3. 第一阶段热补丁性能优化

提交 `d2ce390` 完成以下调整：

- 将序列长度校验从每个语言层执行一次改为每个 batch 执行一次；
- 将 `BlockDiagonalCausalMask` 构造从每层执行改为每个 batch 构造一次并复用；
- 消除 attention forward 中的 `.tolist()` 和 `torch.equal()` GPU/CPU 同步；
- 纯 prefill embedding 不再写入后续不会读取的 paged KV cache；
- 保持 Q/K/V 为连续输入，直接调用 XFormers 预编译 CUTLASS attention；
- `apply_t4_xformers_hotfix.py` 可以识别并升级旧版热补丁；
- `verify_target.py` 检查优化版 marker，防止目标机仍运行旧补丁。

原生 vLLM 0.11.0 的 XFormers prefill 会继续进入 Triton Unified Attention；该
FP16 kernel 无法在当前 Triton 3.3 + T4 SM75 环境正确 lowering。热补丁把纯
prefill pooling 路径改为：

```text
连续 Q/K/V
  → batch 级 BlockDiagonalCausalMask
  → XFormers CUTLASS memory-efficient attention
  → 跳过无用 paged KV-cache 写入
```

## 4. 热补丁适用边界

该路径只保证以下工作负载：

- `pooling` / `embedding`；
- 纯 prefill，不支持生成式 decode；
- FP16；
- 不使用 prefix caching；
- 不使用 chunked prefill；
- 不使用 sliding-window attention；
- 不使用 logits soft cap。

`Cannot use FA version 2` 是 T4 不满足 FA2 的后端探测信息；只要日志随后出现
`Using XFormers backend on V1 engine`，它就不是运行失败原因。FlashInfer 的
sampling 警告对 embedding/pooling 也没有实质影响。

## 5. 当前 Eager 模式配置

| 项目 | 当前值 |
|---|---|
| 引擎 | vLLM V1 |
| 任务 | pooling / embedding |
| dtype | FP16 |
| 执行模式 | PyTorch eager |
| vLLM 编译等级 | O0 |
| CUDA Graph | 关闭 |
| TorchDynamo | 关闭 |
| TorchInductor | 关闭 |
| Attention | XFormers CUTLASS |
| MRoPE | Triton SM75 kernel |
| Tensor Parallel | 1 |
| CPU 线程 | `OMP_NUM_THREADS=16` |
| 最大上下文 | 2048 |
| 最大并发序列 | 8 |
| 单 iteration token budget | 4096 |
| Prefix caching | 关闭 |
| Chunked prefill | 关闭 |
| 默认图片限制 | 每请求 1 张 |
| 默认视频限制 | 0 |
| 服务监听 | IPv6 `[::]:8000` |

请求执行过程：

```text
HTTP 请求
  → CPU 解析图片、文本和 chat template
  → tokenizer / multimodal processor
  → vLLM V1 scheduler 合批
  → 图片进入视觉编码器
  → Qwen3-VL language backbone
  → Triton MRoPE
  → XFormers CUTLASS causal attention
  → LAST pooling
  → L2 normalize
  → 返回 2048 维 embedding
```

## 6. Eager 模式保留的优化

`--enforce-eager -O0` 只关闭图捕获和图编译，不会把 CUDA 推理退化成 CPU 或普通
Transformers 串行执行。当前仍保留：

- vLLM V1 scheduler 和请求级 continuous batching；
- 最多 8 条序列、每 iteration 最多 4096 tokens 的调度预算；
- FP16 Tensor Core GEMM；
- cuBLAS、cuDNN 和 PyTorch CUDA kernel；
- XFormers 预编译 CUTLASS attention；
- 已验证的 Triton MRoPE；
- 多模态 processor cache 和 encoder cache；
- 关闭 Uvicorn 逐请求 access log；
- 16 个 OpenMP CPU 线程。

## 7. 关闭的优化及影响

### 7.1 CUDA Graph

关闭后，每一层 CUDA kernel 需要由 CPU 单独提交。短文本、小 batch 的 launch
overhead 更明显；并发和 batch 增大后，相对影响会降低。当前 piecewise 模式不是
精度失败，而是在引擎 profile 阶段因 TorchDynamo 不支持动态 GEMM dispatch 而
无法启动。

### 7.2 TorchInductor

关闭后，MLP、Norm、激活和 pointwise 运算不能进一步融合，kernel 数量更多；但
能够避开已经复现的 SM75 Triton/LLVM lowering 错误。当前不应通过降低精度阈值或
忽略异常来强行启用。

### 7.3 Prefix caching

关闭后，相同 system prompt、文本前缀或图片会重复计算。对于每次文本和图片都不同
的 embedding 流量，影响有限；热补丁依赖完整连续 K/V，因此必须保持关闭。

### 7.4 Chunked prefill

关闭后，每条请求必须一次完成 prefill，长请求的调度公平性较差。当前最大上下文为
2048，影响可控；热补丁同样要求它保持关闭。

## 8. 目标机部署

### 8.1 更新现有性能分支

```bash
cd /root/vllm-qwen3vl-cu118-t4

git fetch origin perf/t4-xformers-prefill
git checkout perf/t4-xformers-prefill
git merge --ff-only origin/perf/t4-xformers-prefill

git log -6 --oneline
```

注意远端引用必须写成一个完整参数：

```text
origin/perf/t4-xformers-prefill
```

不能写成 `origin perf/t4-xformers-prefill`。

### 8.2 应用或升级热补丁

```bash
conda activate vllm-t4-cu118-torch271
cd /root/vllm-qwen3vl-cu118-t4

python apply_t4_xformers_hotfix.py
mkdir -p logs
python verify_target.py 2>&1 | tee logs/verify_target_perf.log
```

`verify_target.py` 必须通过，并确认优化版 marker
`t4_prefill_attn_bias` 存在。

### 8.3 单命令精度验收

```bash
T4_EXECUTION_MODE=eager \
OMP_NUM_THREADS=16 \
./run_accuracy_check.sh
```

该命令自动完成：

1. 清理已有 vLLM；
2. 运行 Transformers reference；
3. 启动 eager vLLM；
4. 生成 vLLM 候选向量；
5. 自动比较；
6. TERM、等待 3 秒并 KILL 残留 vLLM；
7. 确认端口释放；
8. 在输出最后打印整体 PASS/FAIL 结论。

测试结束后不会保留 vLLM 服务。

### 8.4 启动生产服务

精度通过后重新启动正式服务：

```bash
T4_EXECUTION_MODE=eager \
OMP_NUM_THREADS=16 \
./restart_vllm_server_ipv6.sh
```

等待健康检查：

```bash
until curl -fsS --noproxy '*' -g http://[::1]:8000/health; do
  sleep 2
done
```

查看模型：

```bash
curl --noproxy '*' -s -g http://[::1]:8000/v1/models |
  python -m json.tool
```

确认后端与模式：

```bash
grep -Ei 'Using XFormers|compilation_config|Cudagraph|Supported_tasks' \
  logs/vllm_server.log | tail -n 100
```

预期包含：

```text
Using XFormers backend on V1 engine
Supported_tasks: ['encode', 'embed']
compilation_config={"level":0,...}
Cudagraph is disabled under eager mode
```

## 9. 精度验收结果

真实 T4 上的 Transformers/vLLM 六用例对照结果：

| 指标 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| 最低同输入 cosine | 0.9998480503 | >= 0.995 | PASS |
| 平均同输入 cosine | 0.9999490930 | — | PASS |
| Pairwise similarity MAE | 0.0007551337 | <= 0.02 | PASS |
| Pairwise similarity max error | 0.0027555053 | — | PASS |
| Retrieval Top-1 agreement | 100% | 100% | PASS |

文本、纯图片和图片加文本均通过。结果说明 eager + XFormers CUTLASS 热补丁没有
产生可观测的任务精度回归。

## 10. 后续性能优化方向

当前不再优先投入 CUDA Graph 或 TorchInductor。后续应按以下顺序优化：

1. 根据真实请求逐步测试 `MAX_NUM_SEQS=16/32`；
2. 在显存允许时增加 `MAX_NUM_BATCHED_TOKENS`；
3. 统计图片尺寸与视觉 token 分布，限制异常大图；
4. 优化图片下载、解码、Base64 和 HTTP keep-alive；
5. 对重复图片利用多模态 processor/encoder cache；
6. 多 T4 环境优先采用一卡一实例和上层负载均衡；
7. 只有在修复 Dynamo dynamic GEMM dispatch 且解决 Triton SM75 lowering 后，
   才重新评估图编译。

## 11. 生产结论

> 当前模式是在 T4/CUDA 11.8 上优先保证兼容性与精度的生产路径：保留 vLLM V1
> 合批、FP16 Tensor Core、XFormers CUTLASS、Triton MRoPE 和 16 个 CPU 线程，
> 明确关闭 CUDA Graph、TorchDynamo 与 TorchInductor。后续吞吐提升应主要依靠
> 调度参数、并发合批、图像尺寸控制和 CPU/HTTP 预处理，而不是继续强行开启编译。
