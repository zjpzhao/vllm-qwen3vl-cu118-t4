# v0.11.0-t4-cu118-torch271

这是面向 NVIDIA T4（SM75）与 glibc 2.28 的 vLLM 0.11.0 定制发布，固定 CPython 3.10、
PyTorch 2.7.1+cu118、torchvision 0.22.1+cu118 和源码构建的 XFormers
0.0.31。它仅承诺 FP16、KV Cache dtype `auto` 与 XFormers attention。

主要改动：

- 裁剪 CUDA 11.8 无法编译的 DeepGEMM BF16/FP8 实现。
- 为 CUDA 12-only MoE 算子保留明确报错的 ABI stub。
- vLLM 与 XFormers 核心扩展按 `sm_75` 构建。
- 回移 Qwen3-VL 多模态 embedding/pooling adapter，支持把生成架构转换为
  `Qwen3VLForEmbedding`，并提供端到端验证脚本。
- 增加 `apply_t4_xformers_hotfix.py`：在 T4 纯 prefill embedding 路径中使用
  xFormers CUTLASS contiguous attention，避开 SM75 上无法降低的 Triton
  Unified/Flex Attention FP16 kernel；优化版在 CPU metadata builder 中一次性
  校验长度和构造 block-diagonal bias，所有语言层复用该 bias，并在确认无 decode、
  无历史 KV 的纯 prefill 后跳过无用 paged KV cache 写入。安装脚本会自动应用、
  升级旧热补丁并保留可恢复备份。
- 将 `ray[cgraph]` 改为基础 Ray，避免引入 `cupy-cuda12x`。
- 使用 glibc 2.28 sysroot 构建并移除绝对 Conda RPATH；vLLM wheel 的
  `auditwheel show` 系统符号下限为 `manylinux_2_24_x86_64`。
- 提供完整的 143-wheel 离线环境、安装脚本、目标 T4 验证脚本和构建日志。
- 已在真实 T4、R450.191.01、glibc 2.28 和 `cuda-compat-11-8` 环境完成服务
  验收：OpenAI `/v1/embeddings` 返回 2048 维归一化向量，L2 norm 为
  `1.0000000199780135`。
- IPv6 重启脚本默认开启 Qwen3-VL 视觉分支（每请求 1 张图片，视频关闭），并新增
  Transformers/vLLM 单命令串行精度比较脚本，覆盖逐向量误差、相似度矩阵与检索
  Top-1 一致性。
- IPv6 重启脚本新增可回退的 `T4_EXECUTION_MODE=cudagraph` 实验模式：使用
  vLLM level 3 做 piecewise Dynamo 分段和 CUDA Graph capture，但显式关闭
  TorchInductor，避免重新触发 SM75 Triton lowering；默认仍为已验证的 eager/O0。
- 真实 T4 后续验证确认上述 CUDA Graph 实验在 profile 阶段被 TorchDynamo 的动态
  GEMM dispatch 阻断，尚未进入 capture；生产模式固定为 eager/O0，不开启
  TorchDynamo 或 TorchInductor。完整结论见独立优化部署文档。
- 服务和单命令串行精度脚本默认导出 `OMP_NUM_THREADS=16`；精度脚本在所有详细输出
  之后打印明确的最终 PASS/FAIL 结论和关键阈值对照，失败时仍保留非零退出码。
- `run_accuracy_check.sh` 不再要求手动分阶段：一次调用依次运行 Transformers、
  vLLM、结果比较、最终结论和服务清理，所有产物保存在同一时间戳目录。
- 精度脚本的 vLLM 阶段新增强制退出清理：正常、失败和信号退出都会先 TERM，等待
  3 秒后 KILL 残留 API Server、EngineCore 与 vLLM spawn 进程，并确认端口释放。
- 修复 chat embedding 与官方处理器的 LAST pooling 对齐：所有标准请求显式设置
  `add_special_tokens=true`，确保末尾 `<|endoftext|>` 被保留并参与 pooling。
- 真实 T4 的 6 用例文本/图片精度报告已通过：最小同输入 cosine
  `0.9998480503`，pairwise similarity MAE `0.0007551337`，Top-1 100% 一致；
  完整复现与定位过程见 `Qwen3-VL-Embedding-Transformers-vLLM-精度对齐测试.md`。
- 如需开放文本、图片和视频，可用
  `IMAGE_LIMIT=1 VIDEO_LIMIT=1 ./restart_vllm_server_ipv6.sh` 在 IPv6 `[::]`
  上启动；视频入口已开放，但本轮固定精度报告未覆盖视频样本。

## Release assets

完整离线目录约 3.1 GiB，以小于 2 GiB 的分卷上传：

```bash
cat vllm-qwen3vl-cu118-t4-offline.tar.part-* > \
  vllm-qwen3vl-cu118-t4-offline.tar
sha256sum -c vllm-qwen3vl-cu118-t4-offline.tar.sha256
tar -xf vllm-qwen3vl-cu118-t4-offline.tar
```

Release 还会单独附带 vLLM 与 XFormers 两个核心 wheel，方便已有匹配环境的
用户下载；完整部署仍推荐使用分卷离线包。

## 重要运行边界

- 目标系统：`x86_64`、CPython 3.10、glibc `>=2.28`、NVIDIA T4。
- 禁止把 CUDA toolkit `lib/stubs` 或 `/usr/local/cuda-12.9/compat` 加入
  `LD_LIBRARY_PATH`。
- R450.191.01 已配合 `cuda-compat-11-8` 在当前真实 T4 通过文本、
  纯图片和图片加文本 embedding 验证；这是遗留环境实测结果，不替代
  升级到仍受支持驱动分支的生产建议。
- 不支持 FP8/DeepGEMM、DeepSeek FP8 MLA、FA3、Ray CGraph/流水并行及本文档
  列出的高架构内核。
