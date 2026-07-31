#!/usr/bin/env python3
"""End-to-end Qwen3-VL-Embedding validation for the T4/cu118 build."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["VLLM_T4_XFORMERS_CONTIGUOUS_PREFILL"] = "1"
if Path("/usr/local/cuda-11.8/bin/ptxas").is_file():
    os.environ.setdefault("TRITON_PTXAS_PATH",
                          "/usr/local/cuda-11.8/bin/ptxas")
os.environ.setdefault("TRITON_CACHE_DIR",
                      "/tmp/triton-cache-cu118-sm75-xformers")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", help="Optional local image for multimodal validation")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def validate_model_config(model_dir: Path) -> None:
    config = json.loads((model_dir / "config.json").read_text())
    assert config.get("architectures") == ["Qwen3VLForConditionalGeneration"], config.get(
        "architectures")
    assert config.get("model_type") == "qwen3_vl", config.get("model_type")
    assert config["text_config"]["hidden_size"] == 2048, config["text_config"][
        "hidden_size"]


def make_request(tokenizer, text: str, instruction: str, image=None):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {
            "role": "user",
            "content": content,
        },
    ]
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    request = {"prompt": prompt}
    if image is not None:
        request["multi_modal_data"] = {"image": image}
    return request


def check_embeddings(outputs, expected_count: int) -> list[list[float]]:
    embeddings = [output.outputs.embedding for output in outputs]
    assert len(embeddings) == expected_count, len(embeddings)
    for index, embedding in enumerate(embeddings):
        assert len(embedding) == 2048, (index, len(embedding))
        norm = math.sqrt(sum(value * value for value in embedding))
        assert abs(norm - 1.0) <= 2e-2, (index, norm)
        assert all(math.isfinite(value) for value in embedding), index
    return embeddings


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model).resolve()
    validate_model_config(model_dir)

    import torch
    from PIL import Image
    from vllm import LLM
    from vllm.config import PoolerConfig

    assert torch.__version__ == "2.7.1+cu118", torch.__version__
    assert torch.cuda.get_device_capability(0) == (7, 5)

    limits = {"image": 1 if args.image else 0, "video": 0}
    llm = LLM(
        model=str(model_dir),
        runner="pooling",
        convert="embed",
        dtype="half",
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        pooler_config=PoolerConfig(pooling_type="LAST", normalize=True),
        limit_mm_per_prompt=limits,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        disable_log_stats=True,
    )
    tokenizer = llm.get_tokenizer()
    instruction = "Retrieve images or text relevant to the user's query."
    requests = [
        make_request(tokenizer, "A woman playing with her dog on a beach at sunset.",
                     instruction),
        make_request(tokenizer,
                     "A joyful moment with a golden retriever on a beach at sunset.",
                     instruction),
    ]

    image = None
    if args.image:
        image = Image.open(args.image).convert("RGB")
        requests.append(make_request(tokenizer, "Describe the image for retrieval.",
                                     instruction, image=image))

    embeddings = check_embeddings(llm.embed(requests, use_tqdm=False), len(requests))
    cosine = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
    assert math.isfinite(cosine)
    print("PASS:", {
        "model": str(model_dir),
        "requests": len(requests),
        "dimension": len(embeddings[0]),
        "text_pair_cosine": cosine,
        "image_validated": image is not None,
        "dtype": "float16",
        "pooling": "LAST + L2 normalize",
        "attention": "xFormers CUTLASS contiguous prefill",
    })


if __name__ == "__main__":
    main()
