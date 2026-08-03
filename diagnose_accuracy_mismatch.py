#!/usr/bin/env python3
"""Localize Qwen3-VL embedding mismatches between Transformers and vLLM.

The diagnostic is split into two stages so it can run on a 16 GiB T4:

1. ``mrope`` compares vLLM's Triton MRoPE kernel with a pure PyTorch
   implementation. It does not load model weights and can run while the vLLM
   server is up.
2. ``hidden-scan`` loads the official Transformers model after the vLLM server
   has been stopped. It compares a saved vLLM embedding with every active
   Transformers token hidden state. A high-scoring non-last token identifies a
   pooling/token-position error; no high-scoring token points to model-forward
   or weight-loading divergence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localize a Transformers/vLLM embedding mismatch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mrope = subparsers.add_parser(
        "mrope", help="compare Triton MRoPE against pure PyTorch")
    mrope.add_argument("--model", type=Path, required=True)
    mrope.add_argument("--tokens", type=int, default=257)
    mrope.add_argument("--seed", type=int, default=20260803)
    mrope.add_argument("--output", type=Path)

    scan = subparsers.add_parser(
        "hidden-scan",
        help="compare a saved vLLM vector with every Transformers token",
    )
    scan.add_argument("--model", type=Path, required=True)
    scan.add_argument("--reference", type=Path, required=True)
    scan.add_argument("--candidate", type=Path, required=True)
    scan.add_argument("--case-id", action="append", default=[])
    scan.add_argument("--max-length", type=int, default=2048)
    scan.add_argument("--reference-module", type=Path)
    scan.add_argument("--output", type=Path)
    return parser.parse_args()


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if path is not None:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        print(f"WROTE: {path}")


def tensor_metrics(actual, expected) -> dict[str, float]:
    import torch

    left = actual.detach().float().reshape(-1)
    right = expected.detach().float().reshape(-1)
    delta = (left - right).abs()
    cosine = torch.nn.functional.cosine_similarity(
        left, right, dim=0).item()
    return {
        "cosine": float(cosine),
        "mean_abs_error": float(delta.mean().item()),
        "max_abs_error": float(delta.max().item()),
    }


def pure_torch_mrope(rope, positions, query, key):
    """Mirror MRotaryEmbedding.forward_native without a CUDA custom op."""
    import torch
    from vllm.model_executor.layers.rotary_embedding.common import \
        apply_rotary_emb_torch
    from vllm.model_executor.layers.rotary_embedding.mrope import \
        apply_interleaved_rope

    rope._match_cos_sin_cache_dtype(query)
    num_tokens = positions.shape[-1]
    cos_sin = rope.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        if rope.mrope_interleaved:
            cos = apply_interleaved_rope(cos, rope.mrope_section)
            sin = apply_interleaved_rope(sin, rope.mrope_section)
        else:
            cos = torch.cat([
                section[index]
                for index, section in enumerate(
                    cos.split(rope.mrope_section, dim=-1))
            ], dim=-1)
            sin = torch.cat([
                section[index]
                for index, section in enumerate(
                    sin.split(rope.mrope_section, dim=-1))
            ], dim=-1)

    def apply(value):
        shape = value.shape
        value = value.view(num_tokens, -1, rope.head_size)
        rotated = apply_rotary_emb_torch(
            value[..., :rope.rotary_dim], cos, sin, rope.is_neox_style)
        return torch.cat(
            (rotated, value[..., rope.rotary_dim:]), dim=-1).reshape(shape)

    return apply(query), apply(key)


def run_mrope(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import AutoConfig
    from vllm.model_executor.layers.rotary_embedding.mrope import (
        MRotaryEmbedding,
        triton_mrope,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = torch.cuda.get_device_capability(0)
    if capability != (7, 5):
        raise RuntimeError(f"expected Tesla T4 / SM75, got {capability}")

    model = args.model.expanduser().resolve()
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    rope_scaling = (
        getattr(text_config, "rope_scaling", None)
        or getattr(config, "rope_scaling", None)
        or {}
    )
    sections = rope_scaling.get("mrope_section")
    if not sections:
        raise RuntimeError(
            f"text_config.rope_scaling has no mrope_section: {rope_scaling}")

    head_size = int(getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    ))
    partial = float(getattr(text_config, "partial_rotary_factor", 1.0))
    rotary_dim = int(head_size * partial)
    num_heads = int(text_config.num_attention_heads)
    num_kv_heads = int(text_config.num_key_value_heads)
    max_positions = int(text_config.max_position_embeddings)
    base = float(getattr(text_config, "rope_theta", 10000.0))
    interleaved = bool(rope_scaling.get("mrope_interleaved", False))

    rope = MRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_positions,
        base=base,
        is_neox_style=True,
        dtype=torch.float16,
        mrope_section=list(sections),
        mrope_interleaved=interleaved,
    ).cuda()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    tokens = args.tokens
    query = torch.randn(
        tokens, num_heads * head_size,
        dtype=torch.float16, device="cuda", generator=generator)
    key = torch.randn(
        tokens, num_kv_heads * head_size,
        dtype=torch.float16, device="cuda", generator=generator)

    base_positions = torch.arange(tokens, device="cuda", dtype=torch.long)
    position_sets = {
        "text_equal_thw": base_positions.repeat(3, 1),
        "distinct_thw": torch.stack((
            base_positions,
            (base_positions * 3 + 1) % min(max_positions, 4096),
            (base_positions * 5 + 2) % min(max_positions, 4096),
        )),
    }
    results = []
    for name, positions in position_sets.items():
        ref_q, ref_k = pure_torch_mrope(
            rope, positions, query.clone(), key.clone())
        cos_sin = rope.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        actual_q, actual_k = triton_mrope(
            query.clone(), key.clone(), cos, sin,
            rope.mrope_section, rope.head_size, rope.rotary_dim,
            rope.mrope_interleaved)
        torch.cuda.synchronize()
        results.append({
            "positions": name,
            "query": tensor_metrics(actual_q, ref_q),
            "key": tensor_metrics(actual_k, ref_k),
        })

    passed = all(
        item[tensor]["cosine"] >= 0.9999
        and item[tensor]["mean_abs_error"] <= 0.001
        and item[tensor]["max_abs_error"] <= 0.01
        for item in results for tensor in ("query", "key")
    )
    payload = {
        "diagnostic": "mrope",
        "passed": passed,
        "classification": (
            "MRoPE kernel is numerically aligned"
            if passed else "MRoPE kernel diverges from pure PyTorch"
        ),
        "environment": {
            "model": str(model),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(capability),
        },
        "configuration": {
            "tokens": tokens,
            "head_size": head_size,
            "rotary_dim": rotary_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "mrope_section": list(sections),
            "mrope_interleaved": interleaved,
        },
        "results": results,
    }
    write_json(args.output, payload)
    if not passed:
        raise SystemExit(2)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def records_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in payload["cases"]}


def load_reference_module(model: Path, explicit: Path | None):
    module_path = explicit or model / "scripts" / "qwen3_vl_embedding.py"
    module_path = module_path.expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"official reference module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(
        "qwen3vl_embedding_diagnostic", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, module_path


def cosine_list(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    lnorm = math.sqrt(sum(value * value for value in left))
    rnorm = math.sqrt(sum(value * value for value in right))
    return dot / (lnorm * rnorm)


def run_hidden_scan(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 8 * 2**30:
        raise RuntimeError(
            f"only {free_bytes / 2**30:.2f} GiB GPU memory is free; stop the "
            "vLLM server before hidden-scan")

    model = args.model.expanduser().resolve()
    reference_payload = load_json(args.reference)
    candidate_payload = load_json(args.candidate)
    reference = records_by_id(reference_payload)
    candidate = records_by_id(candidate_payload)
    usages = candidate_payload.get("metadata", {}).get("usage") or []
    usage_by_id = {
        record["id"]: usage
        for record, usage in zip(candidate_payload["cases"], usages)
    }
    default_cases = ["q_beach_dog", "q_city_night"]
    case_ids = args.case_id or default_cases
    missing = [case_id for case_id in case_ids
               if case_id not in reference or case_id not in candidate]
    if missing:
        raise RuntimeError(f"case IDs missing from saved results: {missing}")

    module, module_path = load_reference_module(model, args.reference_module)
    embedder = module.Qwen3VLEmbedder(
        model_name_or_path=str(model),
        max_length=args.max_length,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    results = []
    for case_id in case_ids:
        saved_case = reference[case_id]
        model_input = {
            key: saved_case[key]
            for key in ("text", "instruction") if key in saved_case
        }
        if "image_name" in saved_case:
            raise RuntimeError(
                "hidden-scan currently defaults to text cases; image cases "
                "need an explicit image path")

        conversation = embedder.format_model_input(**model_input)
        prompt = embedder.processor.apply_chat_template(
            [conversation], add_generation_prompt=True, tokenize=False)[0]
        processed = embedder._preprocess_inputs([conversation])
        processed = {
            key: value.to(embedder.model.device)
            for key, value in processed.items()
        }
        with torch.no_grad():
            outputs = embedder.forward(processed)
        mask = outputs["attention_mask"][0].bool()
        hidden = outputs["last_hidden_state"][0, mask].float()
        hidden = F.normalize(hidden, p=2, dim=-1)
        candidate_vector = torch.tensor(
            candidate[case_id]["embedding"],
            device=hidden.device,
            dtype=torch.float32,
        )
        candidate_vector = F.normalize(candidate_vector, p=2, dim=0)
        cosines = hidden @ candidate_vector
        best_value, best_index = cosines.max(dim=0)
        last_index = hidden.shape[0] - 1

        input_ids = processed["input_ids"][0, mask].detach().cpu().tolist()
        tokens = embedder.processor.tokenizer.convert_ids_to_tokens(input_ids)
        tail_start = max(0, last_index - 15)
        tail = [{
            "index": index,
            "token_id": int(input_ids[index]),
            "token": tokens[index],
            "candidate_cosine": float(cosines[index].item()),
        } for index in range(tail_start, last_index + 1)]

        official_saved = reference[case_id]["embedding"]
        computed_last = hidden[last_index].detach().cpu().tolist()
        best_cosine = float(best_value.item())
        last_cosine = float(cosines[last_index].item())
        usage = usage_by_id.get(case_id) or {}
        vllm_prompt_tokens = usage.get("prompt_tokens")
        replay_cosine = cosine_list(official_saved, computed_last)
        if replay_cosine < 0.995:
            classification = "official Transformers replay mismatch"
        elif (vllm_prompt_tokens is not None
                and int(vllm_prompt_tokens) != len(input_ids)):
            classification = "input/token-count mismatch"
        elif best_cosine >= 0.995 and int(best_index.item()) != last_index:
            classification = "pooling/token-position mismatch"
        elif last_cosine >= 0.995:
            classification = "saved vectors align at LAST token"
        else:
            classification = "model-forward or weight-loading divergence"
        results.append({
            "case_id": case_id,
            "classification": classification,
            "prompt": prompt,
            "active_tokens": len(input_ids),
            "saved_vllm_usage": usage,
            "official_saved_vs_recomputed_last_cosine": replay_cosine,
            "candidate_vs_last_cosine": last_cosine,
            "candidate_best_token": {
                "index": int(best_index.item()),
                "token_id": int(input_ids[int(best_index.item())]),
                "token": tokens[int(best_index.item())],
                "cosine": best_cosine,
            },
            "last_token": {
                "index": last_index,
                "token_id": int(input_ids[last_index]),
                "token": tokens[last_index],
            },
            "tail": tail,
        })
        del processed, outputs, hidden, cosines
        torch.cuda.empty_cache()

    classifications = {item["classification"] for item in results}
    if "official Transformers replay mismatch" in classifications:
        overall = "official Transformers replay mismatch"
    elif "input/token-count mismatch" in classifications:
        overall = "input/token-count mismatch"
    elif all(item["candidate_vs_last_cosine"] >= 0.995 for item in results):
        overall = "aligned"
    elif any(item["candidate_best_token"]["cosine"] >= 0.995
             for item in results):
        overall = "pooling/token-position mismatch"
    else:
        overall = "model-forward or weight-loading divergence"
    payload = {
        "diagnostic": "hidden-scan",
        "classification": overall,
        "model": str(model),
        "reference_module": str(module_path),
        "results": results,
    }
    write_json(args.output, payload)


def main() -> None:
    args = parse_args()
    if args.command == "mrope":
        run_mrope(args)
    elif args.command == "hidden-scan":
        run_hidden_scan(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
