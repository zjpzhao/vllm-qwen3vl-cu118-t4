#!/usr/bin/env python3
"""Compare Qwen3-VL embeddings produced by Transformers and vLLM.

The two engines are intentionally run in separate processes so the reference
model and the vLLM server never need to coexist on a 16 GiB T4.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import mimetypes
import sys
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."
OFFICIAL_MIN_PIXELS = 4 * 32 * 32
OFFICIAL_MAX_PIXELS = 1800 * 32 * 32


def add_case_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional local image; enables image-only and image+text cases.",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage Transformers/vLLM embedding accuracy check")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser(
        "reference", help="Run the official Transformers implementation")
    reference.add_argument("--model", type=Path, required=True)
    reference.add_argument("--output", type=Path,
                           default=Path("precision_transformers.json"))
    reference.add_argument("--max-length", type=int, default=2048)
    reference.add_argument(
        "--reference-module",
        type=Path,
        help="Official qwen3_vl_embedding.py; defaults to MODEL/scripts/.",
    )
    add_case_args(reference)

    candidate = subparsers.add_parser(
        "vllm", help="Call the running vLLM OpenAI-compatible endpoint")
    candidate.add_argument(
        "--endpoint",
        default="http://[::1]:8000/v1/embeddings",
    )
    candidate.add_argument("--model-name", default="Qwen3-VL-Embedding-2B")
    candidate.add_argument("--output", type=Path,
                           default=Path("precision_vllm.json"))
    candidate.add_argument("--timeout", type=float, default=300.0)
    add_case_args(candidate)

    compare = subparsers.add_parser(
        "compare", help="Compare saved reference and vLLM vectors")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--min-cosine", type=float, default=0.995)
    compare.add_argument("--max-similarity-mae", type=float, default=0.02)
    compare.add_argument("--allow-top1-mismatch", action="store_true")
    compare.add_argument("--report", type=Path)
    return parser.parse_args()


def make_cases(image: Path | None, instruction: str) -> list[dict[str, Any]]:
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must not be empty")
    if not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    cases: list[dict[str, Any]] = [
        {
            "id": "q_beach_dog",
            "role": "query",
            "instruction": instruction,
            "text": "A woman playing with her dog on a beach at sunset.",
        },
        {
            "id": "q_city_night",
            "role": "query",
            "instruction": instruction,
            "text": "A city skyline illuminated at night.",
        },
        {
            "id": "d_beach_dog",
            "role": "document",
            "instruction": instruction,
            "text": "A joyful moment with a golden retriever on a beach at sunset.",
        },
        {
            "id": "d_city_night",
            "role": "document",
            "instruction": instruction,
            "text": "Skyscrapers and city lights under a dark evening sky.",
        },
    ]
    if image is not None:
        image = image.expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"image does not exist: {image}")
        cases.extend([
            {
                "id": "d_image_only",
                "role": "document",
                "instruction": instruction,
                "image": str(image),
            },
            {
                "id": "d_image_text",
                "role": "document",
                "instruction": instruction,
                "image": str(image),
                "text": "An image supplied for multimodal retrieval.",
            },
        ])
    return cases


def case_fingerprint(case: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(case["id"].encode())
    digest.update(case["role"].encode())
    digest.update(case["instruction"].encode())
    digest.update(case.get("text", "").encode())
    if "image" in case:
        with Path(case["image"]).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in case.items() if key != "image"}
    if "image" in case:
        result["image_name"] = Path(case["image"]).name
    result["fingerprint"] = case_fingerprint(case)
    return result


def validate_vector(vector: list[float], case_id: str) -> None:
    if len(vector) != 2048:
        raise RuntimeError(f"{case_id}: expected 2048 dimensions, got {len(vector)}")
    if not all(math.isfinite(value) for value in vector):
        raise RuntimeError(f"{case_id}: embedding contains NaN or infinity")
    norm = math.sqrt(sum(value * value for value in vector))
    if abs(norm - 1.0) >= 0.02:
        raise RuntimeError(f"{case_id}: expected L2 norm near 1, got {norm}")


def save_result(path: Path, engine: str, metadata: dict[str, Any],
                cases: list[dict[str, Any]], vectors: list[list[float]]) -> None:
    records = []
    for case, vector in zip(cases, vectors, strict=True):
        validate_vector(vector, case["id"])
        record = public_case(case)
        record["embedding"] = vector
        records.append(record)
    payload = {
        "format": "qwen3vl-embedding-accuracy-v1",
        "engine": engine,
        "metadata": metadata,
        "cases": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE: {path} ({len(records)} cases)")


def load_reference_module(model: Path, explicit: Path | None):
    module_path = explicit or model / "scripts" / "qwen3_vl_embedding.py"
    module_path = module_path.expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(
            "official Transformers wrapper not found; pass --reference-module "
            "pointing to qwen3_vl_embedding.py (expected at "
            f"{module_path})")
    spec = importlib.util.spec_from_file_location(
        "qwen3vl_embedding_reference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reference module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, module_path


def run_reference(args: argparse.Namespace) -> None:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError("this reference command is intended for Tesla T4 / SM75")

    model = args.model.expanduser().resolve()
    module, module_path = load_reference_module(model, args.reference_module)
    cases = make_cases(args.image, args.instruction)
    embedder = module.Qwen3VLEmbedder(
        model_name_or_path=str(model),
        max_length=args.max_length,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    vectors = []
    for index, case in enumerate(cases, 1):
        model_input = {
            key: case[key]
            for key in ("text", "image", "instruction") if key in case
        }
        print(f"[{index}/{len(cases)}] Transformers: {case['id']}", flush=True)
        if "image" in case:
            conversation = embedder.format_model_input(**model_input)
            images, _, _ = module.process_vision_info(
                [conversation],
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
            if not images:
                raise RuntimeError(
                    f"{case['id']}: official visual preprocessing returned no image")
        embedding = embedder.process([model_input], normalize=True)[0]
        vectors.append(embedding.detach().float().cpu().tolist())
    save_result(
        args.output,
        "transformers",
        {
            "model": str(model),
            "reference_module": str(module_path),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "float16",
            "attention": "eager",
            "pooling": "LAST + L2 normalize",
        },
        cases,
        vectors,
    )


def image_data_uri(path: str) -> str:
    image = Path(path)
    mime = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def vllm_payload(case: dict[str, Any], model_name: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if "image" in case:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_data_uri(case["image"])},
        })
    if "text" in case:
        content.append({"type": "text", "text": case["text"]})
    if not content:
        content.append({"type": "text", "text": "NULL"})
    return {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": case["instruction"]}],
            },
            {"role": "user", "content": content},
        ],
        "add_generation_prompt": True,
        "encoding_format": "float",
        "normalize": True,
        "mm_processor_kwargs": {
            "min_pixels": OFFICIAL_MIN_PIXELS,
            "max_pixels": OFFICIAL_MAX_PIXELS,
        },
    }


def run_vllm(args: argparse.Namespace) -> None:
    cases = make_cases(args.image, args.instruction)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    vectors = []
    usages = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] vLLM: {case['id']}", flush=True)
        body = json.dumps(vllm_payload(case, args.model_name)).encode()
        request = urllib.request.Request(
            args.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener.open(request, timeout=args.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(
                f"{case['id']}: HTTP {error.code}: {detail}") from error
        if "error" in payload:
            raise RuntimeError(f"{case['id']}: {payload['error']}")
        vector = payload["data"][0]["embedding"]
        vectors.append([float(value) for value in vector])
        usages.append(payload.get("usage"))
    save_result(
        args.output,
        "vllm",
        {
            "endpoint": args.endpoint,
            "model": args.model_name,
            "dtype": "float16",
            "pooling": "LAST + L2 normalize",
            "usage": usages,
        },
        cases,
        vectors,
    )


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(dot(left, left) * dot(right, right))
    return dot(left, right) / denominator


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "qwen3vl-embedding-accuracy-v1":
        raise RuntimeError(f"unsupported result format: {path}")
    return payload


def retrieval_top1(records: list[dict[str, Any]]) -> dict[str, str]:
    queries = [record for record in records if record["role"] == "query"]
    documents = [record for record in records if record["role"] == "document"]
    return {
        query["id"]: max(
            documents,
            key=lambda document: dot(query["embedding"], document["embedding"]),
        )["id"]
        for query in queries
    }


def run_compare(args: argparse.Namespace) -> None:
    reference = load_result(args.reference)
    candidate = load_result(args.candidate)
    ref_by_id = {record["id"]: record for record in reference["cases"]}
    cand_by_id = {record["id"]: record for record in candidate["cases"]}
    if ref_by_id.keys() != cand_by_id.keys():
        raise RuntimeError("reference and candidate case IDs differ")

    direct = []
    ids = list(ref_by_id)
    for case_id in ids:
        ref = ref_by_id[case_id]
        cand = cand_by_id[case_id]
        if ref["fingerprint"] != cand["fingerprint"]:
            raise RuntimeError(f"{case_id}: input fingerprints differ")
        deltas = [abs(a - b) for a, b in zip(
            ref["embedding"], cand["embedding"], strict=True)]
        direct.append({
            "id": case_id,
            "cosine": cosine(ref["embedding"], cand["embedding"]),
            "mean_abs_error": sum(deltas) / len(deltas),
            "max_abs_error": max(deltas),
            "l2_distance": math.sqrt(sum(delta * delta for delta in deltas)),
        })

    similarity_errors = []
    for left_id in ids:
        for right_id in ids:
            ref_score = dot(ref_by_id[left_id]["embedding"],
                            ref_by_id[right_id]["embedding"])
            cand_score = dot(cand_by_id[left_id]["embedding"],
                             cand_by_id[right_id]["embedding"])
            similarity_errors.append(abs(ref_score - cand_score))

    ref_top1 = retrieval_top1(reference["cases"])
    cand_top1 = retrieval_top1(candidate["cases"])
    top1_agreement = sum(
        ref_top1[case_id] == cand_top1[case_id] for case_id in ref_top1
    ) / len(ref_top1)
    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "cases": direct,
        "summary": {
            "min_same_input_cosine": min(item["cosine"] for item in direct),
            "mean_same_input_cosine": sum(item["cosine"] for item in direct) / len(direct),
            "pairwise_similarity_mae": sum(similarity_errors) / len(similarity_errors),
            "pairwise_similarity_max_error": max(similarity_errors),
            "retrieval_top1_agreement": top1_agreement,
            "reference_top1": ref_top1,
            "candidate_top1": cand_top1,
        },
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_similarity_mae": args.max_similarity_mae,
            "require_top1_agreement": not args.allow_top1_mismatch,
        },
    }
    summary = report["summary"]
    failures = []
    if summary["min_same_input_cosine"] < args.min_cosine:
        failures.append("same-input cosine below threshold")
    if summary["pairwise_similarity_mae"] > args.max_similarity_mae:
        failures.append("pairwise similarity MAE above threshold")
    if top1_agreement < 1.0 and not args.allow_top1_mismatch:
        failures.append("retrieval Top-1 differs")
    report["passed"] = not failures
    report["failures"] = failures

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n", encoding="utf-8")
        print(f"WROTE: {args.report}")
    if failures:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.command == "reference":
        run_reference(args)
    elif args.command == "vllm":
        run_vllm(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
