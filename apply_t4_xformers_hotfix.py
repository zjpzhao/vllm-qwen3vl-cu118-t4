#!/usr/bin/env python3
"""Apply or restore the SM75 contiguous-prefill hotfix in site-packages."""

import argparse
import importlib.util
import py_compile
import shutil
import subprocess
from pathlib import Path


MARKER = "VLLM_T4_XFORMERS_CONTIGUOUS_PREFILL"
OPTIMIZED_MARKER = "t4_prefill_attn_bias"


def locate_target() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("ERROR: vllm is not installed in the active Python environment")
    root = Path(next(iter(spec.submodule_search_locations)))
    target = root / "v1" / "attention" / "backends" / "xformers.py"
    if not target.is_file():
        raise SystemExit(f"ERROR: target file not found: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    target = locate_target()
    backup = target.with_name(target.name + ".pre-t4-hotfix")
    patch_file = (Path(__file__).resolve().parent / "patches" /
                  "vllm-t4-xformers-contiguous-prefill.patch")

    if args.restore:
        if not backup.is_file():
            raise SystemExit(f"ERROR: backup not found: {backup}")
        shutil.copy2(backup, target)
        py_compile.compile(str(target), doraise=True)
        print(f"RESTORED: {target}")
        return

    source = target.read_text()
    if MARKER in source and OPTIMIZED_MARKER in source:
        print(f"ALREADY PATCHED: {target}")
        return
    if not patch_file.is_file():
        raise SystemExit(f"ERROR: patch file not found: {patch_file}")
    if shutil.which("patch") is None:
        raise SystemExit("ERROR: GNU patch is required")

    upgrading_legacy_hotfix = MARKER in source
    if upgrading_legacy_hotfix:
        if not backup.is_file():
            raise SystemExit(
                "ERROR: legacy hotfix detected but original backup is missing: "
                f"{backup}")
        shutil.copy2(backup, target)
        print(f"UPGRADING LEGACY HOTFIX: restored {backup}")
    else:
        shutil.copy2(target, backup)
    result = subprocess.run(
        ["patch", "--batch", "--forward", str(target), str(patch_file)],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        shutil.copy2(backup, target)
        raise SystemExit(
            "ERROR: hotfix did not apply; original file was restored\n"
            + result.stdout + result.stderr)
    py_compile.compile(str(target), doraise=True)
    patched_source = target.read_text()
    if MARKER not in patched_source or OPTIMIZED_MARKER not in patched_source:
        shutil.copy2(backup, target)
        raise SystemExit(
            "ERROR: optimized markers missing after patch; original restored")
    print(f"PATCHED: {target}")
    print(f"BACKUP:  {backup}")


if __name__ == "__main__":
    main()
