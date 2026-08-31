"""Trusted loader for the generated W8A8 HIP extension.

The control plane owns this file. Optimization agents edit the HIP kernel,
not the extension-loading contract.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_SOURCE_DIR = Path(__file__).resolve().parent
_LOADED = False


def _compile_source_dir() -> Path:
    configured = os.environ.get("METAINFER_W8A8_COMPILE_SOURCE_DIR")
    return Path(configured).resolve() if configured else _SOURCE_DIR


def _extension_inputs() -> tuple[list[str], list[str]]:
    """Select either an exploration HIP source or final prebuilt objects."""
    compile_source = _compile_source_dir()
    prebuilt_dir = compile_source / "prebuilt"
    prebuilt = sorted(prebuilt_dir.glob("*.o"))
    dispatch = compile_source / "csrc" / "w8a8_dispatch.cpp"
    if prebuilt:
        if not dispatch.is_file():
            raise RuntimeError(
                "prebuilt W8A8 objects exist without csrc/w8a8_dispatch.cpp"
            )
        return (
            [
                str(compile_source / "csrc" / "bindings.cpp"),
                str(dispatch),
            ],
            [str(path) for path in prebuilt],
        )
    return (
        [
            str(compile_source / "csrc" / "bindings.cpp"),
            str(compile_source / "csrc" / "w8a8_gemm_hip.hip"),
        ],
        [],
    )


def load_extension() -> None:
    """Build and load the TORCH_LIBRARY extension once per process."""
    global _LOADED
    if _LOADED:
        return
    sources, prebuilt_objects = _extension_inputs()
    build_key = os.environ.get("METAINFER_W8A8_BUILD_KEY", "default")
    safe_key = "".join(
        char for char in build_key.lower() if char in "0123456789abcdef"
    )[:24] or "default"
    load(
        name=f"metainfer_w8a8_backend_{safe_key}",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx928"],
        extra_ldflags=prebuilt_objects,
        is_python_module=False,
        with_cuda=True,
        verbose=False,
    )
    if not hasattr(torch.ops.zth_w8a8, "gemm_out"):
        raise RuntimeError(
            "W8A8 extension loaded without registering "
            "torch.ops.zth_w8a8.gemm_out"
        )
    _LOADED = True


__all__ = ["load_extension"]
