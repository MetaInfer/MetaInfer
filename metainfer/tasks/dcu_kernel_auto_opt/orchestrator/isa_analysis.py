"""Trusted gfx928 code-object extraction and compact ISA evidence."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence


_DTK_LLVM_CANDIDATES = (
    Path("/opt/dtk/aillvm/bin"),
    Path("/opt/dtk/llvm/bin"),
    Path("/opt/dtk/dcc/bin"),
)
_DTK_LLVM_TOOLS = (
    "llvm-objcopy",
    "clang-offload-bundler",
    "llvm-readobj",
    "llvm-objdump",
)
_KEY_OPCODE_PREFIXES = (
    "s_load_", "global_load", "global_store", "buffer_load",
    "buffer_store", "flat_load", "flat_store", "ds_read", "ds_write",
    "s_waitcnt", "s_barrier", "v_mmac", "v_fma", "v_fmac", "v_mad",
    "v_mac", "v_dot", "v_pk_",
)
_CATEGORY_PATTERNS = {
    "smem_load": r"^s_load_",
    "global_load": r"^global_load",
    "global_store": r"^global_store",
    "buffer_load": r"^buffer_load",
    "buffer_store": r"^buffer_store",
    "flat_load": r"^flat_load",
    "flat_store": r"^flat_store",
    "ds_read": r"^ds_read",
    "ds_write": r"^ds_write",
    "waitcnt": r"^s_waitcnt$",
    "barrier": r"^s_barrier$",
    "mmac": r"^v_mmac",
    "fma_mac": r"^v_(?:fma|fmac|mad|mac)",
    "dot": r"^v_dot",
    "packed_alu": r"^v_pk_",
    "dpp": r"dpp",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: list[str], *, timeout: int = 60) -> str:
    result = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def resolve_dtk_llvm_bin(explicit: Path | None = None) -> Path:
    """Find one complete DTK LLVM toolchain instead of assuming one layout."""
    configured = os.environ.get("METAINFER_DTK_LLVM_BIN")
    candidates = [
        candidate for candidate in (
            explicit,
            Path(configured) if configured else None,
            *_DTK_LLVM_CANDIDATES,
        )
        if candidate is not None
    ]
    checked: list[str] = []
    for candidate in candidates:
        missing = [
            name for name in _DTK_LLVM_TOOLS
            if not (candidate / name).is_file()
        ]
        if not missing:
            return candidate
        checked.append(f"{candidate}: missing {missing}")
    raise FileNotFoundError(
        "no complete DTK LLVM toolchain found; checked " + "; ".join(checked)
    )


def _opcodes(disassembly: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for raw in disassembly.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.endswith(":") or "file format" in stripped:
            continue
        instruction = stripped.split("//", 1)[0].strip()
        opcode = instruction.split(None, 1)[0].lower() if instruction else ""
        if opcode and not opcode.startswith("0"):
            result.append((opcode, stripped))
    return result


def summarize_isa_text(disassembly: str) -> Dict[str, Any]:
    """Summarize observable instructions without inferring performance."""
    instructions = _opcodes(disassembly)
    counts = {
        name: sum(bool(re.search(pattern, opcode)) for opcode, _ in instructions)
        for name, pattern in _CATEGORY_PATTERNS.items()
    }
    waits: Counter[str] = Counter()
    key_instructions: list[tuple[str, str]] = []
    for opcode, line in instructions:
        if opcode == "s_waitcnt":
            expression = line.split("s_waitcnt", 1)[1].split("//", 1)[0].strip()
            waits[expression] += 1
        if opcode.startswith(_KEY_OPCODE_PREFIXES) or "dpp" in opcode:
            key_instructions.append((opcode, line))

    serialized_chains = 0
    for index, (opcode, _) in enumerate(key_instructions):
        if not opcode.startswith(("global_load", "buffer_load", "flat_load")):
            continue
        window = key_instructions[index + 1:index + 5]
        wait_index = next(
            (
                offset for offset, (candidate, line) in enumerate(window)
                if candidate == "s_waitcnt" and "vmcnt(0)" in line
            ),
            None,
        )
        if wait_index is None:
            continue
        if any(
            candidate.startswith("ds_write")
            for candidate, _ in window[wait_index + 1:]
        ):
            serialized_chains += 1

    return {
        "instruction_counts": counts,
        "waitcnt_expressions": dict(sorted(waits.items())),
        "load_wait0_ds_write_windows": serialized_chains,
        "key_instruction_excerpt": [
            line for _, line in key_instructions[:96]
        ],
        "interpretation_guard": (
            "Counts and ordering come from the exact gfx928 code object. "
            "They identify audit targets, not performance conclusions."
        ),
    }


def _metadata_resources(metadata: str) -> Dict[str, Any]:
    fields = {
        "vgpr_count": r"\.vgpr_count:\s+(\d+)",
        "sgpr_count": r"\.sgpr_count:\s+(\d+)",
        "lds_bytes": r"\.group_segment_fixed_size:\s+(\d+)",
        "scratch_bytes": r"\.private_segment_fixed_size:\s+(\d+)",
        "vgpr_spill_count": r"\.vgpr_spill_count:\s+(\d+)",
        "sgpr_spill_count": r"\.sgpr_spill_count:\s+(\d+)",
        "wavefront_size": r"\.wavefront_size:\s+(\d+)",
    }
    resources: Dict[str, Any] = {}
    for name, pattern in fields.items():
        values = [int(value) for value in re.findall(pattern, metadata)]
        resources[name] = max(values) if values else None
    return resources


_FUNCTION_HEADER = re.compile(
    r"(?m)^[0-9a-fA-F]+ <([^>]+)>:\s*$"
)


def _disassembly_functions(disassembly: str) -> Dict[str, str]:
    matches = list(_FUNCTION_HEADER.finditer(disassembly))
    return {
        match.group(1): disassembly[
            match.start():matches[index + 1].start()
            if index + 1 < len(matches) else len(disassembly)
        ]
        for index, match in enumerate(matches)
    }


def _metadata_kernels(metadata: str) -> Dict[str, str]:
    starts = [
        match.start() for match in re.finditer(r"(?m)^  - \.args:\s*$", metadata)
    ]
    result: Dict[str, str] = {}
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(metadata)
        section = metadata[start:end]
        name = re.search(r"(?m)^    \.name:\s+(\S+)\s*$", section)
        if name:
            result[name.group(1)] = section
    return result


def _is_auxiliary_symbol(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in (
        "combine", "reduce", "finalize", "epilogue", "copy", "pack",
    ))


def summarize_kernel_symbols(
    disassembly: str,
    metadata: str,
    *,
    kernel_names: Sequence[str] | None = None,
    primary_kernel_name: str | None = None,
) -> Dict[str, Any]:
    """Bind instruction/resource evidence to exact gfx928 kernel symbols."""
    functions = _disassembly_functions(disassembly)
    metadata_sections = _metadata_kernels(metadata)
    if kernel_names:
        requested = list(dict.fromkeys(str(name) for name in kernel_names))
    else:
        requested = [name for name in functions if "w8a8" in name.lower()]

    kernels: list[Dict[str, Any]] = []
    unmatched: list[str] = []
    for name in requested:
        function = functions.get(name)
        if function is None:
            unmatched.append(name)
            continue
        item = summarize_isa_text(function)
        item.update({
            "kernel_name": name,
            "resources": _metadata_resources(metadata_sections.get(name, "")),
            "resource_semantics": (
                "Code-object metadata for this exact symbol. LDS is static "
                "group-segment size; use the matching PMC launch record for "
                "dynamic LDS."
            ),
        })
        kernels.append(item)

    if not kernels:
        summary = summarize_isa_text(disassembly)
        summary.update({
            "kernel_name": None,
            "resources": _metadata_resources(metadata),
            "profiled_kernels": [],
            "kernel_ownership": {
                "requested": requested,
                "matched": [],
                "unmatched": unmatched,
            },
        })
        return summary

    by_name = {item["kernel_name"]: item for item in kernels}
    primary = by_name.get(str(primary_kernel_name or ""))
    if primary is None:
        non_auxiliary = [
            item for item in kernels
            if not _is_auxiliary_symbol(item["kernel_name"])
        ]
        candidates = non_auxiliary or kernels
        primary = max(
            candidates,
            key=lambda item: (
                item["instruction_counts"].get("mmac", 0),
                item["instruction_counts"].get("global_load", 0),
            ),
        )

    summary = {
        key: value for key, value in primary.items()
        if key not in {"kernel_name", "resources"}
    }
    summary.update({
        "kernel_name": primary["kernel_name"],
        "resources": primary["resources"],
        "profiled_kernels": kernels,
        "kernel_ownership": {
            "requested": requested,
            "matched": [item["kernel_name"] for item in kernels],
            "unmatched": unmatched,
            "primary": primary["kernel_name"],
        },
    })
    return summary


def inspect_gfx928_object(
    object_path: Path,
    output_dir: Path,
    *,
    toolchain_dir: Path | None = None,
    kernel_names: Sequence[str] | None = None,
    primary_kernel_name: str | None = None,
) -> Dict[str, Any]:
    """Extract the gfx928 bundle and archive disassembly plus metadata."""
    object_path = object_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    llvm_bin = resolve_dtk_llvm_bin(toolchain_dir)
    objcopy = llvm_bin / "llvm-objcopy"
    bundler = llvm_bin / "clang-offload-bundler"
    readobj = llvm_bin / "llvm-readobj"
    objdump = llvm_bin / "llvm-objdump"

    code_object = output_dir / "gfx928.co"
    metadata_path = output_dir / "metadata.txt"
    disassembly_path = output_dir / "isa.txt"
    with tempfile.TemporaryDirectory(prefix="metainfer-isa-") as temp:
        temp_dir = Path(temp)
        copied_object = temp_dir / "kernel.o"
        fatbin = temp_dir / "hip.fatbin"
        shutil.copy2(object_path, copied_object)
        _run([
            str(objcopy), "--dump-section",
            f".hip_fatbin={fatbin}", str(copied_object),
        ])
        targets = [
            line.strip() for line in _run([
                str(bundler), "--list", "--type=o", f"--input={fatbin}",
            ]).splitlines()
            if line.strip()
        ]
        gfx928_targets = [target for target in targets if "gfx928" in target]
        if len(gfx928_targets) != 1:
            raise ValueError(
                f"expected one gfx928 offload bundle, found {gfx928_targets}"
            )
        target = gfx928_targets[0]
        _run([
            str(bundler), "--unbundle", "--type=o", f"--targets={target}",
            f"--input={fatbin}", f"--output={code_object}",
        ])

    metadata = _run([str(readobj), "--notes", str(code_object)])
    disassembly = _run([
        str(objdump), "-d", "--mcpu=gfx928", str(code_object),
    ])
    metadata_path.write_text(metadata, encoding="utf-8")
    disassembly_path.write_text(disassembly, encoding="utf-8")
    summary = summarize_kernel_symbols(
        disassembly,
        metadata,
        kernel_names=kernel_names,
        primary_kernel_name=primary_kernel_name,
    )
    summary.update({
        "available": True,
        "bundle_target": target,
        "host_object_sha256": _sha256(object_path),
        "code_object_sha256": _sha256(code_object),
        "artifact_paths": {
            "code_object": str(code_object),
            "metadata": str(metadata_path),
            "disassembly": str(disassembly_path),
        },
    })
    return summary


_ASM_PATTERN = re.compile(
    r"\b(?:asm|__asm|__asm__)\s*"
    r"(?:(?:volatile|__volatile|__volatile__)\s*)?\(\s*"
    r'"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


def analyze_inline_asm_source(source: str) -> Dict[str, Any]:
    """Distinguish compiler barriers from non-empty instruction asm."""
    bodies = [match.group(1) for match in _ASM_PATTERN.finditer(source)]
    normalized = [re.sub(r"\\[ntr]", " ", body).strip() for body in bodies]
    raw = sorted({body for body in normalized if body})
    return {
        "asm_block_count": len(bodies),
        "compiler_barrier_count": sum(not body for body in normalized),
        "raw_instruction_asm_count": sum(bool(body) for body in normalized),
        "raw_instruction_fingerprints": raw,
    }


def evaluate_inline_asm_gate(
    *,
    before: Dict[str, Any],
    after: Dict[str, Any],
    proposal: Dict[str, Any],
    isa_evidence: Dict[str, Any],
    raw_inline_asm_allowed: bool | None = None,
    verified_target_instructions: list[str] | None = None,
) -> Dict[str, Any]:
    """Require explicit intent and trusted ISA evidence for new raw asm."""
    old = set(before.get("raw_instruction_fingerprints") or [])
    new = sorted(
        set(after.get("raw_instruction_fingerprints") or []) - old
    )
    required = bool(new)
    reasons: list[str] = []
    isa_plan = proposal.get("isa_optimization")
    if required:
        if raw_inline_asm_allowed is False:
            reasons.append(
                "control-plane phase does not allow raw inline asm"
            )
        if not isinstance(isa_plan, dict):
            reasons.append("proposal.json is missing isa_optimization")
        else:
            if isa_plan.get("strategy") != "inline_asm":
                reasons.append("isa_optimization.strategy must be inline_asm")
            targets = isa_plan.get("target_instructions")
            if not isinstance(targets, list) or not targets:
                reasons.append("target_instructions must be a non-empty list")
            elif verified_target_instructions is not None and not set(
                targets
            ).issubset(set(verified_target_instructions)):
                reasons.append(
                    "target_instructions exceed the compiler limitation "
                    "verified by the preceding ISA-guided round"
                )
        if isa_evidence.get("available") is not True:
            reasons.append("trusted candidate gfx928 ISA audit is unavailable")
    return {
        "required": required,
        "passed": not reasons,
        "new_raw_instruction_asm": new,
        "reasons": reasons,
    }
