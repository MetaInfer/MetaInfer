"""Trusted parsing and summarization for hipprof PMC CSV output."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable


_DIRECT_COUNTERS = {
    "grbm_count": "GRBM_COUNT",
    "grbm_gui_active": "GRBM_GUI_ACTIVE",
    "valu_active_instructions": "SQ_ACTIVE_INST_VALU",
    "flat_lds_instructions": "SQ_INSTS_FLAT_LDS_ONLY",
    "lds_instructions": "SQ_INSTS_LDS",
    "valu_instructions": "SQ_INSTS_VALU",
    "vmem_read_instructions": "SQ_INSTS_VMEM_RD",
    "vmem_write_instructions": "SQ_INSTS_VMEM_WR",
    "lds_bank_conflicts": "SQ_LDS_BANK_CONFLICT",
    "lds_wait_instructions": "SQ_WAIT_INST_LDS",
}


def _integer(value: object) -> int:
    try:
        return int(str(value or "0"), 0)
    except ValueError:
        return 0


def _sum_prefix(row: Dict[str, str], prefix: str) -> int:
    return sum(
        _integer(value)
        for key, value in row.items()
        if key.startswith(prefix)
    )


def _candidate_rows(
    rows: Iterable[Dict[str, str]],
) -> list[Dict[str, str]]:
    return [
        row for row in rows
        if "w8a8" in str(row.get("KernelName", "")).lower()
    ]


def _last_candidate_group(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matching_indices = [
        index for index, row in enumerate(rows)
        if "w8a8" in str(row.get("KernelName", "")).lower()
    ]
    if not matching_indices:
        raise ValueError(f"no W8A8 kernel dispatch found in {path}")
    # The harness emits one or more consecutive W8A8 dispatches per operator
    # replay. Keep the final contiguous group so split-K partial + combine (or
    # another multi-kernel implementation) remains one attributable replay.
    start = end = matching_indices[-1]
    matching = set(matching_indices)
    while start - 1 in matching:
        start -= 1
    return rows[start:end + 1]


def _last_candidate(path: Path) -> Dict[str, str]:
    """Backward-compatible helper for callers that need the final dispatch."""
    return _last_candidate_group(path)[-1]


def _is_auxiliary_kernel(name: object) -> bool:
    lowered = str(name or "").lower()
    return any(token in lowered for token in (
        "combine", "reduce", "finalize", "epilogue", "copy", "pack",
    ))


def _summarize_pmc_row(row: Dict[str, str]) -> Dict[str, Any]:
    grid_workitems = _integer(row.get("grd"))
    workgroup_size = _integer(row.get("wgr"))
    begin_ns = _integer(row.get("BeginNs"))
    end_ns = _integer(row.get("EndNs"))
    l2_hits = _sum_prefix(row, "TCC_HIT[")
    l2_misses = _sum_prefix(row, "TCC_MISS[")
    grbm_count = _integer(row.get("GRBM_COUNT"))
    grbm_gui_active = _integer(row.get("GRBM_GUI_ACTIVE"))
    return {
        "kernel_name": row.get("KernelName", ""),
        "gpu_id": _integer(row.get("gpu-id")),
        "grid_workitems": grid_workitems,
        "workgroup_size": workgroup_size,
        "grid_blocks": (
            grid_workitems // workgroup_size if workgroup_size else None
        ),
        "lds_bytes": _integer(row.get("lds")),
        "scratch_bytes": _integer(row.get("scr")),
        "arch_vgpr": _integer(row.get("arch_vgpr")),
        "accum_vgpr": _integer(row.get("accum_vgpr")),
        "sgpr": _integer(row.get("sgpr")),
        "wave_size": _integer(row.get("wave_size")),
        "profiled_duration_us": (
            round((end_ns - begin_ns) / 1000.0, 3)
            if end_ns >= begin_ns else None
        ),
        "counters": {
            name: _integer(row.get(column))
            for name, column in _DIRECT_COUNTERS.items()
        },
        "l2_hits": l2_hits,
        "l2_misses": l2_misses,
        "l2_hit_rate_percent": (
            round(100.0 * l2_hits / (l2_hits + l2_misses), 3)
            if l2_hits + l2_misses else None
        ),
        "gpu_active_percent": (
            round(100.0 * grbm_gui_active / grbm_count, 3)
            if grbm_count else None
        ),
    }


def parse_pmc_csv(path: Path) -> Dict[str, Any]:
    """Return per-kernel and operator evidence for the last W8A8 replay."""
    # The harness invokes the candidate once for correctness and again for its
    # timed sample. The last matching dispatch is the closest representation
    # of the code the next optimization round will inherit.
    kernels = [_summarize_pmc_row(row) for row in _last_candidate_group(path)]
    primary = next(
        (item for item in kernels if not _is_auxiliary_kernel(item["kernel_name"])),
        kernels[0],
    )
    aggregate_hits = sum(item["l2_hits"] for item in kernels)
    aggregate_misses = sum(item["l2_misses"] for item in kernels)
    aggregate_counters = {
        name: sum(item["counters"][name] for item in kernels)
        for name in _DIRECT_COUNTERS
    }
    aggregate_duration = sum(
        float(item["profiled_duration_us"] or 0.0) for item in kernels
    )

    evidence: Dict[str, Any] = {
        "available": True,
        "profile_mode": "hipprof --pmc --pmc-type 3",
        **primary,
        "primary_kernel_name": primary["kernel_name"],
        "profiled_kernels": kernels,
        "operator_aggregate": {
            "kernel_count": len(kernels),
            "kernel_names": [item["kernel_name"] for item in kernels],
            "profiled_duration_us": round(aggregate_duration, 3),
            "counters": aggregate_counters,
            "l2_hits": aggregate_hits,
            "l2_misses": aggregate_misses,
            "l2_hit_rate_percent": (
                round(
                    100.0 * aggregate_hits / (aggregate_hits + aggregate_misses),
                    3,
                )
                if aggregate_hits + aggregate_misses else None
            ),
        },
        "interpretation_guard": (
            "PMC perturbs latency. Top-level launch/resources identify the "
            "primary GEMM kernel; profiled_kernels and operator_aggregate keep "
            "all dispatches in the final operator replay. Use the normal "
            "benchmark median/P90 for acceptance. "
            "algorithmic_bandwidth_gb_s is theoretical minimum bytes divided "
            "by time, not measured HBM traffic."
        ),
    }
    return evidence


def parse_memory_traffic_csv(
    read_path: Path,
    write_path: Path,
) -> Dict[str, Any]:
    """Apply DTK's documented TCC/EA request-size formulas."""
    reads = _last_candidate_group(read_path)
    writes = _last_candidate_group(write_path)
    read_names = [row.get("KernelName", "") for row in reads]
    write_names = [row.get("KernelName", "") for row in writes]
    if read_names != write_names:
        raise ValueError(
            "hipprof read/write replays contain different W8A8 dispatches: "
            f"read={read_names}, write={write_names}"
        )

    kernels: list[Dict[str, Any]] = []
    for read, write in zip(reads, writes):
        kernels.append(_memory_traffic_for_rows(read, write))
    read_bytes = sum(item["read_bytes"] for item in kernels)
    write_bytes = sum(item["write_bytes"] for item in kernels)
    total_bytes = read_bytes + write_bytes
    reconciliation = [
        adjustment
        for item in kernels
        for adjustment in item.get("counter_reconciliation", [])
    ]
    return {
        "source": "hipprof TCC/EA request counters",
        "read_bytes_per_operator_replay": read_bytes,
        "write_bytes_per_operator_replay": write_bytes,
        "total_bytes_per_operator_replay": total_bytes,
        # Compatibility aliases. These now represent the complete final
        # operator replay, not whichever W8A8 kernel happened to run last.
        "read_bytes_per_dispatch": read_bytes,
        "write_bytes_per_dispatch": write_bytes,
        "total_bytes_per_dispatch": total_bytes,
        "kernel_count": len(kernels),
        "kernel_names": read_names,
        "kernels": kernels,
        "counter_reconciliation": reconciliation,
        "formula": "DTK derived_counters.xml FETCH_SIZE + WRITE_SIZE",
        "read_kernel_name": read_names[0] if len(read_names) == 1 else None,
        "write_kernel_name": write_names[0] if len(write_names) == 1 else None,
    }


def _memory_traffic_for_rows(
    read: Dict[str, str],
    write: Dict[str, str],
) -> Dict[str, Any]:
    """Calculate traffic for one named kernel across read/write replays."""

    rd = _sum_prefix(read, "TCC_EA_RDREQ[")
    rd32 = _sum_prefix(read, "TCC_EA_RDREQ_32B[")
    rd1 = _sum_prefix(read, "TCC_EA1_RDREQ[")
    rd1_32 = _sum_prefix(read, "TCC_EA1_RDREQ_32B[")
    wr = _sum_prefix(write, "TCC_EA_WRREQ[")
    wr64 = _sum_prefix(write, "TCC_EA_WRREQ_64B[")
    wr1 = _sum_prefix(write, "TCC_EA1_WRREQ[")
    wr1_64 = _sum_prefix(write, "TCC_EA1_WRREQ_64B[")

    reconciliation: list[Dict[str, Any]] = []
    rd = _reconcile_request_subcounter(
        rd, rd32, "TCC_EA_RDREQ", "TCC_EA_RDREQ_32B", reconciliation
    )
    rd1 = _reconcile_request_subcounter(
        rd1,
        rd1_32,
        "TCC_EA1_RDREQ",
        "TCC_EA1_RDREQ_32B",
        reconciliation,
    )
    wr = _reconcile_request_subcounter(
        wr, wr64, "TCC_EA_WRREQ", "TCC_EA_WRREQ_64B", reconciliation
    )
    wr1 = _reconcile_request_subcounter(
        wr1,
        wr1_64,
        "TCC_EA1_WRREQ",
        "TCC_EA1_WRREQ_64B",
        reconciliation,
    )

    read_bytes = (
        rd32 * 32 + (rd - rd32) * 64
        + rd1_32 * 32 + (rd1 - rd1_32) * 64
    )
    write_bytes = (
        (wr - wr64) * 32 + wr64 * 64
        + (wr1 - wr1_64) * 32 + wr1_64 * 64
    )
    return {
        "kernel_name": read.get("KernelName", ""),
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "total_bytes": read_bytes + write_bytes,
        "counter_reconciliation": reconciliation,
    }


def _reconcile_request_subcounter(
    total: int,
    subset: int,
    total_name: str,
    subset_name: str,
    reconciliation: list[Dict[str, Any]],
) -> int:
    """Reconcile only tiny cross-replay counter skew, never large mismatch.

    hipprof may collect a request total and its size subcounter on separate
    deterministic replays. A few requests of skew are possible around replay
    boundaries. Promote the total to the observed subset only when the delta
    is no more than eight requests or 0.01%, whichever is larger, and retain
    explicit evidence. Larger inconsistencies remain hard failures.
    """
    if subset <= total:
        return total
    delta = subset - total
    tolerance = max(8, (total + 9_999) // 10_000)
    if delta > tolerance:
        raise ValueError(
            "invalid hipprof memory request counters: "
            f"{subset_name}={subset} exceeds {total_name}={total} "
            f"by {delta} (tolerance={tolerance})"
        )
    reconciliation.append(
        {
            "total_counter": total_name,
            "subset_counter": subset_name,
            "observed_total": total,
            "observed_subset": subset,
            "promoted_total": subset,
            "delta_requests": delta,
            "tolerance_requests": tolerance,
        }
    )
    return subset


def add_unprofiled_bandwidth(
    evidence: Dict[str, Any],
    median_us: object,
) -> None:
    """Combine counter-derived traffic with an unprofiled benchmark latency."""
    traffic = evidence.get("memory_traffic")
    if not isinstance(traffic, dict):
        return
    try:
        latency = float(median_us)
        total_bytes = int(traffic["total_bytes_per_operator_replay"])
    except (KeyError, TypeError, ValueError):
        return
    if latency <= 0:
        return
    traffic["unprofiled_median_us"] = latency
    bandwidth = round(
        total_bytes / latency / 1000.0, 6
    )
    traffic["counter_derived_operator_hbm_bandwidth_gb_s"] = bandwidth
    traffic["counter_derived_hbm_bandwidth_gb_s"] = bandwidth
    traffic["counter_derived_video_memory_bandwidth_gb_s"] = bandwidth
    traffic["bandwidth_semantics"] = (
        "Sum of DTK FETCH_SIZE/WRITE_SIZE-equivalent TCC/EA request bytes "
        "for every kernel in one marked operator replay, "
        "divided by the separate unprofiled whole-operator median for "
        "the identical source. Per-kernel bytes are not divided by the whole-"
        "operator latency."
    )
