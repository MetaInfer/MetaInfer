#!/usr/bin/env python3
"""Analyze task-local hipprof traces and PMC into per-shape core metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any


HARDWARE_READ_GBS = 608.357
INT8_PEAK_TOPS = 123.310


def _manifest(root: Path, label: str) -> dict[str, Any]:
    return json.loads(
        (root / f"{label}-harness.json").read_text(encoding="utf-8"))


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as db:
        names = dict(db.execute(
            "SELECT CAST(STR_ID AS TEXT), STR_NAME FROM STR_TABLE WHERE TYPE=6"))
        tables = [row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")
            if row[0].startswith("HIPOPS_")]
        rows = []
        for table in tables:
            for begin, duration, name in db.execute(
                f'SELECT BeginNs, DurationNs, CAST(Name AS TEXT) FROM "{table}"'):
                rows.append({
                    "begin_ns": int(begin), "duration_ns": int(duration),
                    "kernel_name": names.get(name, name),
                })
    return sorted(rows, key=lambda row: row["begin_ns"])


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _sum_indexed(row: dict[str, str], base: str) -> int:
    pattern = re.compile(rf"^{re.escape(base)}\[(\d+)\]$")
    return sum(
        int(_number(value)) for key, value in row.items()
        if pattern.match(key))


def _read_bytes(row: dict[str, str]) -> int:
    total = 0
    for prefix in ("TCC_EA", "TCC_EA1"):
        req = _sum_indexed(row, f"{prefix}_RDREQ")
        req32 = _sum_indexed(row, f"{prefix}_RDREQ_32B")
        total += req32 * 32 + (req - req32) * 64
    return total


def _write_bytes(row: dict[str, str]) -> int:
    total = 0
    for prefix in ("TCC_EA", "TCC_EA1"):
        req = _sum_indexed(row, f"{prefix}_WRREQ")
        req64 = _sum_indexed(row, f"{prefix}_WRREQ_64B")
        total += (req - req64) * 32 + req64 * 64
    return total


def _counter_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _begin(row: dict[str, str]) -> int:
    for key in ("BeginNs", "Begin_Ns", "StartNs", "Start_Timestamp"):
        if row.get(key):
            return int(_number(row[key]))
    raise RuntimeError("hipprof CSV has no begin timestamp column")


def _in_case(rows: list[Any], case: dict[str, Any], begin_fn) -> list[Any]:
    lo = int(case["host_monotonic_begin_ns"])
    hi = int(case["host_monotonic_end_ns"])
    return [row for row in rows if lo <= begin_fn(row) <= hi]


def _trace_case_times(
    rows: list[dict[str, Any]],
    case: dict[str, Any],
    calls: int,
    samples: int,
    epoch_offset_ns: int = 0,
) -> tuple[float, dict[str, float], int, list[float]]:
    if "host_epoch_begin_ns" in case:
        lo = int(case["host_epoch_begin_ns"])
        hi = int(case["host_epoch_end_ns"])
    else:
        # Compatibility for collections produced before the evaluator wrote
        # realtime boundaries. hipprof trace DB timestamps are CLOCK_REALTIME,
        # while PMC CSV and the old manifest use CLOCK_MONOTONIC.
        lo = int(case["host_monotonic_begin_ns"]) + epoch_offset_ns
        hi = int(case["host_monotonic_end_ns"]) + epoch_offset_ns
    selected = [row for row in rows if lo <= row["begin_ns"] <= hi]
    if not selected or len(selected) % calls:
        raise RuntimeError(
            f"{case['id']}: {len(selected)} trace dispatches not divisible by {calls}")
    if samples < 1 or samples > calls:
        raise RuntimeError(f"{case['id']}: invalid trace sample count {samples}")
    dispatches_per_call = len(selected) // calls
    patterns = [
        tuple(row["kernel_name"] for row in selected[
            index * dispatches_per_call:(index + 1) * dispatches_per_call])
        for index in range(calls)
    ]
    if any(pattern != patterns[-1] for pattern in patterns[-samples:]):
        raise RuntimeError(f"{case['id']}: unstable measured dispatch pattern")

    operator_values = []
    kernel_contributions: dict[str, list[float]] = {}
    for index in range(calls - samples, calls):
        group = selected[index * dispatches_per_call:(index + 1) * dispatches_per_call]
        operator_values.append(sum(row["duration_ns"] for row in group) / 1000.0)
        per_call: dict[str, float] = {}
        for row in group:
            name = row["kernel_name"]
            per_call[name] = per_call.get(name, 0.0) + row["duration_ns"] / 1000.0
        for name, contribution in per_call.items():
            kernel_contributions.setdefault(name, []).append(contribution)
    return statistics.fmean(operator_values), {
        name: statistics.fmean(items)
        for name, items in kernel_contributions.items()
    }, dispatches_per_call, operator_values


def _aggregate_counters(
    rows: list[dict[str, str]], case: dict[str, Any], calls: int,
) -> dict[str, Any]:
    selected = _in_case(rows, case, _begin)
    if not selected:
        raise RuntimeError(f"{case['id']}: no PMC dispatches in host interval")
    if calls < 1 or len(selected) % calls:
        raise RuntimeError(
            f"{case['id']}: {len(selected)} PMC dispatches not divisible by {calls}"
        )
    # DTK hipprof --pmc-type 3 performs six hardware replay passes internally,
    # then exports their counter groups on one merged CSV row per original
    # dispatch. Do not apply the older values[index * 6 + 5] rule to this
    # merged format: it would discard five real operator dispatches.
    # Sum traffic/caches across every original operator dispatch, and use the
    # longest dispatch for per-kernel resource metadata.
    main = max(selected, key=lambda row: _number(
        row.get("DurationNs") or row.get("DispatchNs") or 0))
    hits = sum(_sum_indexed(row, "TCC_HIT") for row in selected)
    misses = sum(_sum_indexed(row, "TCC_MISS") for row in selected)
    return {
        "dispatch_count": len(selected) // calls,
        "hbm_read_bytes": sum(_read_bytes(row) for row in selected) / calls,
        "hbm_write_bytes": sum(_write_bytes(row) for row in selected) / calls,
        "l2_hit_pct": 100.0 * hits / (hits + misses) if hits + misses else math.nan,
        "vgpr": int(_number(main.get("arch_vgpr") or main.get("VGPR_Count"))),
        "agpr": int(_number(main.get("accum_vgpr") or main.get("AGPR_Count"))),
        "sgpr": int(_number(main.get("sgpr") or main.get("SGPR_Count"))),
        "lds_bytes": int(_number(main.get("lds") or main.get("LDS_Block_Size"))),
        "scratch_bytes": int(_number(main.get("scr") or main.get("Scratch_Size"))),
        "grid_size": int(_number(main.get("grd") or main.get("GridSize"))),
        "workgroup_size": int(_number(main.get("wgr") or main.get("WorkgroupSize"))),
        "wave_size": int(_number(main.get("wave_size") or main.get("WaveSize"))),
        "waves_per_workgroup": (
            math.ceil(
                _number(main.get("wgr") or main.get("WorkgroupSize"))
                / _number(main.get("wave_size") or main.get("WaveSize"))
            )
            if _number(main.get("wgr") or main.get("WorkgroupSize")) > 0
            and _number(main.get("wave_size") or main.get("WaveSize")) > 0
            else None
        ),
        "occupancy_pct": None,
        "main_kernel": main.get("KernelName", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.profile_dir.resolve()
    collection = json.loads(
        (root / "collection.json").read_text(encoding="utf-8"))
    collection_mode = str(collection.get("passes") or "full")
    if collection_mode == "diagnostic":
        raise RuntimeError("diagnostic collection must be analyzed with a trace collection")
    rows = []
    for impl in collection["implementations"]:
        trace_manifest = _manifest(root, f"{impl}-trace")
        trace = _trace_rows(root / f"{impl}-trace.db")
        epoch_offset_ns = 0
        if trace and trace_manifest["cases"] and "host_epoch_begin_ns" not in trace_manifest["cases"][0]:
            # CLOCK_REALTIME - CLOCK_MONOTONIC is stable for the lifetime of
            # a boot. This accurately translates legacy manifests collected
            # on this worker; using the first trace row is wrong because it
            # includes each case's unmarked Triton warm-up launch.
            epoch_offset_ns = time.time_ns() - time.monotonic_ns()
        has_pmc = collection_mode == "full"
        manifests = {}
        counter_data = {}
        if has_pmc:
            manifests = {
                label: _manifest(root, f"{impl}-{label}")
                for label in ("pmc", "read", "write")
            }
            counter_data = {
                label: _counter_rows(root / f"{impl}-{label}.csv")
                for label in ("pmc", "read", "write")
            }
        expected_trace_calls = int(collection["trace_calls"])
        if int(trace_manifest.get("calls_per_case") or 0) != expected_trace_calls:
            raise RuntimeError(
                f"{impl}: trace call count differs from frozen collection protocol"
            )
        expected_pmc_calls = int(collection["pmc_calls"])
        for label, manifest in manifests.items():
            if int(manifest.get("calls_per_case") or 0) != expected_pmc_calls:
                raise RuntimeError(
                    f"{impl}: {label} call count differs from frozen collection protocol"
                )
        case_maps = {
            label: {str(item["id"]): item for item in manifest["cases"]}
            for label, manifest in manifests.items()
        }
        trace_ids = [str(item["id"]) for item in trace_manifest["cases"]]
        for label, case_map in case_maps.items():
            if set(case_map) != set(trace_ids):
                raise RuntimeError(
                    f"{impl}: {label} manifest cases differ from trace manifest"
                )

        samples = int(collection["samples"])
        for case in trace_manifest["cases"]:
            case_id = str(case["id"])
            operator_us, kernels, trace_dispatches, operator_samples = _trace_case_times(
                trace,
                case,
                int(trace_manifest["calls_per_case"]),
                samples,
                epoch_offset_ns,
            )
            if not kernels:
                raise RuntimeError(f"{case_id}: no GPU kernel dispatch")
            timed_kernel = max(kernels.items(), key=lambda item: item[1])[0]
            aligned = {label: case_maps[label][case_id] for label in case_maps}
            for label, manifest_case in aligned.items():
                if any(
                    int(manifest_case[key]) != int(case[key])
                    for key in ("m", "n", "k")
                ):
                    raise RuntimeError(
                        f"{case_id}: {label} manifest shape differs from trace"
                    )
            empty_meta = {
                "dispatch_count": 0, "hbm_read_bytes": 0, "hbm_write_bytes": 0,
                "l2_hit_pct": math.nan, "vgpr": 0, "agpr": 0, "sgpr": 0,
                "lds_bytes": 0, "scratch_bytes": 0, "grid_size": 0,
                "workgroup_size": 0, "wave_size": 0,
                "waves_per_workgroup": None, "occupancy_pct": None,
                "main_kernel": "",
            }
            meta = dict(empty_meta)
            read_meta = dict(empty_meta)
            write_meta = dict(empty_meta)
            if has_pmc:
                meta = _aggregate_counters(
                    counter_data["pmc"], aligned["pmc"],
                    int(manifests["pmc"]["calls_per_case"])
                )
                read_meta = _aggregate_counters(
                    counter_data["read"], aligned["read"],
                    int(manifests["read"]["calls_per_case"])
                )
                write_meta = _aggregate_counters(
                    counter_data["write"], aligned["write"],
                    int(manifests["write"]["calls_per_case"])
                )
            m, n, k = (int(case[key]) for key in ("m", "n", "k"))
            seconds = operator_us * 1e-6
            rd = read_meta["hbm_read_bytes"]
            wr = write_meta["hbm_write_bytes"]
            rows.append({
                "case_id": case_id, "implementation": impl,
                "M": m, "N": n, "K": k,
                "operator_mean_us": operator_us,
                "operator_median_us": statistics.median(operator_samples),
                "operator_stddev_us": (
                    statistics.stdev(operator_samples)
                    if len(operator_samples) > 1 else 0.0
                ),
                "operator_cv": (
                    statistics.stdev(operator_samples) / operator_us
                    if len(operator_samples) > 1 and operator_us > 0 else 0.0
                ),
                "operator_min_us": min(operator_samples),
                "operator_max_us": max(operator_samples),
                "operator_samples_us": operator_samples,
                "trace_dispatches_per_call": trace_dispatches,
                "timed_kernel": timed_kernel,
                "effective_int8_tops": 2 * m * n * k / seconds / 1e12,
                "hbm_read_bytes": int(rd), "hbm_write_bytes": int(wr),
                "hbm_read_gbs": rd / seconds / 1e9,
                "hbm_write_gbs": wr / seconds / 1e9,
                "hbm_total_gbs": (rd + wr) / seconds / 1e9,
                "hbm_read_attainment_pct":
                    100.0 * rd / seconds / 1e9 / HARDWARE_READ_GBS,
                "l2_hit_pct": meta["l2_hit_pct"],
                "vgpr": meta["vgpr"], "agpr": meta["agpr"],
                "sgpr": meta["sgpr"], "lds_bytes": meta["lds_bytes"],
                "scratch_bytes": meta["scratch_bytes"],
                "grid_size": meta["grid_size"],
                "workgroup_size": meta["workgroup_size"],
                "wave_size": meta["wave_size"],
                "waves_per_workgroup": meta["waves_per_workgroup"],
                "occupancy_pct": meta["occupancy_pct"],
                "pmc_dispatch_count": meta["dispatch_count"],
                "main_kernel": meta["main_kernel"] or timed_kernel,
                "trace_kernel_means_json": json.dumps(kernels, sort_keys=True),
                "hardware_read_peak_gbs": HARDWARE_READ_GBS,
                "int8_compute_peak_tops": INT8_PEAK_TOPS,
            })
    output = (args.output or root / "metrics.csv").resolve()
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (root / "metrics.json").write_text(json.dumps({
        "timing": (
            "hipprof arithmetic mean after frozen warmup; "
            "each operator sample sums all GPU kernel dispatch durations"
        ),
        "hbm_peak_read_gbs": HARDWARE_READ_GBS,
        "collection_mode": collection_mode,
        "pass_records": collection.get("pass_records") or [],
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
