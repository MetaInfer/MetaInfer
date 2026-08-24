#!/usr/bin/env python3
"""Task-local Custom/Triton hipprof trace and PMC collection.

All paths come from this frozen evaluator bundle and METAINFER_* runtime
inputs. No external benchmark checkout or prebuilt kernel path is used.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVALUATE = HERE / "evaluate.py"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing environment variable: {name}")
    return value


def _clear(base: Path) -> None:
    for suffix in ("", ".db", ".csv", ".hipkernel.csv", ".hiptrace.csv"):
        path = Path(str(base) + suffix)
        if path.exists():
            path.unlink()


def _run_pass(
    hipprof: Path,
    output_dir: Path,
    label: str,
    profiler_args: list[str],
    implementation: str,
    calls: int,
    timeout_s: int,
) -> dict[str, object]:
    base = output_dir / label
    _clear(base)
    manifest = output_dir / f"{label}-harness.json"
    env = dict(os.environ)
    env.update({
        "METAINFER_EVALUATION_PHASE": "profile-batch",
        "METAINFER_EVALUATION_ROLE": (
            "baseline" if implementation == "triton" else "candidate"
        ),
        "METAINFER_REPORT_PATH": str(manifest),
    })
    command = [
        str(hipprof), *profiler_args, "-o", str(base),
        sys.executable, str(EVALUATE), "profile-batch", implementation, str(calls),
    ]
    print("+", " ".join(command), flush=True)
    started_at = time.time()
    started_monotonic = time.monotonic()
    try:
        completed = subprocess.run(
            command, cwd=HERE, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        (output_dir / f"{label}.log").write_text(output, encoding="utf-8")
        raise RuntimeError(f"{label} timed out after {timeout_s}s") from exc
    (output_dir / f"{label}.log").write_text(
        completed.stdout or "", encoding="utf-8")
    print(completed.stdout or "", flush=True)
    if completed.returncode:
        raise RuntimeError(f"{label} failed with status {completed.returncode}")
    report = json.loads(manifest.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise RuntimeError(f"{label} harness did not report success")
    return {
        "label": label,
        "command": command,
        "started_at": started_at,
        "ended_at": time.time(),
        "duration_s": time.monotonic() - started_monotonic,
        "timeout_s": timeout_s,
        "calls_per_case": calls,
        "case_count": len(report.get("cases") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hipprof", default="/opt/dtk/bin/hipprof")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pmc-calls", type=int, default=1)
    parser.add_argument("--passes", choices=("trace", "diagnostic", "full"), default="full")
    parser.add_argument("--trace-timeout-s", type=int, default=600)
    parser.add_argument("--pmc-timeout-s", type=int, default=480)
    parser.add_argument("--case-ids", default="")
    parser.add_argument(
        "--implementations", default="candidate,triton",
        help="comma-separated subset of candidate,triton")
    args = parser.parse_args()
    protocol = json.loads(_required_env("METAINFER_BENCHMARK_PROTOCOL"))
    warmup = int(protocol["warmup"])
    samples = int(protocol["samples"])
    trace_calls = int(protocol["trace_calls"])
    if trace_calls != warmup + samples or samples < 3 or args.pmc_calls <= 0:
        raise RuntimeError(
            "frozen protocol requires trace_calls=warmup+samples and positive counts"
        )

    # Validate frozen runtime inputs before starting expensive profiler passes.
    _required_env("METAINFER_WEIGHT_BUNDLE")
    _required_env("METAINFER_BUILD_ARTIFACT_DIR")
    root = args.output_dir
    if root is None:
        report = Path(_required_env("METAINFER_REPORT_PATH")).resolve()
        root = report.parent / "hipprof-suite"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    hipprof = Path(args.hipprof).resolve()
    if not hipprof.is_file():
        raise RuntimeError(f"hipprof not found: {hipprof}")

    implementations = tuple(
        token.strip() for token in args.implementations.split(",")
        if token.strip())
    if not implementations or any(
        value not in ("candidate", "triton") for value in implementations
    ):
        raise RuntimeError("implementations must contain candidate and/or triton")

    case_ids = [token.strip() for token in args.case_ids.split(",") if token.strip()]
    if case_ids:
        os.environ["METAINFER_PROFILE_CASE_IDS"] = ",".join(case_ids)
    pass_records: list[dict[str, object]] = []

    for impl in implementations:
        if args.passes in ("trace", "full"):
            pass_records.append(_run_pass(
                hipprof, root, f"{impl}-trace", ["--hip-trace", "--stats"],
                impl, trace_calls, args.trace_timeout_s))
        if args.passes in ("diagnostic", "full"):
            for label, mode in (
                ("pmc", "--pmc"),
                ("read", "--pmc-read"),
                ("write", "--pmc-write"),
            ):
                pass_records.append(_run_pass(
                    hipprof, root, f"{impl}-{label}",
                    [mode, "--pmc-type", "3"], impl, args.pmc_calls,
                    args.pmc_timeout_s))

    (root / "collection.json").write_text(json.dumps({
        "passed": True,
        "implementations": implementations,
        "trace_calls": trace_calls,
        "warmup": warmup,
        "samples": samples,
        "trace_timing": "arithmetic mean after frozen warmup calls",
        "pmc_calls": args.pmc_calls,
        "passes": args.passes,
        "case_ids": case_ids,
        "pass_records": pass_records,
        "weight_bundle": str(Path(_required_env("METAINFER_WEIGHT_BUNDLE")).resolve()),
        "artifact_dir": str(Path(_required_env("METAINFER_BUILD_ARTIFACT_DIR")).resolve()),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote profiler suite to {root}")


if __name__ == "__main__":
    main()
