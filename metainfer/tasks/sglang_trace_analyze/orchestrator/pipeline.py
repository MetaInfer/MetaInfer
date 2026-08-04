"""Pipeline — the sglang_trace_analyze core iteration loop.

Five-phase linear pipeline:

    MAPPING → BENCHMARK → ANALYZE → HINTS → SUMMARIZE → done

Each phase may internally iterate (e.g. ANALYZE loops over batch_sizes ×
stages). All analysis outputs are written to ``state_dir/analysis/`` as
the authoritative source of truth.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.orchestrator.requirements import req_field, req_field_int
from metainfer.orchestrator.state import StateStore

from .gpu_specs import GPU_SPECS, GpuSpec
from .iteration_record import (
    AnalyzeRecord,
    BenchmarkRecord,
    HintsRecord,
    MappingRecord,
    SummarizeRecord,
)
from .phases import next_phase


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def _parse_batch_sizes(raw: str) -> List[int]:
    """Parse comma-separated batch sizes, e.g. "1,4,8,16" → [1,4,8,16]."""
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _iter_n(store: StateStore) -> int:
    """Next iteration number for timeline ordering.

    Because this pipeline runs phases sequentially with a single iteration
    counter (not per-phase counters), we use a simple global counter.
    """
    run = store.load_run()
    return run.current_iteration + 1 if run else 1


class Pipeline:
    """Five-phase profiler-analysis pipeline."""

    def __init__(
        self,
        req: Dict[str, Any],
        store: StateStore,
        state_dir: Path,
        workspace_dir: Path,
    ):
        self.req = req
        self.store = store
        self.state_dir = state_dir
        self.workspace_dir = workspace_dir
        self._analysis_dir = state_dir / "analysis"

        # Extract form fields
        self.model_path = req_field(req, "model_path", default="")
        self.version = req_field(req, "version", default="dev")
        self.batch_sizes = _parse_batch_sizes(
            req_field(req, "batch_sizes", default="1")
        )
        self.mapping_batch_size = req_field_int(req, "mapping_batch_size", default=8)
        self.input_len = req_field_int(req, "input_len", default=512)
        self.output_len = req_field_int(req, "output_len", default=2000)
        self.tp_size = req_field_int(req, "tp_size", default=1)
        self.pp_size = req_field_int(req, "pp_size", default=1)
        gpu_label = req_field(req, "gpu_model", default="K100")
        self.gpu_spec: GpuSpec = GPU_SPECS.get(gpu_label, GPU_SPECS["K100"])

        # Only decode stage for now
        self.stages = ["decode"]  # future: ["prefill", "decode"]

    # ------------------------------------------------------------------ #
    #  Public entry point
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Run the full pipeline."""
        run = self.store.load_run()
        phase = run.current_phase or "mapping"

        while phase not in ("done", "failed"):
            self.store.update_run(current_phase=phase)
            self.store.append_timeline("phase_enter", {"phase": phase})

            method = getattr(self, f"_run_{phase}", None)
            if method is None:
                print(f"[pipeline] unknown phase {phase!r}, stopping")
                break

            try:
                ok = method()
            except Exception as exc:
                print(f"[pipeline] phase {phase} crashed: {exc}")
                self.store.append_timeline("phase_error", {"phase": phase, "error": str(exc)})
                self.store.update_run(finished=True, final_status="failed")
                return

            if not ok:
                print(f"[pipeline] phase {phase} returned failure, stopping")
                self.store.update_run(finished=True, final_status="failed")
                return

            self.store.append_timeline("phase_exit", {"phase": phase})
            phase = next_phase(phase)

        self.store.update_run(finished=True, final_status="success",
                              current_phase="done")
        self.store.append_timeline("run_done", {"status": "success"})

    # ================================================================== #
    #  Phase: MAPPING
    # ================================================================== #

    def _run_mapping(self) -> bool:
        """Run mapping benchmark (--disable-cuda-graph) then build the
        kernel→model-structure mapping from call stacks."""
        print("[pipeline] === MAPPING phase ===")
        n = _iter_n(self.store)
        rec = MappingRecord(batch_size=self.mapping_batch_size)
        rec.start()
        self.store.write_iteration(n, rec.to_dict())
        self.store.update_run(current_iteration=n)

        trace_dir = self.workspace_dir / "traces" / "mapping"

        # 1. Generate bench_config.json
        bench_config = self._build_bench_config(
            batch_sizes=[self.mapping_batch_size],
            output_dir=str(self.workspace_dir / "traces"),
        )
        config_path = self.state_dir / "bench_config.json"
        config_path.write_text(json.dumps(bench_config, indent=2))

        # 2. Run mapping benchmark
        script = Path(__file__).resolve().parent / "run_benchmark.py"
        print(f"[pipeline]   running mapping benchmark (batch={self.mapping_batch_size})...")
        try:
            subprocess.run(
                [
                    "python", str(script),
                    "--config", str(config_path),
                    "--mapping-only",
                ],
                check=True,
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            rec.fail("mapping benchmark timed out")
            self.store.write_iteration(n, rec.to_dict())
            return False
        except subprocess.CalledProcessError as e:
            rec.fail(f"mapping benchmark exit code {e.returncode}")
            self.store.write_iteration(n, rec.to_dict())
            return False

        # 3. Parse trace → build mapping table (rule engine)
        decode_trace_dir = trace_dir / "decode"
        if not decode_trace_dir.exists():
            # try: the wrapper may have used --profile-by-stage naming
            candidates = sorted(trace_dir.glob("*.json.gz"))
            if not candidates:
                rec.fail("no trace files found after mapping benchmark")
                self.store.write_iteration(n, rec.to_dict())
                return False
            trace_path = candidates[0]  # best effort
        else:
            traces = sorted(decode_trace_dir.glob("*.trace.json.gz"))
            if not traces:
                rec.fail("no trace files in mapping/decode/")
                self.store.write_iteration(n, rec.to_dict())
                return False
            trace_path = traces[0]

        print(f"[pipeline]   parsing trace: {trace_path}")
        mapping_entries = self._build_mapping(trace_path)
        if not mapping_entries:
            rec.fail("mapping produced zero entries — trace may be empty or format unsupported")
            self.store.write_iteration(n, rec.to_dict())
            return False

        # 4. LLM sanity check (placeholder — calls sub-agent when available)
        mapping_entries = self._llm_mapping_sanity_check(mapping_entries)

        # 5. Write mapping.json
        self._analysis_dir.mkdir(parents=True, exist_ok=True)
        mapping_file = self._analysis_dir / "mapping.json"
        mapping_file.write_text(json.dumps({
            "model": self.model_path,
            "gpu": self.gpu_spec.label,
            "mapping_batch_size": self.mapping_batch_size,
            "entries": mapping_entries,
        }, indent=2))

        confidence_issues = sum(
            1 for e in mapping_entries
            if e.get("confidence", "high") == "low"
        )

        rec.done(
            trace_dir=str(trace_dir),
            kernel_count=len(mapping_entries),
            confidence_issues=confidence_issues,
        )
        self.store.write_iteration(n, rec.to_dict())
        return True

    # ------------------------------------------------------------------ #
    #  Phase: BENCHMARK
    # ------------------------------------------------------------------ #

    def _run_benchmark(self) -> bool:
        """Run formal benchmarks (CUDA Graph ON) for each batch size."""
        print("[pipeline] === BENCHMARK phase ===")

        bench_config = self._build_bench_config(
            batch_sizes=self.batch_sizes,
            output_dir=str(self.workspace_dir / "traces"),
        )
        config_path = self.state_dir / "bench_config.json"
        config_path.write_text(json.dumps(bench_config, indent=2))

        script = Path(__file__).resolve().parent / "run_benchmark.py"

        all_ok = True
        for bs in self.batch_sizes:
            n = _iter_n(self.store)
            rec = BenchmarkRecord(batch_size=bs)
            rec.start()
            self.store.write_iteration(n, rec.to_dict())
            self.store.update_run(current_iteration=n)

            trace_dir = self.workspace_dir / "traces" / f"bs_{bs}"
            print(f"[pipeline]   batch_size={bs}")

            try:
                subprocess.run(
                    [
                        "python", str(script),
                        "--config", str(config_path),
                        "--formal-only",
                        "--single-batch", str(bs),
                    ],
                    check=True,
                    timeout=3600,
                )
            except subprocess.TimeoutExpired:
                rec.fail("timed out")
                self.store.write_iteration(n, rec.to_dict())
                all_ok = False
                continue
            except subprocess.CalledProcessError as e:
                rec.fail(f"exit code {e.returncode}")
                self.store.write_iteration(n, rec.to_dict())
                all_ok = False
                continue

            # Extract throughput/latency from sglang output log if available
            rec.done(trace_dir=str(trace_dir))
            self.store.write_iteration(n, rec.to_dict())

        # We continue even if some batches failed — ANALYZE skips them
        return all_ok or any(
            (self.workspace_dir / "traces" / f"bs_{bs}" / "decode").exists()
            for bs in self.batch_sizes
        )

    # ================================================================== #
    #  Phase: ANALYZE
    # ================================================================== #

    def _run_analyze(self) -> bool:
        """Analyze each (batch_size, stage) pair that has a trace."""
        print("[pipeline] === ANALYZE phase ===")

        mapping = _load_json(self._analysis_dir / "mapping.json", {})
        mapping_entries = mapping.get("entries", [])
        if not mapping_entries:
            print("[pipeline]   WARNING: no mapping entries — analysis may be incomplete")

        any_ok = False
        for bs in self.batch_sizes:
            for stage in self.stages:
                trace_dir = self.workspace_dir / "traces" / f"bs_{bs}" / stage
                if not trace_dir.exists():
                    print(f"[pipeline]   skipping bs_{bs}/{stage} — no trace dir")
                    continue

                traces = sorted(trace_dir.glob("*.trace.json.gz"))
                if not traces:
                    print(f"[pipeline]   skipping bs_{bs}/{stage} — no trace files")
                    continue

                n = _iter_n(self.store)
                rec = AnalyzeRecord(batch_size=bs, stage=stage)
                rec.start()
                self.store.write_iteration(n, rec.to_dict())
                self.store.update_run(current_iteration=n)

                print(f"[pipeline]   analyzing bs_{bs}/{stage} ({traces[0].name})")
                try:
                    result = self._analyze_one(
                        traces[0], mapping_entries, bs, stage
                    )
                except Exception as exc:
                    rec.fail(str(exc))
                    self.store.write_iteration(n, rec.to_dict())
                    continue

                # Write output files
                out_dir = self._analysis_dir / "batches" / f"bs_{bs}" / stage
                out_dir.mkdir(parents=True, exist_ok=True)

                (out_dir / "kernel_table.json").write_text(
                    json.dumps(result["kernel_table"], indent=2))
                (out_dir / "overlap.json").write_text(
                    json.dumps(result["overlap"], indent=2))
                (out_dir / "fuse.json").write_text(
                    json.dumps(result["fuse"], indent=2))

                mfu_vals = [
                    k.get("mfu", 0) for k in result["kernel_table"].get("kernels", [])
                    if k.get("mfu") is not None
                ]
                top_kernels = result["kernel_table"].get("kernels", [])
                rec.done(
                    kernel_count=len(top_kernels),
                    top_kernel=top_kernels[0]["kernel_name"] if top_kernels else None,
                    top_kernel_pct=top_kernels[0]["time_pct"] if top_kernels else None,
                    mfu_avg=round(sum(mfu_vals) / len(mfu_vals), 1) if mfu_vals else None,
                    fuse_hits=len(result["fuse"].get("matches", [])),
                )
                self.store.write_iteration(n, rec.to_dict())
                any_ok = True

        return any_ok

    # ================================================================== #
    #  Phase: HINTS
    # ================================================================== #

    def _run_hints(self) -> bool:
        """Generate LLM optimization hints from all analysis results."""
        print("[pipeline] === HINTS phase ===")
        n = _iter_n(self.store)
        rec = HintsRecord(
            model_used=self.model_path,
            batch_count=len(self.batch_sizes),
        )
        rec.start()
        self.store.write_iteration(n, rec.to_dict())
        self.store.update_run(current_iteration=n)

        # Collect summaries from all analyzed batches
        kernel_summaries = []
        overlap_summaries = []
        fuse_summaries = []

        for bs in self.batch_sizes:
            for stage in self.stages:
                out_dir = self._analysis_dir / "batches" / f"bs_{bs}" / stage
                kt = _load_json(out_dir / "kernel_table.json")
                ov = _load_json(out_dir / "overlap.json")
                fu = _load_json(out_dir / "fuse.json")
                if kt:
                    top3 = (kt.get("kernels", []) or [])[:3]
                    kernel_summaries.append({
                        "batch_size": bs, "stage": stage,
                        "top_kernels": top3,
                    })
                if ov:
                    overlap_summaries.append(ov)
                if fu:
                    fuse_summaries.append(fu)

        if not kernel_summaries:
            print("[pipeline]   no kernel tables — skipping hints")
            rec.fail("no analysis data available")
            self.store.write_iteration(n, rec.to_dict())
            return True  # not fatal — hints are optional

        # Generate hints (placeholder — real impl calls LLM sub-agent)
        hints = self._llm_generate_hints(
            kernel_summaries, overlap_summaries, fuse_summaries
        )
        self._analysis_dir.mkdir(parents=True, exist_ok=True)
        (self._analysis_dir / "hints.json").write_text(
            json.dumps(hints, indent=2))

        rec.done()
        self.store.write_iteration(n, rec.to_dict())
        return True

    # ================================================================== #
    #  Phase: SUMMARIZE
    # ================================================================== #

    def _run_summarize(self) -> bool:
        """Aggregate cross-batch summary."""
        print("[pipeline] === SUMMARIZE phase ===")
        n = _iter_n(self.store)
        rec = SummarizeRecord(batch_count=len(self.batch_sizes))
        rec.start()
        self.store.write_iteration(n, rec.to_dict())
        self.store.update_run(current_iteration=n)

        batch_summaries = []
        best_batch = None
        best_mfu = None

        for bs in self.batch_sizes:
            for stage in ["decode"]:
                out_dir = self._analysis_dir / "batches" / f"bs_{bs}" / stage
                kt = _load_json(out_dir / "kernel_table.json")
                if not kt:
                    batch_summaries.append({
                        "batch_size": bs, "stage": stage,
                        "status": "missing",
                    })
                    continue

                kernels = kt.get("kernels", []) or []
                mfu_vals = [k.get("mfu", 0) for k in kernels if k.get("mfu")]
                avg_mfu = round(sum(mfu_vals) / len(mfu_vals), 1) if mfu_vals else None
                top = kernels[0] if kernels else {}

                info = {
                    "batch_size": bs,
                    "stage": stage,
                    "top_kernel": top.get("kernel_name"),
                    "top_kernel_pct": top.get("time_pct"),
                    "mfu_avg": avg_mfu,
                    "kernel_count": len(kernels),
                }
                batch_summaries.append(info)

                if avg_mfu is not None and (best_mfu is None or avg_mfu > best_mfu):
                    best_mfu = avg_mfu
                    best_batch = bs

        self._analysis_dir.mkdir(parents=True, exist_ok=True)
        (self._analysis_dir / "summary.json").write_text(json.dumps({
            "model": self.model_path,
            "gpu": self.gpu_spec.label,
            "batches": batch_summaries,
        }, indent=2))

        rec.done(best_batch=best_batch, best_mfu=best_mfu)
        self.store.write_iteration(n, rec.to_dict())
        return True

    # ================================================================== #
    #  Helpers
    # ================================================================== #

    def _build_bench_config(
        self, batch_sizes: List[int], output_dir: str,
    ) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "version": self.version,
            "batch_sizes": batch_sizes,
            "mapping_batch_size": self.mapping_batch_size,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "output_dir": output_dir,
        }

    def _build_mapping(self, trace_path: Path) -> List[Dict[str, Any]]:
        """Parse a torch profiler Chrome trace and extract kernel→layer
        mappings from call stacks.

        Placeholder implementation — real logic will live in
        ``trace_parser.py`` and ``structure_mapper.py``.
        """
        # TODO: implement trace_parser.py
        print("[pipeline]   _build_mapping: parsing trace (placeholder)")
        return []

    def _llm_mapping_sanity_check(
        self, entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run LLM sanity check on mapping entries.

        Placeholder — real impl calls SubAgentManager.
        """
        # TODO: wire SubAgentManager
        for e in entries:
            e.setdefault("confidence", "high")
        return entries

    def _analyze_one(
        self,
        trace_path: Path,
        mapping_entries: List[Dict[str, Any]],
        bs: int,
        stage: str,
    ) -> Dict[str, Any]:
        """Analyze a single trace file and return kernel_table, overlap,
        and fuse results.

        Placeholder — real logic in trace_parser / flops_calculator /
        overlap_detector / fuse_matcher.
        """
        # TODO: implement real analysis pipeline
        return {
            "kernel_table": {
                "model": self.model_path,
                "gpu": self.gpu_spec.label,
                "batch_size": bs,
                "stage": stage,
                "kernels": [],
            },
            "overlap": {
                "batch_size": bs,
                "stage": stage,
                "gaps": [],
                "summary": {"total_gap_us": 0, "total_gap_pct": 0, "cuda_graph_effective": True},
            },
            "fuse": {
                "batch_size": bs,
                "stage": stage,
                "matches": [],
            },
        }

    def _llm_generate_hints(
        self,
        kernel_summaries: list,
        overlap_summaries: list,
        fuse_summaries: list,
    ) -> Dict[str, Any]:
        """Generate optimization hints via LLM.

        Placeholder — real impl calls SubAgentManager.
        """
        return {
            "bottleneck": {"kernel_or_pattern": "TBD", "reason": "", "impact_pct": 0},
            "suggestions": [],
            "surprises": [],
            "status": "skipped",
            "reason": "LLM hints not yet wired",
        }
