"""Performance oracle: fixed, deterministic benchmark harness for HTTP-served
inference frameworks.

This replaces the old E-step pattern of "perf-tester agent writes perf.sh
and runs it". The agent-written benchmark was unreliable across iterations
(different prompts, different concurrency, different warmup policy, different
tokenizer choice), so iter(N) vs iter(N-1) comparisons were meaningless.

The perf oracle fixes this by owning the benchmark:

  - Fixed prompt set (``prompts.yaml`` — immutable from agent perspective).
  - Fixed concurrency sweep (``[1, 4, 16]`` by default).
  - Fixed warmup policy (drop the first ``warmup_requests`` per run).
  - Fixed metric definitions (see :func:`_aggregate_run`).
  - Multiple runs per concurrency level (median for the headline number,
    stdev for noise estimate).
  - GPU telemetry sampled in the background (best-effort).
  - Target comparison if ``req['target_tokens_per_sec']`` is set.
  - Regression check vs the previous iteration's perf dict.

Output: ``perf-report.json`` with a structured schema (see :class:`PerfReport`).
Top-level backward-compatible fields (``tokens_per_sec``, ``p50_ms`` etc.)
are still emitted so existing pipeline code that does
``{k: float(v) for k, v in obj.items()}`` keeps working.

The harness uses the SAME ``serve.sh`` the correctness oracle validated,
so the artifact contract is unchanged. No agent prep is needed for E.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .gpu import GpuTelemetry
from ..hardware import (
    HardwareProfileError,
    execution_environment,
    materialize_hardware_binding,
    profiler_artifact_globs,
    profiler_launch_command,
    require_hardware_profile,
)


PROMPTS_FILE = Path(__file__).parent / "data" / "perf_prompts.yaml"

# Concurrency ladder. The order matters: c=1 measures pure batch=1 latency,
# c=4 is a typical serving load, c=16 stresses continuous batching.
DEFAULT_CONCURRENCY_LADDER = (1, 4, 16)

# Number of independent runs per (concurrency, prompt-set) combination.
# We take the median tokens_per_sec as the headline and report stdev so
# the F-step planner can see whether differences between iterations are
# larger than the run-to-run noise.
RUNS_PER_LEVEL = 3

# Drop this many initial requests per run before measuring. Warmup
# absorbs: first-call kernel compilation, weight preloading, KV cache
# pool allocation, JIT cache misses.
WARMUP_REQUESTS = 4

# Per-request HTTP timeout (seconds). Generous because long-output
# prompts can legitimately take 30+ seconds at low concurrency on big
# models.
REQUEST_TIMEOUT_S = 120

# How long to wait for serve.sh to come up before giving up.
# Default is generous — non-NVIDIA accelerators (Hygon DCU, etc.) can
# take several minutes to load a 7-8B model; 70B+ can take 10+ min.
# Override with METAINFER_PERF_STARTUP_TIMEOUT_S env var.
DEFAULT_STARTUP_TIMEOUT_S = 600
STARTUP_TIMEOUT_S = DEFAULT_STARTUP_TIMEOUT_S


def _resolve_startup_timeout_s() -> int:
    """Pick the startup timeout, honoring env override. Clamped to
    [30, 3600]."""
    raw = os.environ.get("METAINFER_PERF_STARTUP_TIMEOUT_S")
    if raw:
        try:
            v = int(raw)
            return max(30, min(3600, v))
        except ValueError:
            pass
    return DEFAULT_STARTUP_TIMEOUT_S


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class RequestResult:
    """One HTTP request's outcome."""
    case_id: str
    elapsed_s: float
    http_status: Optional[int] = None
    output_text: str = ""
    output_tokens: int = 0
    input_tokens: int = 0
    error: Optional[str] = None


@dataclass
class PerfRunResult:
    """Aggregated numbers from one (concurrency, run) execution."""
    concurrency: int
    run_idx: int
    tokens_per_sec: float
    ms_per_request: float
    p50_ms: float
    p99_ms: float
    total_output_tokens: int
    total_input_tokens: int
    total_wall_s: float
    num_requests: int
    errors: int = 0
    raw_latencies_ms: List[float] = field(default_factory=list)


@dataclass
class PerfReport:
    """Structured perf report — serialized to perf-report.json."""
    schema_version: int = 2
    task_id: str = ""
    iteration: Optional[int] = None
    measured_at: float = 0.0
    methodology: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    primary: Dict[str, Any] = field(default_factory=dict)
    target: Optional[Dict[str, Any]] = None
    regression: Optional[Dict[str, Any]] = None
    profile_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    # Backward-compat: flat top-level fields so old _read_perf_report
    # keeps working. Mirror the primary (saturated-concurrency) run.
    tokens_per_sec: float = 0.0
    ms_per_request: float = 0.0
    p50_ms: float = 0.0
    p99_ms: float = 0.0
    total_tokens: float = 0.0
    total_wall_s: float = 0.0
    concurrency: int = 0
    num_requests: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        return d


# --------------------------------------------------------------------------- #
# Tokenizer resolution
# --------------------------------------------------------------------------- #

def _resolve_token_counter(req: Dict[str, Any]) -> Tuple[str, Any]:
    """Return ``(source_label, counter_fn)``.

    The counter_fn(text) -> int. We try in order:
      1. HuggingFace tokenizer if ``target_model`` is a HF snapshot dir
         and transformers is installed.
      2. tiktoken if installed (cl100k_base — close to GPT-style BPE).
      3. Heuristic: count whitespace-split tokens + CJK char count.

    Source label is recorded in the methodology so cross-iteration
    comparisons can detect methodology drift (you should compare
    numbers only when the source matches).
    """
    model_dir = req.get("target_model") or (req.get("answers") or {}).get("target_model")
    if model_dir and Path(model_dir).exists():
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained(str(model_dir))
            return "huggingface:" + str(model_dir), lambda s: len(tok.encode(s))
        except Exception:  # noqa: BLE001 — fall through
            pass
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return "tiktoken:cl100k_base", lambda s: len(enc.encode(s))
    except Exception:  # noqa: BLE001
        pass
    return "heuristic:whitespace+cjk", _heuristic_count


def _heuristic_count(text: str) -> int:
    """Whitespace tokens for ASCII + per-char for CJK. Rough but stable."""
    if not text:
        return 0
    # CJK Unicode ranges — count each char as ~1 token (close to BPE for
    # Chinese, underestimates Japanese kana slightly).
    cjk = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF
              or 0x3040 <= ord(c) <= 0x30FF
              or 0xAC00 <= ord(c) <= 0xD7AF)
    ascii_tokens = len([t for t in text.encode("ascii", errors="ignore").decode("ascii", errors="ignore").split() if t])
    return cjk + ascii_tokens


# --------------------------------------------------------------------------- #
# Server lifecycle (mostly copied from the correctness oracle — same contract)
# --------------------------------------------------------------------------- #

def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(
    serve_sh: Path, port: int, report_dir: Path,
    model_dir: Optional[str] = None,
    profile_dir: Optional[Path] = None,
    profile_duration_s: int = 1800,
    extra_env: Optional[Dict[str, str]] = None,
    profiler_prefix: Optional[List[str]] = None,
) -> subprocess.Popen:
    # Kill any orphan process holding GPU VRAM before booting. The perf
    # sweep loads the model once and stresses it; any leftover allocation
    # from a previous run causes spurious OOM or skewed throughput numbers.
    from metainfer.orchestrator.gpu_preflight import preflight_gpu
    preflight_gpu(label="e-oracle")

    log_fp = open(report_dir / "perf-server.stdout.log", "wb")
    err_fp = open(report_dir / "perf-server.stderr.log", "wb")
    env = dict(os.environ)
    env["METAINFER_PERF_ORACLE_PORT"] = str(port)
    if model_dir:
        env["MODEL_DIR"] = str(model_dir)
    if extra_env:
        env.update(extra_env)
    env.setdefault("PYTHONHASHSEED", "0")
    # Enable the in-framework profiler hook. The contract in
    # notebooks/00_contracts/profiling_contracts.md REQUIRES the framework
    # to honor these vars. We set duration to a large value as a backstop;
    # the real stop trigger is the SIGTERM we send at the end of the sweep
    # (the framework MUST call profiler.stop() + export on SIGTERM).
    if profiler_prefix:
        # The platform profile owns profiling. Do not require generated C++
        # code to emulate the old CUDA/PyTorch profile hook.
        env["METAINFER_PROFILE"] = "0"
    elif profile_dir is not None:
        env["METAINFER_PROFILE"] = "1"
        env["METAINFER_PROFILE_OUTDIR"] = str(profile_dir)
        env.setdefault("METAINFER_PROFILE_DURATION_S", str(profile_duration_s))
        env.setdefault("METAINFER_PROFILE_ACTIVITIES", "CPU,CUDA")
    else:
        # Explicitly disable in case the user had it set in their shell env.
        env["METAINFER_PROFILE"] = "0"
    command = [*(profiler_prefix or []), "bash", str(serve_sh), str(port)]
    return subprocess.Popen(
        command,
        stdout=log_fp, stderr=err_fp,
        cwd=str(serve_sh.parent),
        env=env, start_new_session=True,
    )


def _wait_healthy(
    port: int,
    proc: subprocess.Popen,
    *,
    timeout_s: int,
    report_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """Poll /v1/models until healthy, deadline expires, OR serve.sh dies.

    503 (model still loading) is treated as keep-waiting, not fatal.
    """
    deadline = time.time() + timeout_s
    url_models = f"http://127.0.0.1:{port}/v1/models"
    url_chat = f"http://127.0.0.1:{port}/v1/chat/completions"
    last_err: Optional[str] = None
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            tail = ""
            if report_dir is not None:
                tail = _server_log_tail(report_dir, max_chars=800)
            return False, (
                f"serve.sh exited (rc={rc}) before becoming healthy. "
                + (f"stderr tail:\n{tail}" if tail else "")
            )
        try:
            req = urllib.request.Request(url_models, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return True, None
        except urllib.error.HTTPError as e:
            if e.code in (404, 405, 401):
                if _probe_chat(url_chat):
                    return True, None
            last_err = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            last_err = f"URLError: {e.reason!r}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0)
    return False, last_err


def _server_log_tail(report_dir: Path, *, max_chars: int = 1200) -> str:
    """Best-effort tail of perf-server.stderr.log + stdout."""
    chunks: List[str] = []
    for name in ("perf-server.stderr.log", "perf-server.stdout.log"):
        p = report_dir / name
        if not p.exists():
            continue
        try:
            data = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not data.strip():
            continue
        chunks.append(f"--- {name} (tail) ---\n{data[-max_chars:]}")
    out = "\n".join(chunks)
    return out[-(max_chars * 2):] if out else out


def _probe_chat(url: str) -> bool:
    body = json.dumps({
        "model": "probe", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1, "temperature": 0, "stream": False,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 500
    except Exception:  # noqa: BLE001
        return False


def _kill_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


def _scan_profile_dir(profile_dir: Path) -> List[Dict[str, Any]]:
    """Collect ``*.json.gz`` artifacts the framework's profiler wrote.

    Each entry records path, size, and mtime. Non-recursive — the
    contract says the framework writes flat into METAINFER_PROFILE_OUTDIR.
    Files smaller than 1 KiB are flagged as ``suspect_empty=True`` (a
    0-byte or header-only file means the profiler didn't get to flush,
    usually because the framework was SIGKILLed without honoring the
    SIGTERM path).
    """
    out: List[Dict[str, Any]] = []
    if not profile_dir.is_dir():
        return out
    for p in sorted(profile_dir.glob("*.json.gz")):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({
            "path": str(p),
            "size_bytes": st.st_size,
            "mtime": int(st.st_mtime),
            "suspect_empty": st.st_size < 1024,
        })
    return out


def _collect_external_profile_artifacts(
    iter_dir: Path, profile_dir: Path, patterns: List[str],
) -> None:
    """Copy profiler-owned files emitted beside ``serve.sh`` into report dir."""
    copied: set[Path] = set()
    for pattern in patterns:
        for source in iter_dir.glob(pattern):
            if not source.is_file() or source in copied:
                continue
            copied.add(source)
            try:
                shutil.copy2(source, profile_dir / source.name)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# HTTP request + tokenizer
# --------------------------------------------------------------------------- #

def _send_one_request(
    port: int, case: Dict[str, Any],
    counter_fn: Any, tokenizer_source: str,
) -> RequestResult:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    prompt = case["prompt"]
    payload = {
        "model": case.get("model", "default"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(case.get("max_tokens", 128)),
        "temperature": float(case.get("temperature", 0.0)),
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            elapsed = time.time() - t0
            out_text = _extract_text(raw)
            return RequestResult(
                case_id=case["id"],
                elapsed_s=elapsed,
                http_status=resp.status,
                output_text=out_text,
                output_tokens=counter_fn(out_text),
                input_tokens=counter_fn(prompt),
            )
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # noqa: BLE001
            err_body = ""
        return RequestResult(
            case_id=case["id"], elapsed_s=elapsed, http_status=e.code,
            error=f"HTTP {e.code}: {err_body}",
        )
    except Exception as e:  # noqa: BLE001
        return RequestResult(
            case_id=case["id"], elapsed_s=time.time() - t0, error=f"{type(e).__name__}: {e}",
        )


def _extract_text(raw_body: str) -> str:
    try:
        obj = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body[:2000]
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    if isinstance(first.get("text"), str):
        return first["text"]
    # Stream-style: usage field reports completion_tokens directly.
    usage = obj.get("usage") if isinstance(obj, dict) else None
    if isinstance(usage, dict):
        ct = usage.get("completion_tokens")
        if isinstance(ct, int):
            return " " * ct  # placeholder text — counter_fn will give len-based count
    return ""


# --------------------------------------------------------------------------- #
# Per-run logic: warmup + measured window at a given concurrency
# --------------------------------------------------------------------------- #

def _run_one_level(
    port: int, cases: List[Dict[str, Any]],
    concurrency: int, run_idx: int,
    counter_fn: Any, tokenizer_source: str,
) -> PerfRunResult:
    """Run one (concurrency, run_idx) execution. Warmup, then measure."""
    # Warmup phase: serial requests to load weights / warm KV cache /
    # JIT caches. We don't measure these.
    warmup_cases = cases[:WARMUP_REQUESTS] if len(cases) >= WARMUP_REQUESTS else cases
    for c in warmup_cases:
        _send_one_request(port, c, counter_fn, tokenizer_source)

    # Measured phase: dispatch `cases` prompts through a thread pool of
    # size `concurrency`. Each case dispatched once per run.
    measured: List[RequestResult] = []
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(_send_one_request, port, c, counter_fn, tokenizer_source)
                for c in cases]
        for f in concurrent.futures.as_completed(futs):
            measured.append(f.result())
    t_end = time.time()

    return _aggregate_run(measured, concurrency, run_idx, t_end - t_start)


def _aggregate_run(
    results: List[RequestResult], concurrency: int, run_idx: int, wall_s: float,
) -> PerfRunResult:
    """Strict metric definitions:

    - ``tokens_per_sec`` = sum(output_tokens) / wall_s
      Rationale: throughput is dominated by output token generation;
      prompt prefill is one-shot per request. Using output tokens keeps
      this number comparable across different prompt-length mixes.
    - ``ms_per_request`` = mean(per-request wall clock)
      End-to-end HTTP latency including any server-side queueing under
      concurrency.
    - ``p50_ms / p99_ms`` = percentile of per-request wall clock.
      N = num_requests; p99 unstable for N < 100.
    - ``total_tokens`` = sum(input + output). Reported but not used
      as a throughput metric.
    """
    ok = [r for r in results if r.error is None and r.http_status
          and 200 <= r.http_status < 300]
    err_count = len(results) - len(ok)
    if not ok:
        return PerfRunResult(
            concurrency=concurrency, run_idx=run_idx,
            tokens_per_sec=0.0, ms_per_request=0.0,
            p50_ms=0.0, p99_ms=0.0,
            total_output_tokens=0, total_input_tokens=0,
            total_wall_s=wall_s, num_requests=0, errors=err_count,
            raw_latencies_ms=[],
        )
    latencies_ms = sorted(r.elapsed_s * 1000.0 for r in ok)
    total_out = sum(r.output_tokens for r in ok)
    total_in = sum(r.input_tokens for r in ok)
    tps = total_out / wall_s if wall_s > 0 else 0.0
    return PerfRunResult(
        concurrency=concurrency, run_idx=run_idx,
        tokens_per_sec=tps,
        ms_per_request=statistics.mean(latencies_ms),
        p50_ms=_percentile(latencies_ms, 0.50),
        p99_ms=_percentile(latencies_ms, 0.99),
        total_output_tokens=total_out,
        total_input_tokens=total_in,
        total_wall_s=wall_s,
        num_requests=len(ok),
        errors=err_count,
        raw_latencies_ms=latencies_ms,
    )


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# --------------------------------------------------------------------------- #
# Prompt loading
# --------------------------------------------------------------------------- #

def _load_cases(req: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """Load benchmark cases. Returns (cases, source_label).

    Default: package-bundled prompts.yaml.
    Override: ``req['perf_prompts_path']`` or ``req['answers']['perf_prompts_path']``
    points at a user-provided yaml file with the same shape.
    """
    custom = (req.get("perf_prompts_path")
              or (req.get("answers") or {}).get("perf_prompts_path"))
    if custom and Path(custom).exists():
        data = yaml.safe_load(Path(custom).read_text(encoding="utf-8")) or []
        cases = [c for c in data if isinstance(c, dict) and "id" in c and "prompt" in c]
        if cases:
            return cases, f"user_override:{custom}"
    if PROMPTS_FILE.exists():
        data = yaml.safe_load(PROMPTS_FILE.read_text(encoding="utf-8")) or []
        cases = [c for c in data if isinstance(c, dict) and "id" in c and "prompt" in c]
        return cases, f"package_default:{PROMPTS_FILE.name}"
    return [], "(none)"


# --------------------------------------------------------------------------- #
# Main oracle class
# --------------------------------------------------------------------------- #

class PerfOracle:
    """Generic performance oracle for HTTP-served inference frameworks.

    Not a subclass of :class:`metainfer.oracles.base.Oracle` — that
    base is for correctness checks. The perf oracle has a different
    contract (no pass/fail judge, instead structured measurements).
    """

    def __init__(
        self,
        concurrency_ladder: Tuple[int, ...] = DEFAULT_CONCURRENCY_LADDER,
        runs_per_level: int = RUNS_PER_LEVEL,
        warmup_requests: int = WARMUP_REQUESTS,
    ) -> None:
        self.concurrency_ladder = concurrency_ladder
        self.runs_per_level = runs_per_level
        self.warmup_requests = warmup_requests

    def run(
        self,
        *,
        iter_dir: Path,
        req: Dict[str, Any],
        report_dir: Path,
        iteration: Optional[int] = None,
        prev_perf: Optional[Dict[str, float]] = None,
        timeout_s: int = 1800,
    ) -> PerfReport:
        """Run the perf benchmark and write ``perf-report.json``.

        ``report_dir`` is typically the iteration's logs dir. The
        report file is also written to ``iter_dir / "perf-report.json"``
        so the existing pipeline's :func:`_read_perf_report` finds it.
        """
        report_dir.mkdir(parents=True, exist_ok=True)
        serve_sh = iter_dir / "serve.sh"
        report = PerfReport(
            task_id=req.get("task_id", ""),
            iteration=iteration,
            measured_at=time.time(),
            methodology={
                "concurrency_ladder": list(self.concurrency_ladder),
                "runs_per_level": self.runs_per_level,
                "warmup_requests": self.warmup_requests,
                "request_timeout_s": REQUEST_TIMEOUT_S,
                "startup_timeout_s": _resolve_startup_timeout_s(),
            },
        )

        if not serve_sh.exists():
            report.notes = f"no serve.sh at {serve_sh}"
            self._write_report(report, iter_dir, report_dir)
            return report

        try:
            materialize_hardware_binding(req, iter_dir)
            selected_hardware, hardware_profile = require_hardware_profile(req)
            hardware_env = execution_environment(req, iter_dir)
            profiler_prefix = profiler_launch_command(req)
            external_artifact_globs = profiler_artifact_globs(req)
        except HardwareProfileError as exc:
            report.notes = f"hardware profile error: {exc}"
            self._write_report(report, iter_dir, report_dir)
            return report

        report.methodology["hardware_profile"] = {
            "selected": selected_hardware,
            "id": hardware_profile["id"],
            "external_profiler_command": profiler_prefix,
        }

        cases, prompt_source = _load_cases(req)
        if not cases:
            report.notes = "no benchmark prompts available"
            self._write_report(report, iter_dir, report_dir)
            return report

        tokenizer_source, counter_fn = _resolve_token_counter(req)
        report.methodology["prompts_source"] = prompt_source
        report.methodology["prompts_count"] = len(cases)
        report.methodology["tokenizer_source"] = tokenizer_source

        port = _pick_free_port()
        model_dir = req.get("target_model") or (req.get("answers") or {}).get("target_model")
        # Profile output dir — created here, scanned after the sweep.
        # The framework writes ``metainfer-profile-rank{R}-{ts}.json.gz``
        # files into it per the contract.
        profile_dir = report_dir / "profile"
        try:
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_dir_ok = True
        except OSError:
            profile_dir_ok = False
            profile_dir = None  # type: ignore[assignment]
        report.methodology["profile_enabled"] = profile_dir_ok is not False
        # Backstop duration = benchmark timeout + 60s buffer. The real
        # stop+export is triggered by SIGTERM (see _kill_server); this is
        # only for the case where the framework's own timer fires first.
        profile_duration = int(timeout_s) + 60 if timeout_s else 1800

        proc: Optional[subprocess.Popen] = None
        with GpuTelemetry() as tel:
            try:
                proc = _start_server(
                    serve_sh, port, report_dir,
                    model_dir=model_dir,
                    profile_dir=profile_dir if profile_dir_ok else None,
                    profile_duration_s=profile_duration,
                    extra_env=hardware_env,
                    profiler_prefix=profiler_prefix,
                )
                startup_to = _resolve_startup_timeout_s()
                ok, err = _wait_healthy(
                    port, proc, timeout_s=startup_to, report_dir=report_dir,
                )
                if not ok:
                    tail = _server_log_tail(report_dir, max_chars=1200)
                    msg = f"server did not become healthy within {startup_to}s: {err}"
                    if tail:
                        msg += "\n" + tail
                    report.notes = msg
                    self._write_report(report, iter_dir, report_dir)
                    return report

                # Per-level sweep.
                for c in self.concurrency_ladder:
                    level_runs: List[PerfRunResult] = []
                    for run_idx in range(self.runs_per_level):
                        result = _run_one_level(
                            port, cases, c, run_idx,
                            counter_fn, tokenizer_source,
                        )
                        level_runs.append(result)
                    report.results.append(_summarize_level(c, level_runs))
            finally:
                if proc is not None:
                    _kill_server(proc)
            report.hardware = tel.aggregate()

        # Collect any profiling artifacts the framework wrote. Non-fatal
        # if empty: the implementer may not have wired the hook (older
        # iteration or skipped the contract). The F-step perf planner
        # treats presence of artifacts as a bonus signal, not a gate.
        if profile_dir_ok and profile_dir is not None:
            _collect_external_profile_artifacts(
                iter_dir, profile_dir, external_artifact_globs,
            )
            report.profile_artifacts = _scan_profile_dir(profile_dir)

        # Pick the primary headline (saturated concurrency = max in ladder).
        if report.results:
            primary = max(report.results, key=lambda r: r["concurrency"])
            report.primary = {
                "concurrency": primary["concurrency"],
                "tokens_per_sec": primary["tokens_per_sec_median"],
                "tokens_per_sec_stdev": primary["tokens_per_sec_stdev"],
                "ms_per_request_median": primary["ms_per_request_median"],
                "p50_ms_median": primary["p50_ms_median"],
                "p99_ms_median": primary["p99_ms_median"],
            }
            # Backward-compat flat fields.
            report.tokens_per_sec = primary["tokens_per_sec_median"]
            report.ms_per_request = primary["ms_per_request_median"]
            report.p50_ms = primary["p50_ms_median"]
            report.p99_ms = primary["p99_ms_median"]
            report.total_tokens = float(primary["total_output_tokens_sum"])
            report.total_wall_s = primary["wall_s_sum"]
            report.concurrency = primary["concurrency"]
            report.num_requests = int(primary["num_requests_sum"])

        # Target comparison (optional).
        target_tps = req.get("target_tokens_per_sec")
        if target_tps:
            try:
                target_tps = float(target_tps)
                achieved = (report.tokens_per_sec / target_tps * 100.0) if target_tps > 0 else 0.0
                report.target = {
                    "tokens_per_sec": target_tps,
                    "achieved_pct": round(achieved, 2),
                }
            except (TypeError, ValueError):
                pass

        # Regression vs previous iteration.
        if prev_perf and "tokens_per_sec" in prev_perf and report.tokens_per_sec > 0:
            prev_tps = float(prev_perf["tokens_per_sec"])
            if prev_tps > 0:
                delta_pct = (report.tokens_per_sec - prev_tps) / prev_tps * 100.0
                if delta_pct > 5.0:
                    verdict = "improvement"
                elif delta_pct < -5.0:
                    verdict = "regression"
                else:
                    verdict = "noise"
                report.regression = {
                    "vs_iteration": prev_perf.get("iteration"),
                    "prev_tokens_per_sec": prev_tps,
                    "delta_pct": round(delta_pct, 2),
                    "verdict": verdict,
                }

        self._write_report(report, iter_dir, report_dir)
        return report

    def _write_report(
        self, report: PerfReport, iter_dir: Path, report_dir: Path,
    ) -> None:
        data = report.to_dict()
        # Write to iter_dir (where _read_perf_report looks) AND to
        # report_dir (where logs / debug dumps go). Use the underscore
        # spelling to match the existing pipeline convention
        # (`perf_report.json` is what perf_test_prompt historically told
        # the agent to write, and what _read_perf_report reads).
        for d in (iter_dir, report_dir):
            try:
                p = d / "perf_report.json"
                tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(p)
            except OSError:
                pass


def _summarize_level(concurrency: int, runs: List[PerfRunResult]) -> Dict[str, Any]:
    """Aggregate multiple runs at the same concurrency level."""
    tps_vals = [r.tokens_per_sec for r in runs if r.tokens_per_sec > 0]
    ms_vals = [r.ms_per_request for r in runs if r.ms_per_request > 0]
    p50_vals = [r.p50_ms for r in runs if r.p50_ms > 0]
    p99_vals = [r.p99_ms for r in runs if r.p99_ms > 0]
    return {
        "concurrency": concurrency,
        "runs": len(runs),
        "tokens_per_sec_median": round(statistics.median(tps_vals), 2) if tps_vals else 0.0,
        "tokens_per_sec_stdev": round(statistics.stdev(tps_vals), 2) if len(tps_vals) >= 2 else 0.0,
        "tokens_per_sec_min": round(min(tps_vals), 2) if tps_vals else 0.0,
        "tokens_per_sec_max": round(max(tps_vals), 2) if tps_vals else 0.0,
        "ms_per_request_median": round(statistics.median(ms_vals), 2) if ms_vals else 0.0,
        "p50_ms_median": round(statistics.median(p50_vals), 2) if p50_vals else 0.0,
        "p99_ms_median": round(statistics.median(p99_vals), 2) if p99_vals else 0.0,
        "total_output_tokens_sum": sum(r.total_output_tokens for r in runs),
        "total_input_tokens_sum": sum(r.total_input_tokens for r in runs),
        "wall_s_sum": round(sum(r.total_wall_s for r in runs), 2),
        "num_requests_sum": sum(r.num_requests for r in runs),
        "errors_total": sum(r.errors for r in runs),
    }
