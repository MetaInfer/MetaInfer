"""Prompt templates for the orchestrator's sub-agents.

Each builder returns a fully-rendered prompt string. Prompts deliberately
include:

* the task's frozen requirements
* a pointer to the ``notebooks/`` knowledge base
* the iteration's working directory
* the previous iteration's failure context (when applicable)
* the previous iteration's post-test reviewer feedback (when applicable)
* a hard prohibition on reading or referencing existing inference-framework
  source code (vLLM / SGLang / TensorRT-LLM / ...) — only operator calls
  are allowed; everything else (scheduler, memory, KV cache, server) must
  be written from scratch. Only ``notebooks/`` and the current iteration
  directory may be read.

The orchestrator writes the returned string to a temp file and feeds it to
``claude -p`` via stdin (see :mod:`metainfer.subagent_manager`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .plan_gate import (
    CORE_CAPABILITIES,
    FULL_GATE,
    INCREMENTAL_GATE,
    PlanGateContext,
    frozen_requirements,
)
from .capabilities import normalize_features
from .task_types import is_cpp_framework_task, is_http_framework_task


NOTEBOOKS_HINT = """A knowledge base of reference designs, known pitfalls, and worked
examples lives in the `notebooks/` directory.

Read it EFFICIENTLY:
- Use the Read tool directly. Do NOT spawn sub-agents (no Agent / Task /
  TaskOutput / Explore tool calls) — each sub-agent cold-starts a separate
  `claude -p` process and burns 1-3 minutes of startup time per call,
  which dominates your budget without producing better answers.
- Start with `Glob notebooks/**/*.md` to see the layout, then Read only
  the files whose names match this iteration's task (typically 3-6 files).
  Do NOT read every notebook "to be complete".
- Do not re-read a file you have already read in this session — its
  contents are already in your context window.
- Hard cap: at most ~8 Read calls to notebooks for planning, ~4 for
  implement / review / perf-plan."""


_CPP_PHASE_NOTEBOOKS = {
    "plan": [
        "00_overview/README.md",
        "00_contracts/hardware_profile_contracts.md",
        "00_contracts/framework_contracts.md",
        "00_contracts/engine_contracts.md",
        "00_contracts/qwen3_model_contracts.md",
        "01_framework_design/01_architecture.md",
    ],
    "implement": [
        "00_contracts/testing_contracts.md",
        "00_contracts/weight_loading_contracts.md",
        "00_contracts/operator_contracts.md",
        "00_contracts/memory_contracts.md",
        "01_framework_design/04_model_runner.md",
        "01_framework_design/10_cmake_dtk_toolchain.md",
    ],
    "repair": [
        "00_contracts/testing_contracts.md",
        "00_contracts/engine_contracts.md",
        "00_contracts/qwen3_model_contracts.md",
        "01_framework_design/07_request_lifecycle.md",
    ],
    "review": [
        "00_contracts/testing_contracts.md",
        "00_contracts/framework_contracts.md",
        "00_contracts/engine_contracts.md",
        "00_contracts/memory_contracts.md",
        "00_contracts/operator_contracts.md",
        "00_contracts/native_service_contracts.md",
    ],
    "perf": [
        "00_contracts/profiling_contracts.md",
        "06_profiling/00_overview.md",
        "06_profiling/01_native_trace.md",
        "06_profiling/03_gpu_event_benchmark.md",
        "03_operators/05_hip_blas_backend.md",
        "07_improvementPlan/04_hip_operator_optimization.md",
        "07_improvementPlan/08_profiling_and_benchmarking.md",
    ],
    "retrospective": [
        "00_overview/README.md",
        "00_contracts/testing_contracts.md",
        "06_experience/README.md",
        "07_improvementPlan/README.md",
        "08_issues/README.md",
    ],
}


def _notebooks_hint(req: Dict[str, Any], phase: str) -> str:
    """Return deterministic C++ routing while preserving Python behavior."""
    if not is_cpp_framework_task(req.get("task_type", "")):
        return NOTEBOOKS_HINT

    required = list(_CPP_PHASE_NOTEBOOKS.get(phase, _CPP_PHASE_NOTEBOOKS["implement"]))
    components: List[str] = []
    model_family = str(req.get("model_family") or "").strip().lower()
    if phase in ("plan", "implement", "repair", "review"):
        components.extend([
            "02_model_specifics/02_qwen3/README.md",
            "02_model_specifics/02_qwen3/04_model_loader.md",
        ])
        if "hybrid" in model_family or "3.5" in model_family or "3.6" in model_family:
            components.append(
                "02_model_specifics/02_qwen3/03_qwen3_5_hybrid.md"
            )
        elif "moe" in model_family:
            components.append("02_model_specifics/02_qwen3/02_moe.md")
        else:
            components.append("02_model_specifics/02_qwen3/01_dense.md")
    if phase == "plan":
        components.extend([
            "07_improvementPlan/00_sources_and_governance.md",
            "07_improvementPlan/01_sampling_and_generation.md",
            "07_improvementPlan/09_model_compatibility_and_loader.md",
        ])
    if phase in ("implement", "repair"):
        components.extend([
            "01_framework_design/08_tensor_ownership.md",
            "01_framework_design/05_sampler.md",
            "03_operators/04_sampling_ops.md",
            "05_inference_service/02_openai_http_api.md",
        ])
    features = set(normalize_features(req.get("features")))
    if "paged kv cache" in features:
        components.extend([
            "00_contracts/attention_kv_contracts.md",
            "01_framework_design/03_kv_cache.md",
            "03_operators/01_attention_ops.md",
        ])
        if phase in ("plan", "perf"):
            components.append(
                "07_improvementPlan/03_paged_kv_and_prefix_cache.md"
            )
    if "continuous batching" in features:
        components.extend([
            "01_framework_design/07_request_lifecycle.md",
            "01_framework_design/02_scheduler.md",
        ])
        if phase in ("plan", "perf"):
            components.append("07_improvementPlan/02_continuous_batching.md")
    try:
        tp_size = int(req.get("tensor_parallel_size") or 1)
    except (TypeError, ValueError):
        tp_size = 1
    if "tensor parallelism" in features or tp_size > 1:
        components.extend([
            "00_contracts/tp_communication_contracts.md",
            "04_parallel_strategies/01_tensor_parallel.md",
            "04_parallel_strategies/02_qwen_dense_tp.md",
            "04_parallel_strategies/04_rccl_collectives.md",
        ])
        if "moe" in model_family:
            components.append("04_parallel_strategies/03_qwen_moe_tp.md")
        if phase in ("plan", "perf"):
            components.append("07_improvementPlan/05_tensor_parallel_rccl.md")
    if "streaming responses" in features:
        components.extend([
            "00_contracts/native_service_contracts.md",
            "05_inference_service/02_openai_http_api.md",
            "05_inference_service/03_sse_streaming.md",
            "05_inference_service/04_process_lifecycle.md",
        ])
        if phase == "plan":
            components.append("07_improvementPlan/07_native_service_streaming.md")
    quantization_policy = str(req.get("quantization_format") or "").strip().lower()
    dtype_policy = " ".join((
        str(req.get("weight_dtype") or ""),
        str(req.get("kv_cache_dtype") or ""),
        quantization_policy,
    )).lower()
    if (
        phase in ("plan", "perf")
        and (
            quantization_policy not in ("", "none / full precision")
            or any(token in dtype_policy for token in (
                "int4", "int8", "fp8", "quant", "awq", "gptq", "gguf",
            ))
        )
    ):
        components.append(
            "07_improvementPlan/06_quantization_and_weight_formats.md"
        )
    backend = str(req.get("accelerator_backend") or "").lower()
    hardware = str(req.get("target_hardware") or "").lower()
    if any(token in f"{backend} {hardware}" for token in ("hip", "dtk", "hygon", "amd")):
        components.extend([
            "03_operators/05_hip_blas_backend.md",
            "01_framework_design/09_hardware_discovery.md",
            "01_framework_design/10_cmake_dtk_toolchain.md",
            "06_profiling/02_roctx_rocprof.md",
        ])
        if phase == "plan":
            components.append(
                "07_improvementPlan/04_hip_operator_optimization.md"
            )

    # Preserve order while preventing phase + feature duplicates.
    required = list(dict.fromkeys(required))
    required_set = set(required)
    components = [
        path for path in dict.fromkeys(components)
        if path not in required_set
    ]
    required_rendered = "\n".join(f"- `{path}`" for path in required)
    component_rendered = (
        "\n".join(f"- `{path}`" for path in components)
        if components else "- (none for this phase)"
    )
    return f"""# C++ knowledge-base route (phase: {phase})
Do NOT glob or read the whole knowledge base.

Read this small REQUIRED phase bundle in order:

{required_rendered}

Then read `plan.md` (or the concrete failure/review evidence). From the
following COMPONENT guides, read only the 2-5 files directly involved in this
iteration. Phase A may read every component guide needed to produce a complete
architecture plan; B/C must not consume context on untouched components:

{component_rendered}

This C++ knowledge base is self-contained. Do not inspect sibling task
notebooks to fill a gap, and do not substitute Python when a C++ guide is
incomplete. More reading is not progress: extract exact interfaces,
invariants, ownership rules, and tests, then implement and run a focused
proof."""


# Subdirectory (relative to iter_dir/.metainfer-logs/) where the previous
# iteration's diagnostic files land via the copy-forward rule in
# metainfer.iteration. Kept here as a constant so prompt text stays in sync
# with the path the orchestrator actually writes to.
PREV_ITER_LOGS_SUBDIR = "prev-iter"


# C++-specific source and dependency boundary.  This deliberately does not
# reuse the sibling Python card's rule: allowing Torch/FastAPI in a native
# task contradicts the language contract, while forbidding every path outside
# the iteration prevents the implementer from reading the checkpoint and SDK
# that the user explicitly placed in scope.
CPP_REFERENCE_AND_ACCESS_RULE = """# CRITICAL: native C++ source and access boundary
You are building a native C++ inference framework from scratch.

## Allowed read-only inputs

- the exact checkpoint directory from the frozen requirements, including
  config, tokenizer metadata, weight indexes, and weight bytes;
- the routed `notebooks/` files listed in this prompt;
- the detected hardware profile and diagnostic logs explicitly named by the
  orchestrator;
- system compiler, CMake, SDK headers, shared-library metadata, and bounded
  tool output needed to compile/link/probe the selected backend;
- files already present inside the current iteration directory.

These permissions are for inspection and bounded probes only. Do not modify
the checkpoint, system SDK, hardware profile, notebooks, or prior logs. Write
all generated source, build output, tests, and reports inside the current
iteration directory.

## Forbidden framework references

1. Do not read, browse, grep, open, copy, import, link, or invoke scheduler,
   KV-cache, model-engine, serving, tokenizer, or runtime source from vLLM,
   SGLang, TensorRT-LLM, TGI, ExLlamaV2, llama.cpp, LightLLM, DeepSpeed-MII,
   Aphrodite, OpenLLM, LMCache, or forks of those projects. This applies to
   local checkouts, installed packages, caches, and online sources.
2. Do not embed CPython or use Torch, NumPy, SciPy, Triton, FastAPI, Uvicorn,
   or `http.server` in the production server or inference path. Short-lived
   offline test-data/diagnostic tools remain subject to the separate language
   contract and `LANGUAGE_BOUNDARY.md` disclosure.
3. Do not copy an existing framework's files and do not wrap/proxy an existing
   model server. Scheduler, KV allocator, request lifecycle, model runner,
   sampler, weight loader, and HTTP serving loop are this deliverable's work.

## Allowed native dependency surface

- the C++ standard library, OS/POSIX primitives, and CMake;
- the selected platform runtime and leaf math/collective libraries that pass
  local compile/link/runtime probes (for example HIP/DTK, a vendor BLAS, or a
  compatible collective library);
- focused C/C++ parsing or transport libraries only when already available,
  license-compatible, recorded in the build manifest, and not an inference
  framework.

If a required platform capability is unavailable, report Unsupported with the
probe evidence. Never replace it with Python, CPU model execution, a mock
response, or an unrequested backend. Use the self-contained notebook design
instead of consulting an existing inference engine implementation."""


def _reference_and_access_rule(req: Dict[str, Any]) -> str:
    if is_cpp_framework_task(req.get("task_type", "")):
        return CPP_REFERENCE_AND_ACCESS_RULE
    # Compatibility only: the C++ task package should not define the sibling
    # Python card's dependency policy. Its own prompts remain authoritative.
    return """# Framework source boundary
Do not read or copy an existing inference framework implementation. Use this
task's routed knowledge base and write the requested framework in the current
iteration directory."""

# Self-test mandate for the implementer (gen-infer-framework only).
# Past iterations declared "done" with broken serve.sh scripts that crashed
# on the first request; the immutable oracle then burned a full C step just
# to surface the crash. This rule pushes that feedback loop into B.
IMPORT_SMOKE_TEST_MANDATE = """# MANDATORY import smoke test BEFORE booting the server (gen-infer-framework)
A huge fraction of failed iterations crash at IMPORT time (missing `import
torch`, circular import, syntax error, bad type annotation evaluated at
class-definition time, ...). These are 50-millisecond failures but if you
only catch them via the full server-boot smoke test below, you waste 30-60s
per retry and thrash for 10+ minutes. So before you even try to start the
server, gate on this:

1. From your iteration directory, run:
   ```
   python3 -c "import server"
   ```
   (Adjust the entry module to match whatever `serve.sh` invokes first —
   typically `server`, `engine`, `model`, `kvcache`, `attention`,
   `weight_loader`, `scheduler`, `sampler`, `sequence`, `config`. Importing
   `server` should transitively import all of them.)

2. If that command prints ANYTHING to stderr or exits non-zero, **you are
   not done — do not proceed to the server-boot smoke test**. The traceback
   names the exact file and line. Fix it, re-run the same one-liner, repeat
   until `python3 -c "import server"` exits 0 silently.

3. Only after the import smoke test passes cleanly, run the server-boot
   smoke test below.

This gate exists because the alternative (booting the server and waiting
for a health-check timeout) is 1000× slower and the iteration logs show
real runs burning 13+ minutes in B this way. Do not skip it."""

SMOKE_TEST_MANDATE = """# MANDATORY self-test before declaring B done (gen-infer-framework)
Before you finish, you MUST verify your server actually starts and answers
an OpenAI-style request. Do NOT hand off to C with an unbooted server.

Minimum bar — perform ALL of these locally inside the iteration directory:

0. **Import smoke test FIRST** — see the IMPORT_SMOKE_TEST_MANDATE section
   above. Do not run any of the steps below until `python3 -c "import server"`
   exits 0. Most "server won't boot" failures are actually import-time
   crashes that this one-liner catches instantly.
1. Start your server in the background on a free port:
   `PORT=127.0.0.1:0 ; # or pick any free localhost port`
   `bash serve.sh $PORT &`
2. Poll `GET /v1/models` (or `POST /v1/chat/completions` with a 1-token
   prompt) until it answers 200, or until ~60s have elapsed. Use only
   stdlib (`python -c "import urllib.request ..."`, `curl`) — no requests/
   httpx. Kill the server as soon as it answers.
3. If the server does NOT answer within 60s, or answers with a non-2xx
   status, or returns a response that does not contain
   `choices[0].message.content`, **you are NOT done**. Read
   `server.stderr.log` (redirect the server's stderr to a file when you
   launch it), fix the underlying problem, and re-run the smoke test.
4. Once the smoke test passes, kill the server and delete any background
   PID files you created. Do NOT leave processes running.

Only after the smoke test passes may you declare B complete. The next
phase (C) is an immutable oracle that does almost exactly this — if you
skip the self-test, C will fail and you will be back here next iteration
reading this same prompt. Save everyone the round-trip."""


# GPU preflight mandate. Occupancy is normal on a shared host; ownership must
# be established from an exact PID created by the current task.
GPU_PREFLIGHT_MANDATE = """# MANDATORY: shared-host-safe GPU preflight
Before every command that loads model weights, inspect GPU occupancy with the
available read-only management command (`nvidia-smi`, `rocm-smi`, or the DTK
equivalent) and respect `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, and
`ROCR_VISIBLE_DEVICES`.

This is a multi-user host. Never terminate a process because it uses a GPU,
holds `/dev/kfd` or `/dev/dri/renderD*`, has a familiar executable name, or
occupies a desired port. Other users' workloads are expected. You may stop
only an exact PID that YOU started in this agent turn, after recording that
PID and confirming it still belongs to the same command. Send SIGTERM first;
use SIGKILL only for that same child after a bounded wait.

If an assigned device is busy, report the occupancy or select another device
within the task's explicit allocation. Do not broaden device visibility and
do not add cleanup logic to `serve.sh`. The C and E oracles observe occupancy
but never evict unregistered processes."""


# Process-safety mandate (gen-infer-framework).
#
# Born from a real incident: an implementer agent picked port 8765 (the
# WebUI's port) for a smoke test, found it occupied, ran `lsof -i :8765`,
# saw PID 772908 (the WebUI), and unilaterally executed `kill -9 772908`
# to free the port. The WebUI died; the orchestrator + agents kept
# running because they're start_new_session=True children. The user
# lost the live dashboard mid-experiment. The agent didn't even know
# what process it killed — it just mechanically "port in use → kill".
#
# This mandate closes that hole with three rules: don't pick reserved
# ports, don't kill processes you didn't start, never SIGKILL first.
PROCESS_SAFETY_MANDATE = """# MANDATORY: process & port safety (gen-infer-framework)
You are running inside a long-lived WebUI session that hosts the very
dashboard controlling you. There are ALSO sibling agents (other iters'
c-debuggers, reviewers) and orchestrator subprocesses sharing this host.
A careless `kill` or a poor port choice WILL take down the WebUI or a
neighbor agent. Real incident: an implementer smoked-test on port 8765
(the WebUI's port), found it occupied, and ran `kill -9 772908` on the
WebUI process to free it. The WebUI died mid-iteration.

## Rule 1 — never pick a fixed or reserved port for smoke tests

Reserved ports you MUST NOT use (the host uses them):
  - **8765** — the MetaInfer WebUI (serve.py default)
  - **22, 80, 443, 3000, 4444, 5000, 6000, 7000, 8000, 8080, 8888,
    9000, 9090, 9999** — common services / dashboards
  - any port your `lsof` / `ss` shows as already bound

ALWAYS pick a free ephemeral port programmatically. One of these two:

  ```bash
  # Bash + python one-liner (works everywhere)
  PORT=$(python3 -c "import socket; s=socket.socket(); \\
    s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
  bash serve.sh "$PORT" ...
  ```

  ```bash
  # Pure python script
  python3 -c "
  import socket
  s = socket.socket()
  s.bind(('127.0.0.1', 0))
  print(s.getsockname()[1])
  s.close()
  "
  ```

Never hard-code `PORT=8765`, `PORT=8000`, etc. Never "test on the same
port the WebUI uses". If `serve.sh $PORT` fails with
`bind: address already in use`, **that is a signal to pick a different
port**, NOT a signal to kill whatever holds it.

## Rule 2 — never kill a process you did not personally start

You may ONLY kill:
  - the PID you just captured from `bash serve.sh ... & echo $!`
  - an exact PID from this task's orchestrator-owned PID registry, after
    confirming it still belongs to the current Unix user and task

You may NEVER kill:
  - a process found only by `lsof -i :PORT` or `ss -lntp | grep PORT`
    (you don't know what it is — it could be the WebUI, a sibling
    agent, or the orchestrator)
  - `python` / `python3` by name (e.g. `pkill python`, `killall python`,
    `kill $(ps aux | grep python | awk ...)`) — this kills the WebUI,
    the orchestrator, ccb, and every sibling agent simultaneously
  - `serve.sh` or `python -m server` globally (e.g. `pkill -f serve.sh`,
    `pkill -9 -f "python3 -m server"`) — these match OTHER iters'
    servers too; kill by exact PID instead

If you find an unexpected process on a port you want: **pick a
different port**. Do not kill it. Do not "investigate by killing".

## Rule 3 — SIGKILL (`kill -9`) is forbidden as a first move

Even on a process you legitimately own (your own serve.sh child):
  1. First: `kill -TERM $PID` (graceful)
  2. Wait at least 5 seconds (`sleep 5`)
  3. Check if it exited (`ps -p $PID`)
  4. Only if still alive: `kill -9 $PID`

`kill -9` skips Python `atexit`, FastAPI shutdown, profiler flush,
and ROCm/CUDA context teardown. It strands VRAM (you'll OOM on the
next boot) and produces truncated profile artifacts. The 5-second
TERM path pays for itself on the next iteration.

## Quick reference

  ✓ `PORT=$(python3 -c "...bind(('127.0.0.1',0))..."); bash serve.sh "$PORT"`
  ✓ `kill -TERM $MY_OWN_SERVE_PID; sleep 5; ps -p $MY_OWN_SERVE_PID || kill -9 $MY_OWN_SERVE_PID`
  ✗ `bash serve.sh 8765`  (reserved port)
  ✗ `kill -9 $(lsof -ti:8765)`  (might be the WebUI — never checked)
  ✗ `pkill -f python` / `killall python`  (kills everything)
  ✗ `kill $(ps aux | grep python | awk '{print $2}')`  (kills everything)
  ✗ `kill -9 $PID` as the first attempt  (skips cleanup)
"""


# Profiling interface mandate (gen-infer-framework).
#
# The E-step perf oracle will, on every iteration where the implementation
# boots cleanly, set METAINFER_PROFILE=1 and an output dir, then re-run the
# benchmark. The framework MUST honor these env vars so we get a per-iter
# Chrome tracing artifact for the F-step perf planner to read. Without it,
# the perf planner is blind — it can only guess at bottlenecks from
# aggregate throughput numbers.
#
# The profiling contract is owned by this task package.  Keeping the reference
# local prevents the C++ card from depending on the sibling Python knowledge
# base when prompts are assembled through a compatibility path.
PROFILING_INTERFACE_MANDATE = """# MANDATORY: reserve a profiling interface (gen-infer-framework)
Your framework MUST honor a small set of `METAINFER_PROFILE*` env vars so
the E-step perf oracle can capture a Chrome tracing artifact per iteration.
Without this hook, the F-step perf planner has no kernel-level visibility
and can only guess at bottlenecks.

Read the contract before writing the hook:
  - `metainfer/tasks/gen_infer_framework_cpp/notebooks/00_contracts/profiling_contracts.md`
  - `metainfer/tasks/gen_infer_framework_cpp/notebooks/06_profiling/00_overview.md`

Required behavior (summarized — the contract is authoritative):

1. **Default off.** When `METAINFER_PROFILE` is unset or `0`, do NOT start
   any profiler. Steady-state perf must be unaffected by the mere presence
   of the hook. A 5-15% overhead from a always-on profiler would make the
   E-step throughput numbers incomparable across iterations.

2. **Env-var driven.** When `METAINFER_PROFILE=1`:
   - Read `METAINFER_PROFILE_OUTDIR` (default `.`), create it if missing.
   - Read `METAINFER_PROFILE_DURATION_S` (default `60`). Start a daemon
     timer that calls `profiler.stop()` + export after this many seconds.
   - Read `METAINFER_PROFILE_ACTIVITIES` (default `CPU,CUDA`).
   - Read schedule vars `..._SCHEDULE_WAIT/WARMUP/ACTIVE/REPEAT`
     (defaults `1/1/3/1`).
   - Build the profiler with `torch.profiler.profile(...)`, start it
     immediately, and call `profiler.step()` after every batch forward.
   - Output a `.json.gz` Chrome trace via a custom `on_trace_ready`
     handler named `metainfer-profile-rank{LOCAL_RANK}-{unix_ts}.json.gz`.

3. **Lifecycle tied to server.** On SIGTERM / atexit, ALWAYS call
   `profiler.stop()` before the process exits. An unflushed trace file is
   zero bytes and useless to F. NEVER use `os._exit()` from a profiled
   server — it skips the export.

4. **One startup log line.** Print exactly one line at server boot:
   `[metainfer-profile] enabled=1 outdir=/path activities=CPU,CUDA duration_s=60`
   or `[metainfer-profile] enabled=0` so the perf oracle can confirm by
   grepping server stdout.

5. **Multi-rank safe.** When TP > 1, each rank writes its own trace file
   with its `LOCAL_RANK` in the name. Never hardcode a single filename.

Keep the profiler hook in ONE module (e.g. `profile.py`) and import it
from `server.py`. Do NOT scatter `profiler.step()` calls across the codebase
— one call site in the forward path is enough.

The smoke tests you already run (import + boot) MUST cover the profile
hook: temporarily `METAINFER_PROFILE=1 METAINFER_PROFILE_DURATION_S=2`
when booting the server for the smoke test, then verify a non-empty
`.json.gz` appeared in the output dir. If the file is 0 bytes or missing,
your hook is broken — fix it before declaring B done."""


CPP_GPU_PREFLIGHT_MANDATE = """# MANDATORY: shared-host-safe GPU preflight
Before starting the generated C++ server, inspect `nvidia-smi` or `rocm-smi`
for compute processes on the task's assigned devices. This is a multi-user
host: occupancy, executable name, working directory, an open render node, or
a desired port is NOT ownership proof. Only terminate an exact PID that YOU
started and recorded in this agent turn. Send SIGTERM, wait five seconds, and
use SIGKILL only if that same child remains alive. Never broaden the device
visibility inherited from the orchestrator. The C and E oracles observe GPU
occupancy but never evict unregistered processes."""


CPP_SERVER_SMOKE_TEST_MANDATE = """# MANDATORY: native server smoke test
After the CMake build and focused native tests pass:
1. Choose a free ephemeral localhost port; never reuse the WebUI port 8765.
   Assign it in the parent shell on its own line. Do NOT background an
   `PORT=$(...) && ...` command list: that loses `$PORT` in the polling shell.
2. Start `bash serve.sh <port>` in the background and capture the actual
   native server PID, not a transient wrapper/subshell PID.
3. Poll `/v1/models`, then send one deterministic `/v1/chat/completions`
   request and verify `choices[0].message.content` is a string.
4. Exercise a GET request, a POST whose body arrives in multiple writes, and
   a client that disconnects before reading the response. The server must
   honor `Content-Length`, loop over partial read/write, and survive SIGPIPE.
5. Verify chat messages are rendered with the checkpoint's tokenizer/chat
   template and special-token IDs, never ad-hoc `user:`/`assistant:` text.
6. On failure, inspect the native server stderr and fix the C++ root cause.
7. SIGTERM only the PID you started, wait for cleanup/profile flush, and
   remove temporary PID files. Do not leave a server or GPU context running.

No Python import test applies to this task. Python or curl may be used as a
short-lived HTTP probe, but the server being tested must be the compiled C++
binary."""


CPP_PROFILING_INTERFACE_MANDATE = """# MANDATORY: native C++ profiling interface
Read `00_contracts/profiling_contracts.md` before implementing this
interface; it is the source of truth for signal safety, artifacts, and ranges.

The E-step perf oracle sets `METAINFER_PROFILE=1` and
`METAINFER_PROFILE_OUTDIR=<dir>`. The native runtime MUST honor these
variables without introducing a Python serving process.

Preferred implementation:
1. Add NVTX/ROCTX ranges around request admission, scheduling, prefill,
   decode, KV-cache operations, model forward, sampling, and response write.
2. When profiling is enabled, have `serve.sh` launch the native binary under
   an available platform profiler (`nsys`, `rocprof`, or a vendor tool) and
   write its artifact directly into `METAINFER_PROFILE_OUTDIR`.
3. If no platform profiler exists, emit an internal Chrome-trace JSON or
   `.json.gz` from C++. A missing optional profiler must not prevent serving.
4. Flush profiling data on SIGTERM before exit. Multi-rank runs must include
   the rank in each filename.
5. Print exactly one startup line:
   `[metainfer-profile] enabled=<0|1> backend=<name|none> outdir=<path>`.

Accepted artifact families are `.json`, `.json.gz`, `.nsys-rep`, `.qdrep`,
`.sqlite`, `.csv`, and `.txt`. Do not use `torch.profiler` through a Python
sidecar merely to satisfy this contract; the trace must describe the native
C++ runtime that the oracle is benchmarking."""


def _language_contract(req: Dict[str, Any]) -> str:
    """Return the task's non-negotiable implementation-language boundary."""
    if not is_cpp_framework_task(req.get("task_type", "")):
        return ""
    cpp_standard = req.get("cpp_standard") or "C++17"
    backend = req.get("accelerator_backend") or "the selected native backend"
    model_family = req.get("model_family") or "Auto-detect from config.json"
    weight_dtype = req.get("weight_dtype") or "Auto"
    kv_cache_dtype = req.get("kv_cache_dtype") or "Auto"
    accelerator_clause = ""
    if str(backend).strip().lower() != "cpu":
        accelerator_clause = f"""

# NON-NEGOTIABLE accelerator execution boundary
The selected backend is **{backend}**, so linking an accelerator library or
compiling a device probe is not sufficient. The production server MUST:
- initialize the assigned device before reporting model readiness;
- keep model weights in device memory in the resolved checkpoint dtype;
- execute Q/K/V, attention, output projection, MLP, KV-cache operations, and
  LM-head projection through native accelerator kernels/vendor BLAS;
- retain an accelerator device FD for the healthy server lifetime and show
  non-zero device memory while the model is loaded;
- keep scalar/host reference kernels test-only and out of the serving hot path.

Forbidden serving implementations include full-checkpoint `std::vector<float>`
expansion, nested-loop CPU GEMM, copying device weights back to the host per
request, and a startup manifest that claims a dtype/backend different from the
actual storage/execution path. The immutable oracle checks process device FDs;
the perf oracle additionally checks device memory and activity.
"""
    hardware = json.dumps(
        req.get("hardware_profile") or {"validation": {"status": "not-probed"}},
        indent=2,
        sort_keys=True,
    )
    hardware_path = req.get("hardware_profile_path") or "(not available)"
    return f"""# NON-NEGOTIABLE implementation language: C++ first
This task produces a native inference framework, not a Python framework with
a C++ extension. Use **{cpp_standard}** and **{backend}**.
Model policy: **{model_family}**; weight dtype: **{weight_dtype}**; KV-cache
dtype: **{kv_cache_dtype}**. Every `Auto` value must be resolved from the
checkpoint plus detected backend capabilities and written to the build/runtime
manifest.

The following production runtime components MUST be implemented in C++ (and
CUDA/HIP/native kernel files where appropriate):
- model/config and weight loading
- tokenizer metadata, special-token IDs, and exact checkpoint chat-template rendering
- tensor and device-memory ownership
- scheduler and continuous batching
- logical KV cache and physical memory/block pool
- model runner, prefill/decode loop, operators, and sampler
- request lifecycle and the OpenAI-compatible HTTP server main loop

Required repository shape:
- `CMakeLists.txt`
- `include/` for public/internal C++ headers
- `src/` for C++ runtime and server sources
- `tests/` for native unit/integration tests
- `serve.sh` as a thin build/launch wrapper that ultimately `exec`s the
  compiled native server binary
- `LANGUAGE_BOUNDARY.md` listing every Python file and why it is unavoidable

Python is allowed only for small OFFLINE utilities such as one-time tokenizer
or weight conversion, test-data generation, or diagnostics. It is forbidden
for the long-running HTTP server, scheduling, KV-cache management, model
forward, sampling, per-request subprocesses, or any steady-state hot path.
Do not embed CPython, call a Python model server, or make C++ a proxy around a
Python runtime. If a Python file is added, the same capability must not be
reasonably implementable in C++, and its justification must be recorded in
`LANGUAGE_BOUNDARY.md`.

Every phase must preserve this boundary. A fix that makes tests pass by moving
runtime work into Python is a correctness failure for this task.
{accelerator_clause}

# NON-NEGOTIABLE detected hardware contract
Full read-only evidence: `{hardware_path}`

```json
{hardware}
```

Treat this profile as host evidence, not a suggestion:
- preserve inherited assigned-device visibility
- use `hip_architectures`, not a marketing model, for HIP compiler targets
- preserve PCI and SMI names separately when they disagree
- do not guess missing SDK libraries or capabilities; compile/run a bounded
  probe or report unsupported
- do not use sudo, change device permissions, reset devices, terminate an
  unregistered process, or install system packages
- a validation blocker is an environment failure to report, never permission
  to use a CPU, mock-model, Python, or different-backend fallback"""


def _cpp_build_smoke_mandate(req: Dict[str, Any], iter_dir: Path) -> str:
    build_type = req.get("build_type") or "Release"
    return f"""# MANDATORY: native build and smoke-test gate
Before declaring B complete:
1. Configure out-of-source with
   `cmake -S . -B build -DCMAKE_BUILD_TYPE={build_type}`.
2. Build with `cmake --build build --parallel`. Fix every compiler/linker
   error; warnings that indicate UB, narrowing, lifetime, or ABI issues are
   treated as errors.
3. Run `ctest --test-dir build --output-on-failure` when tests are registered.
4. Verify the actual server executable starts and answers `/v1/models` plus
   one `/v1/chat/completions` request using a free port.
5. Verify `serve.sh` does not start Python and does not download/build
   dependencies on every request. It may configure/build once when the native
   binary is missing, then must `exec` that binary in the foreground.

Keep build artifacts under `{iter_dir}/build/`; never place generated CMake
files beside source files. The immutable C and E oracles invoke `serve.sh`, so
a source tree that only compiles manually under a different command is not a
valid deliverable."""


def _runtime_mandates(req: Dict[str, Any], iter_dir: Path) -> str:
    if is_cpp_framework_task(req.get("task_type", "")):
        return "\n\n".join((
            CPP_GPU_PREFLIGHT_MANDATE,
            CPP_PROFILING_INTERFACE_MANDATE,
            _cpp_build_smoke_mandate(req, iter_dir),
            CPP_SERVER_SMOKE_TEST_MANDATE,
        ))
    return "\n\n".join((
        GPU_PREFLIGHT_MANDATE,
        PROFILING_INTERFACE_MANDATE,
        IMPORT_SMOKE_TEST_MANDATE,
        SMOKE_TEST_MANDATE,
    ))


def _plan_test_spec_contract(req: Dict[str, Any]) -> str:
    task_type = req.get("task_type", "")
    if is_cpp_framework_task(task_type):
        return """Describe the native `serve.sh` contract, CMake/build checks,
OpenAI request/response schemas, health probe, deterministic decoding checks,
and the expected correctness/performance acceptance criteria. The immutable
orchestrator oracle owns the real test cases, so do not plan an agent-authored
`test.sh` as the source of truth."""
    if is_http_framework_task(task_type):
        return """Describe the `serve.sh` contract, import/boot checks, OpenAI
request/response schemas, health probe, deterministic decoding checks, and the
expected correctness/performance acceptance criteria. The immutable
orchestrator oracle owns the real test cases, so do not plan an agent-authored
`test.sh` as the source of truth."""
    return """Describe inputs, expected outputs, and the JSON line that
`test.sh` must print on success: `{"passed": true, "perf": {...}}`."""


def _repair_verification_steps(req: Dict[str, Any]) -> str:
    if is_cpp_framework_task(req.get("task_type", "")):
        return """4. **Verify locally before exiting**: rebuild with
   `cmake --build build --parallel`, run the relevant native test/CTest target,
   then start `serve.sh` on a free port and probe `/v1/models`. Do not replace
   a broken C++ path with Python to get a green probe.
5. **Do NOT re-run the full oracle yourself.** Stop after the native build,
   focused test, and local server smoke check pass."""
    return """4. **Verify locally before exiting**: run
   `python3 -c "import server"` first, then start `serve.sh` on a free port
   and probe `/v1/models`. Fix import/boot errors before exiting.
5. **Do NOT re-run the full oracle yourself.** Stop after the local smoke
   check passes."""


def _review_language_checks(req: Dict[str, Any]) -> str:
    if not is_cpp_framework_task(req.get("task_type", "")):
        return ""
    return """- **C++ boundary audit (CRITICAL):** verify the server process,
  scheduler, KV cache, model runner, sampler, and hot path are C++ sources.
  Inspect every `.py` file and cross-check `LANGUAGE_BOUNDARY.md`. Report
  `CRITICAL: Python runtime substitution` if Python owns any production
  runtime component or C++ merely proxies to Python.
- **Build audit:** verify CMake performs a real Release/RelWithDebInfo native
  build and `serve.sh` foreground-execs the resulting binary."""


def _prev_logs_section(
    prev_failure: Optional[str],
    prev_logs_dir: Optional[Path] = None,
) -> str:
    """Render the 'previous iteration diagnostics' block.

    Only emitted when there IS a prev_failure (i.e. we're retrying after a
    failed C step). When emitted, it tells the agent in no uncertain terms
    to READ the actual log files before writing any code — the prev_failure
    text is a summary, the real root cause is usually in server.stderr.log
    or the per-case judge reasons inside oracle-report.json.

    ``prev_logs_dir`` is the ABSOLUTE path to the prev-iter snapshot
    directory (typically ``<cwd>/.metainfer/logs/<task_id>/<NNN>/prev-iter/``).
    Required since the layout refactor: logs no longer live inside the
    iteration code dir, so a relative ``.metainfer-logs/...`` path would
    not resolve from the agent's CWD.
    """
    if not prev_failure:
        return ""
    if prev_logs_dir is None:
        # No absolute path available — fall back to the legacy relative
        # pointer. Should not happen in the new layout, but keeps the
        # prompt useful if a caller forgets to pass the path.
        snap = f".metainfer-logs/{PREV_ITER_LOGS_SUBDIR}"
        p_oracle = f"{snap}/oracle-report.json"
        p_stderr = f"{snap}/server.stderr.log"
        p_stdout = f"{snap}/server.stdout.log"
        p_judge = f"{snap}/judge.*.log"
        loc_phrase = f"your working directory under `{snap}/`"
    else:
        snap = prev_logs_dir
        p_oracle = snap / "oracle-report.json"
        p_stderr = snap / "server.stderr.log"
        p_stdout = snap / "server.stdout.log"
        p_judge = snap / "judge.*.log"
        loc_phrase = f"`{snap}/`"
    return f"""

# Previous iteration's diagnostic logs (READ BEFORE CODING)
The previous C step failed. Its diagnostic artifacts have been copied into
{loc_phrase}:

  - {p_oracle}
    Full structured verdict: every test case's prompt, the server's actual
    response, the judge's verdict + reason, http status, latency. This is
    the authoritative record of what went wrong.
  - {p_stderr}
    The server's stderr capture — Python tracebacks, OOM messages, CUDA
    errors, "address already in use", etc. **For a crashed/hung server this
    is almost always where the root cause lives.**
  - {p_stdout}
    Server stdout — startup banner, model load progress, "Uvicorn running
    on ...". Useful for confirming whether the server even started.
  - {p_judge}
    The LLM-judge sub-agent's raw output per case (when judge_mode=llm).

The `prev_failure` text above is a condensed summary. BEFORE writing any
code, open these files (especially `server.stderr.log` and the failing
cases' entries in `oracle-report.json`) and identify the concrete root
cause. Quote the relevant lines in your plan/commit message.
"""


def _review_feedback_section(review_feedback: Optional[str]) -> str:
    """Render the 'previous iteration's reviewer suggestions' block.

    Emitted only when we have post-test reviewer feedback from a prior
    iteration. The reviewer's job (see :func:`review_prompt`) is to read
    the code AND the test outcome and propose concrete improvements. Those
    suggestions land here so the next implementer / optimizer can act on
    them — review no longer disappears into the void after C.
    """
    if not review_feedback:
        return ""
    return f"""

# Previous iteration's review (ACT ON IT)
After the previous C step — regardless of pass or fail — a reviewer read
both the code and the test outcome and wrote concrete improvement
suggestions. You are expected to act on them. The reviewer's notes:

{review_feedback}

Address each suggestion explicitly. If you disagree with one, say why in
`plan.md` (planner) or your final commit message (implementer / optimizer)
— do not silently ignore review feedback.
"""


def _read_prev_stderr_tail(prev_snap: Optional[Path], max_bytes: int = 4096) -> Optional[str]:
    """Read the last ~4KB of the previous iteration's server.stderr.log.

    The point is to surface the actual Python traceback INLINE in the next
    iteration's prompt, so the implementer doesn't have to open a file to
    see why the previous C step failed. Returns None when the file is
    missing (e.g. iteration 1, or a non-crash failure that never produced
    stderr). The tail is preferred over the head because tracebacks live
    at the end.
    """
    if prev_snap is None:
        return None
    p = prev_snap / "server.stderr.log"
    try:
        if not p.is_file():
            return None
        size = p.stat().st_size
        with open(p, "rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
            data = f.read()
        text = data.decode("utf-8", errors="replace").strip()
        return text or None
    except Exception:
        return None


def _inline_traceback_section(prev_snap: Optional[Path]) -> str:
    """Render the INLINE previous-stderr-tail block.

    Distinct from `_prev_logs_section`, which points the agent at the log
    files for full context. This block pastes the actual traceback text
    so the agent sees the failure cause without any file I/O — iteration
    logs showed agents burning 10+ minutes re-discovering an import-time
    crash whose traceback was already sitting in server.stderr.log.
    """
    tail = _read_prev_stderr_tail(prev_snap)
    if not tail:
        return ""
    return f"""

# Previous iteration's crash traceback (INLINE — read this FIRST)
This is the tail of the previous iteration's `server.stderr.log`, inlined
here so you don't have to open a file to see why it crashed. The root
cause of the previous failure is almost certainly the LAST traceback in
this block:

```
{tail}
```

Fix the file:line named at the bottom of the LAST traceback before doing
anything else. The full log (with stdout and oracle detail) is still
available at the paths listed further below — open those only if the
inline tail is insufficient.
"""


def _iteration_mode_section(iteration: int, iter_dir: Path) -> str:
    """Render the first-iteration-vs-incremental block.

    Iteration 1 starts in an empty directory → implementer must generate
    every deliverable from scratch as planned.

    Iteration N>1's working directory is **pre-populated by the
    orchestrator** before this prompt is rendered: the previous
    iteration's code tree has been copied in (minus per-iteration cruft —
    `__pycache__/`, stale server logs, PID files, `.metainfer-completed`).
    The implementer should treat the directory as already seeded and make
    INCREMENTAL edits limited to what the previous review flagged — NOT
    regenerate files from scratch (which iteration logs showed burning
    13+ min/iter with no convergence).

    Note: the orchestrator owns the copy-forward; the agent should NOT run
    `cp -r` or any other copy command itself. Doing so wastes a Bash turn
    and re-pollutes the directory with the cruft the orchestrator already
    filtered out.
    """
    if iteration <= 1:
        return """
# Iteration mode: FRESH FULL IMPLEMENTATION
This is iteration #1. The working directory is empty. Generate the full
implementation as specified in `plan.md` and `test_spec.md` — every
deliverable listed below. Do not waste cycles looking for "existing code
to start from"; there is none.
"""
    prev_iter_dir = iter_dir.parent / f"{iteration - 1:03d}"
    exists = prev_iter_dir.is_dir()
    if not exists:
        return f"""
# Iteration mode: FRESH FULL IMPLEMENTATION (no previous code found)
This is iteration #{iteration}, but the previous iteration's code directory
(`{prev_iter_dir}`) does not exist or is empty. Generate the full
implementation as specified in `plan.md` and `test_spec.md`.
"""
    return f"""
# Iteration mode: INCREMENTAL MODIFICATION (DO NOT regenerate working code)
This is iteration #{iteration}. The orchestrator has ALREADY pre-populated
your working directory (`{iter_dir}`) with the previous iteration's code
tree (filtered: no `__pycache__/`, no stale server logs, no PID files, no
`.metainfer-completed`). The code from iteration #{iteration - 1} is now
sitting in `{iter_dir}` as your starting point.

**DO NOT run `cp -r`, `rsync`, or any other copy command.** The copy is
done. Running it yourself wastes a Bash turn and re-introduces the cruft
the orchestrator just cleaned up.

Start editing the existing files directly. Make ONLY the changes indicated
by:
  1. The inline crash traceback above (if any) — fix that file:line first.
  2. The previous iteration's review notes (the "Previous iteration's
     review (ACT ON IT)" section above).
  3. Anything explicitly required by this iteration's `plan.md`.

**DO NOT rewrite files that were already working.** The reviewer's notes
should have named the specific files / lines to change; touch only those.
Regenerating the whole tree from scratch wastes 10+ minutes per iteration
and re-introduces bugs the previous iteration had already fixed. If a file
is not mentioned in the traceback, not mentioned in the review, and not
required by `plan.md`, leave it as-is.
"""


def _perf_plan_section(perf_plan: Optional[str]) -> str:
    """Render the 'previous iteration's perf plan' block.

    Emitted only when we have a perf_plan.md from a prior iteration's F
    step. The plan was written specifically to drive THIS iteration's
    perf work — the planner / implementer should treat it as a brief and
    address each item in the plan.
    """
    if not perf_plan:
        return ""
    return f"""

# Previous iteration's perf plan (THIS IS YOUR BRIEF)
The previous cycle ended with F (perf plan) writing a concrete perf
improvement plan. That plan was written to drive THIS iteration's work —
treat it as your primary brief, alongside the requirements above.

{perf_plan}

Address each "Planned optimizations" item in your `plan.md` (planner) or
your commits (implementer). If you skip an item, explain why in `plan.md`.
"""


def _render_req(req: Dict[str, Any]) -> str:
    """Render the frozen requirements for prompt injection.

    Returns a bulleted list of every structured field the user supplied —
    including ``target_model``, ``target_hardware``, ``features``,
    ``perf_target``, ``perf_budget``, ``max_iterations``, plus anything
    captured under ``answers`` by the interview. Top-level keys are dumped
    directly (NOT just ``answers``) because that's where the interview-
    generated requirements.json puts the model path, hardware, etc.

    Hiding these from the agent has caused multi-iteration death spirals:
    e.g. the agent knew the model name from ``raw_request`` but not the
    on-disk path, so its serve.sh always fell back to a mock dir.
    """
    lines = [f"- task_type: {req.get('task_type', '?')}",
             f"- task_id: {req.get('task_id', '?')}",
             f"- raw_request: {req.get('raw_request', '')}"]
    # Top-level structured fields — the actual contract (model path,
    # hardware, perf target, etc.). Skip the ones already rendered above
    # and any internal bookkeeping keys.
    _skip = {"task_type", "task_id", "raw_request", "answers", "hardware_profile"}
    for k, v in req.items():
        if k in _skip:
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v) if v else "(none)"
        lines.append(f"- {k}: {v}")
    # Backwards-compat: also surface legacy ``answers`` nesting if present.
    answers = req.get("answers") or {}
    for k, v in answers.items():
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v) if v else "(none)"
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _plan_gate_feedback_section(feedback: Optional[str]) -> str:
    if not feedback:
        return ""
    return f"""

# Previous Plan Gate rejection (MUST FIX BEFORE B)
The deterministic gate rejected your previous A-stage artifacts. Correct every
error below in place; do not work around the schema and do not write code.

```json
{feedback}
```
"""


def _plan_manifest_contract(
    req: Dict[str, Any], iteration: int, context: PlanGateContext,
) -> str:
    features = frozen_requirements(req).get("features", [])
    example: Dict[str, Any] = {
        "schema_version": 2,
        "task_type": "gen-infer-framework-cpp",
        "iteration": iteration,
        "gate_mode": context.mode,
        "architecture": {"file": "architecture.md", "complete": True},
        "test_spec": {"file": "test_spec.md", "oracle": "immutable_full_c"},
        "requirements": frozen_requirements(req),
        "requested_features": {
            str(feature): {
                "architecture": "covered",
                "delivery": "now",
                "reason": "REPLACE_WITH_REASON_OR_EMPTY_STRING",
            }
            for feature in features
        },
        "delivery_items": [{
            "id": "REPLACE_WITH_STABLE_ITEM_ID",
            "summary": "REPLACE_WITH_ONE_BOUNDED_IMPLEMENTATION_RESULT",
            "files": ["REPLACE_WITH_RELATIVE_CHANGED_OR_CREATED_FILE"],
            "tests": ["REPLACE_WITH_ASSERTION_BEARING_TEST_OR_COMMAND"],
        }],
        "deferred": [],
    }
    if context.mode == INCREMENTAL_GATE:
        baseline = context.baseline_iteration
        example.update({
            "inherits_verified_iteration": baseline,
            "plan": {
                "file": "plan.md",
                "goal": "incremental_change_ready_for_oracle",
            },
            "iteration_objective": {
                "summary": "REPLACE_WITH_THIS_ITERATION_OBJECTIVE",
                "source": "REPLACE_WITH_requirements_OR_failure_OR_review_OR_perf_plan_OR_maintenance",
                "capabilities": ["REPLACE_WITH_CAPABILITY_CHANGED_THIS_ITERATION"],
            },
            "change_scope": {
                "changed_files": ["REPLACE_WITH_RELATIVE_CHANGED_FILE"],
                "test_files": ["REPLACE_WITH_RELATIVE_TEST_FILE"],
                "affected_core_capabilities": ["target_device_execution"],
                "regression_oracle": "immutable_full_c",
            },
            "core_capabilities": {
                capability: {
                    "status": "inherited_verified",
                    "from_iteration": baseline,
                }
                for capability in CORE_CAPABILITIES
            },
        })
        example["core_capabilities"]["target_device_execution"] = {
            "status": "delivered_after_b",
            "files": ["REPLACE_WITH_RELATIVE_CHANGED_FILE"],
        }
        mode_rules = f"""This is an INCREMENTAL manifest inheriting verified
iteration {baseline}. Do not repeat the previous iteration's full file
inventory. Mark unchanged core capabilities `inherited_verified` with
`from_iteration: {baseline}`. List only genuinely affected core capabilities
in `change_scope.affected_core_capabilities`; mark those `delivered_after_b`
and name only their changed files. `changed_files` and `test_files` must be
non-empty and appear in `plan.md` / `test_spec.md`."""
    else:
        example.update({
            "plan": {
                "file": "plan.md",
                "goal": "minimum_end_to_end_ready_for_oracle",
            },
            "core_capabilities": {
                capability: {
                    "status": "delivered_after_b",
                    "files": ["REPLACE_WITH_RELATIVE_PATH_FROM_PLAN"],
                }
                for capability in CORE_CAPABILITIES
            },
        })
        mode_rules = """This is a FULL manifest. Every core capability must be
`delivered_after_b`, with its non-empty relative file inventory also accounted
for in `plan.md`. Core capabilities may not be inherited or deferred."""
    return f"""`plan_manifest.json` is the deterministic Plan Gate input. Use
this exact schema and replace every `REPLACE_...` value with real relative
paths/reasons. Every file in the current delivery scope must also appear in
`plan.md`.

```json
{json.dumps(example, indent=2, sort_keys=True)}
```

{mode_rules}

`delivered_after_b` means the capability works when B exits. Core capabilities
may never be deferred. Requested advanced features must all be covered by
`architecture.md`; set their delivery to `now` or `deferred` with a reason.
`delivery_items` is the machine-readable A-to-B checklist. Give every planned
implementation result a stable id, concrete files, and an assertion-bearing
test. Prefer each test entry in the form `P01-T01: assertion` (or another
stable, non-prose test id), and reference the same test id on a line or Markdown
table row associated with the delivery item id in `test_spec.md`; the assertion
prose does not need to be duplicated verbatim. Their combined file lists must
cover every `delivered_after_b` core
file and, in incremental mode, every `change_scope.changed_files` and
`change_scope.test_files` entry. B's `implementation_report.json` must account
for the same ids with each id retaining its planned file and test mapping.
Additional non-core deferrals may be listed in `deferred` as objects with
`capability` and `reason`."""


def _plan_gate_mode_section(context: PlanGateContext) -> str:
    if context.mode == INCREMENTAL_GATE:
        return f"""# Plan Gate mode: INCREMENTAL
Iteration {context.baseline_iteration} is a durable, C-verified baseline.
Preserve its complete architecture and verified capabilities. Plan only this
iteration's objective and changed/test files; do not restate every inherited
source file. The immutable full C Oracle remains the regression gate.
Reason: {context.reason}
"""
    return f"""# Plan Gate mode: FULL
No eligible C-verified baseline is available. Re-establish the complete native
architecture and minimum end-to-end delivery before B.
Reason: {context.reason}
"""


# --------------------------------------------------------------------------- #
# A — Plan
# --------------------------------------------------------------------------- #


def plan_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    prev_failures: Optional[str] = None,
    review_feedback: Optional[str] = None,
    perf_plan: Optional[str] = None,
    logs_dir: Optional[Path] = None,
    plan_gate_feedback: Optional[str] = None,
    plan_gate_context: Optional[PlanGateContext] = None,
) -> str:
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    context = plan_gate_context or PlanGateContext(
        FULL_GATE, "no verified baseline context was supplied",
    )
    if context.mode == INCREMENTAL_GATE:
        planning_job = """1. Preserve the COMPLETE architecture blueprint inherited from the verified baseline.
2. Produce a focused incremental delivery plan for this iteration's objective,
   naming only changed files, affected capabilities, and regression tests."""
        plan_requirements = """- **Iteration objective** and its source (requirements, failure, review, perf plan, or maintenance)
   - **Change scope**: changed files, affected core capabilities, and test files
   - Concrete path/function work only for this delta; do not repeat inherited file inventories
   - **Regression verification** using the immutable full C Oracle
   - Risks and rollback/disable conditions for the proposed changes
   - If `review_feedback` was provided: a **Review response** section
   - If a perf-plan brief was provided: a **Perf plan response** section"""
    else:
        planning_job = """1. Maintain a COMPLETE architecture blueprint for the final requested native C++ inference framework.
2. Produce this iteration's minimum end-to-end delivery plan. When B exits,
   the real checkpoint must be ready for the FULL immutable C Oracle."""
        plan_requirements = """- **Goal of this iteration** (1-2 sentences)
   - **File-by-file work items** (path → what to create/modify, key APIs)
   - The real native vertical path: OpenAI request → exact chat
     template/tokenizer → real checkpoint loading → target-device forward →
     deterministic greedy decode plus seeded stochastic sampling → detokenize
     → `choices[0].message.content`
   - **Test plan** matching the immutable full C Oracle
   - **Performance targets** (only if this iteration includes perf work)
   - **Risks** (anything that might block the implementer)
   - If `review_feedback` was provided: a **Review response** section
   - If a perf-plan brief was provided: a **Perf plan response** section"""
    return f"""You are the **PLANNER** for MetaInfer iteration #{iteration}.

Your job has two coupled parts:
{planning_job}

The implementer follows the validated artifacts, so use concrete relative file
paths, native APIs, and commands. Later iterations may repair or optimize
existing code, but their plan must preserve the complete serving path.

{_plan_gate_mode_section(context)}

# Task requirements (frozen)
{_render_req(req)}

# Working directory for this iteration
{iter_dir}
All code and artifacts you plan for must live INSIDE this directory.

# Knowledge base
{_notebooks_hint(req, "plan")}
Knowledge base path: {notebooks_dir}

{_reference_and_access_rule(req)}

{_language_contract(req)}

# Previous iteration failures (if any)
{prev_failures or "(none — this is the first iteration)"}
{_prev_logs_section(prev_failures, prev_snap)}
{_perf_plan_section(perf_plan)}
{_review_feedback_section(review_feedback)}
{_plan_gate_feedback_section(plan_gate_feedback)}

# Deliverables
Write or update exactly these four planning artifacts inside `{iter_dir}`:

1. `architecture.md` — the complete final-system blueprint. It must cover the
   native C++/CMake boundary, model/weight loading, exact tokenizer + chat
   template handling, target-accelerator execution, scheduler, KV cache,
   decode/sampler, OpenAI HTTP service, process lifecycle, every frozen
   requested feature, and component boundaries.

2. `plan.md` — the work plan for the selected Gate mode. Must contain:
   {plan_requirements}

3. `test_spec.md` — the verification contract for this iteration.
   {_plan_test_spec_contract(req)} It must explicitly cover foreground
   `serve.sh`, the real checkpoint on the selected accelerator, deterministic
   temperature=0 `/v1/chat/completions`, seeded `temperature>0` + `top_p`
   sampling whose parameters affect generation,
   `choices[0].message.content`, native process ownership, and cleanup.

4. `plan_manifest.json` — structured proof that the plan is C-ready:
   {_plan_manifest_contract(req, iteration, context)}

Do NOT write code or run tests. Planning only. Be concise but do not omit any
required architecture or end-to-end element. Read only the 3-4 notebook files
most relevant to this iteration's task.
"""


# --------------------------------------------------------------------------- #
# B — Implement
# --------------------------------------------------------------------------- #


def implement_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    prev_failure: Optional[str] = None,
    review_feedback: Optional[str] = None,
    perf_plan: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    task_type = req.get("task_type", "")
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    return f"""You are the **IMPLEMENTER** for MetaInfer iteration #{iteration}.

# Task requirements (frozen)
{_render_req(req)}

# Working directory
{iter_dir}  (everything you write goes here — visible code only)

# Validated planning contract
Read `{iter_dir}/architecture.md`, `{iter_dir}/plan.md`,
`{iter_dir}/test_spec.md`, and `{iter_dir}/plan_manifest.json`. They passed the
deterministic Plan Gate. Implement the planned scope without omitting any core
capability marked `delivered_after_b`. If prose conflicts, frozen requirements
and the manifest's full native end-to-end contract take precedence. Do not add
unrelated scope.

# Knowledge base
{_notebooks_hint(req, "implement")}
Knowledge base path: {notebooks_dir}

{_reference_and_access_rule(req)}

{_language_contract(req)}

{_iteration_mode_section(iteration, iter_dir)}

# Previous failure (if retrying after a failed test)
{prev_failure or "(none — fresh implementation)"}
If a previous failure is shown, your FIRST commit must address it.
{_inline_traceback_section(prev_snap)}{_prev_logs_section(prev_failure, prev_snap)}
{_perf_plan_section(perf_plan)}
{_review_feedback_section(review_feedback)}

# Deliverables
1. The code described by the validated planning contract, inside `{iter_dir}`.
{_deliverables_for_task(task_type, iter_dir, req)}

3. Write `{iter_dir}/implementation_report.json` after all checks finish.
   This report is consumed by a deterministic B-stage gate; prose in your
   final answer is not a substitute. Use this schema:

```json
{json.dumps({
    "schema_version": 1,
    "iteration": iteration,
    "status": "complete",
    "plan_items": [{
        "id": "stable-plan-item-id",
        "status": "implemented",
        "files": ["relative/source.cpp"],
        "tests": ["exact command or test target that passed"],
    }],
    "verification": {
        "build": {"passed": True, "command": "cmake --build build --parallel"},
        "reference_differential": {
            "passed": True,
            "command": "the assertion-bearing differential test command",
            "metrics": {"prefill_decode_cosine": 0.99},
        },
        "end_to_end": {"passed": True, "command": "native E2E command"},
    },
}, indent=2)}
```

Every `delivered_after_b` file and every incremental `change_scope` file from
`plan_manifest.json` must appear in a `plan_items[].files` entry. Do not mark a
planned item implemented unless its file exists and its named test passed.
Use exactly the planned delivery-item ids, and keep every planned file/test
under its corresponding id; moving evidence between ids does not satisfy the
Implementation Gate.
The reference differential test must FAIL non-zero when its cosine or token
agreement threshold is missed; merely printing a metric is not verification.
After your report passes the static checklist, the orchestrator independently
runs `cmake -S . -B build`, `cmake --build build --parallel`, and
`ctest --test-dir build --output-on-failure --no-tests=error`. Your JSON cannot
override those exit codes, and a project with no registered CTest targets is
rejected before C.

{PROCESS_SAFETY_MANDATE}

Keep the implementation minimal and correct. No gold-plating.
"""


def implement_redo_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    prev_failure: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    """Short follow-up used when B is redone in-place (B LOGIC_FAIL → B,
    same iter_dir, same ccb session).

    Runs under ``--resume`` so the implementer already has the prior turn's
    loaded plan / source files / partial diagnosis in context. We only need
    to surface what went wrong and ask for the targeted fix — re-running
    the full bootstrap would re-pay for the bulk of context at full input
    cost.
    """
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    return f"""Your previous B (implement) turn did not produce a working
implementation. You're running in a resumed session, so the plan you read,
the files you wrote, and the diagnosis you had are all still in context —
do NOT re-bootstrap, do NOT re-read files you already read, do NOT rewrite
working code from scratch.

# What went wrong
{prev_failure or "(no failure detail provided — open the diagnostic logs below)"}

If a previous-iteration crash traceback is inlined below, fix that
file:line first.
{_inline_traceback_section(prev_snap)}

# Working directory (your code from last turn is still in place)
{iter_dir}

{_language_contract(req)}

# Discipline
- Same minimal-diff rule: use Edit, not Write. Touch only the file(s)
  the failure points at.
- Run the task-appropriate native build/import and server-boot checks from
  the language contract before exiting. Your last turn's exit without a
  passing smoke check is exactly why we're back here.
- Be terse. Stop as soon as smoke checks pass.

{PROCESS_SAFETY_MANDATE}

# Knowledge base
{_notebooks_hint(req, "implement")}
Knowledge base path: {notebooks_dir}
"""


def _deliverables_for_task(task_type: str, iter_dir: Path, req: Dict[str, Any]) -> str:
    """Task-type-specific deliverable contract.

    For HTTP inference-framework tasks, the correctness check is owned by an
    immutable oracle that boots your ``serve.sh`` and sends OpenAI-style
    HTTP requests. You do NOT write the test — only ``serve.sh``.
    For other task types, the agent writes ``test.sh`` itself.
    """
    if is_http_framework_task(task_type):
        target_model = req.get("target_model") or (req.get("answers") or {}).get("target_model") or "<path from requirements>"
        return f"""2. A serving script at `{iter_dir}/serve.sh` (bash, executable) that:
   - Takes the port as `$1` (e.g. `./serve.sh 8080`).
   - Starts your inference framework's HTTP server on that port.
   - Must expose an **OpenAI-compatible API**:
     * `POST /v1/chat/completions`  (required)
     * `POST /v1/completions`      (recommended)
     * `GET  /v1/models`           (recommended; used for health-check)
   - Must BLOCK in the foreground (no daemonize, no `&`) — the orchestrator's
     oracle owns process lifecycle and will SIGTERM/SIGKILL when done.
   - Must implement both deterministic greedy decoding (`temperature=0`) and
     genuine seeded stochastic sampling (`temperature>0`, `top_p`, `seed`).
     Accepted request fields must affect generation; silently ignoring them
     is a correctness failure.
   - Request/response bodies must follow the OpenAI schema. At minimum the
     chat completion response must contain
     `choices[0].message.content` as a string.

   ## Model path resolution (CRITICAL — read carefully)
   The real model weights live at: `{target_model}`
   The orchestrator's oracle launches your serve.sh as `bash serve.sh <port>`
   and sets the env var `MODEL_DIR={target_model}`. Your serve.sh MUST
   resolve the model directory in this order:
     1. Positional argument `$2` (if provided)
     2. Env var `$MODEL_DIR`
     3. **A hardcoded default baked in from the requirements above** —
        i.e. literally `DEFAULT_MODEL_DIR="{target_model}"` near the top
        of the script, used when neither $2 nor $MODEL_DIR is set.

   ## Mock mode is FORBIDDEN as a silent fallback
   You MAY include a `--mock` flag for local smoke testing without a GPU,
   BUT the server must NEVER silently fall back to mock / echo mode when
   the real model is missing. If the model dir is missing or empty:
     - print a clear error to stderr naming the path that was tried, AND
     - exit non-zero.
   Do NOT auto-enable mock mode based on `ls *.safetensors` failing, do
   NOT auto-degrade to "echo back the prompt" responses, do NOT return
   200 with a `[MetaInfer mock]` prefix. Any of these is a CRITICAL bug:
   the oracle's LLM-judge will mark every case as fail because the
   response is obviously not real model output, and you will burn every
   remaining iteration the same way.

   The orchestrator runs an **immutable oracle** (outside `{iter_dir}`,
   you cannot see or modify it) which:
     1. starts `serve.sh <port>` on a free localhost port,
     2. waits for `/v1/models` (or a probe to `/v1/chat/completions`),
     3. sends a fixed set of prompts via HTTP,
     4. dispatches a separate judge sub-agent to verdict each response,
     5. kills the server and writes `oracle-report.json`.

   You do NOT write the oracle, prompts, or judge. Deliver `serve.sh` plus
   the framework implementation required by this task's language contract.

{_runtime_mandates(req, iter_dir)}
"""
    # default: agent-written test.sh
    return f"""2. A test script at `{iter_dir}/test.sh` (bash, executable) that:
   - Sets up any needed env / weights paths
   - Runs the implementation
   - On success, prints exactly one JSON line to stdout:
     `{{"passed": true, "perf": {{"<metric>": <number>, ...}}, "notes": "..."}}`
   - On failure, prints exactly one JSON line:
     `{{"passed": false, "error": "<short reason>", "traceback": "<last 4KB>"}}`
   - Exit code 0 even on test failure (the orchestrator parses the JSON).
"""


def _review_artifact(req: Dict[str, Any]) -> str:
    """Tell the reviewer which artifact to inspect for this task type."""
    if is_http_framework_task(req.get("task_type", "")):
        return "serve.sh"
    return "test.sh"


# --------------------------------------------------------------------------- #
# B → C — Post-test Review (runs after every non-infra C step)
# --------------------------------------------------------------------------- #


def review_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    outcome: Optional[str] = None,
    failure: Optional[str] = None,
    perf: Optional[Dict[str, float]] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    """Prompt for the **post-test reviewer**.

    The reviewer runs AFTER C (the test / oracle step), not before it. Its
    job is no longer to gate the test — the test has already happened.
    Instead, its job is to:

    1. read the code in the iteration directory,
    2. read the test outcome (`outcome`, `failure`, `perf` below) plus the
       diagnostic logs at ``logs_dir``,
    3. produce concrete, file-level improvement suggestions that the NEXT
       iteration's planner / implementer / optimizer can act on.

    Its output lands in ``ctx.review_feedback`` and is injected into the
    next iteration's prompts. So: be specific, cite file:line, and prefer
    "do X at <path>:<line> because Y" over generic advice.
    """
    outcome_line = (
        f"- C outcome: **{outcome}**"
        if outcome is not None
        else "- C outcome: (not provided — review the code on its own merits)"
    )
    failure_block = (
        f"- Failure reason:\n```\n{failure}\n```"
        if failure
        else "- Failure reason: (none — C passed)"
    )
    perf_block = (
        f"- Measured perf: {json.dumps(perf, sort_keys=True)}"
        if perf
        else "- Measured perf: (none)"
    )
    if logs_dir is not None:
        logs_section = f"""# Diagnostic logs (READ THESE — they're more useful than the code alone)
This iteration's logs (server stdout/stderr, oracle report, judge outputs,
agent prompts) live OUTSIDE the code directory, under:
  {logs_dir}/
Open the relevant files there:
  - {logs_dir}/oracle-report.json     — full per-case verdict
  - {logs_dir}/server.stderr.log      — server crash tracebacks
  - {logs_dir}/server.stdout.log      — server startup banner"""
        review_path = logs_dir / "review.md"
    else:
        logs_section = """# Diagnostic logs (READ THESE — they're more useful than the code alone)
`.metainfer-logs/` in this iteration directory contains the server logs,
oracle report, and judge outputs from the test you're reviewing. Open them:
  - `.metainfer-logs/oracle-report.json`     — full per-case verdict
  - `.metainfer-logs/server.stderr.log`      — server crash tracebacks
  - `.metainfer-logs/server.stdout.log`      — server startup banner"""
        review_path = iter_dir / "review.md"
    return f"""You are the **REVIEWER** for MetaInfer iteration #{iteration}.

You run AFTER the test step. Your verdict does NOT gate anything — the
test has already happened and the iteration is already being closed. Your
job is to produce improvement suggestions that the NEXT iteration will
inherit through `ctx.review_feedback`.

# Task requirements (frozen)
{_render_req(req)}

# This iteration's test outcome
{outcome_line}
{failure_block}
{perf_block}

If C failed, the failure reason + diagnostic logs below tell you what went
wrong. If C passed, focus on what's about to break next (perf cliffs, edge
cases, correctness risks). **Do not claim that performance improved, regressed,
or stayed flat:** E has not run yet. You may identify code-level performance
risks, but measured performance conclusions belong to G after E.

{logs_section}

# Working directory (code only — logs are NOT here, see above)
{iter_dir}

# Plan and spec
Read `{iter_dir}/plan.md` and `{iter_dir}/test_spec.md` for context.

# Knowledge base
{_notebooks_hint(req, "review")}
Knowledge base path: {notebooks_dir}

{_reference_and_access_rule(req)}

{_language_contract(req)}

# Checks (act on what you observed in the logs + code)
- Did the code match the plan? Where did it diverge?
- If C failed: what is the root cause (cite the exact log line or traceback
  frame)? Is it a logic bug, a server-startup bug, a dtype / shape bug, an
  OpenAI-schema mismatch, or an environment issue?
- If C passed: what's the next likely failure? What perf cliff is coming?
  What edge case is unhandled?
- **CRITICAL**: flag any sign that the implementer read or copied from
  vLLM / SGLang / TensorRT-LLM / etc. (imports, suspiciously-idiomatic
  class names, file headers copied from those projects). Report as
  `CRITICAL: plagiarism suspected at <file>:<line>`.
{_review_language_checks(req)}

# Deliverable
Write `{review_path}` with:
- **Verdict**: PASS / NEEDS_FIX  (advisory only — does not gate anything)
- **Root cause** (if C failed): one paragraph citing log lines
- **Issues**: list, each with file:line and a concrete fix the next
  implementer / optimizer can apply directly
- **Suggestions for next iteration**: prioritized list (highest impact
  first). Be specific — "rewrite <file>:<func> to do X because Y".
- **Confidence**: 1-5

Do NOT modify the implementation. Review only.
"""


def perf_review_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    measured_perf: Optional[Dict[str, float]],
    correctness_review: Optional[str],
    logs_dir: Path,
) -> str:
    """Prompt for G, which is the first phase allowed to judge performance."""
    return f"""You are the **MEASURED PERFORMANCE REVIEWER** for MetaInfer
iteration #{iteration}. E has finished. Review the actual immutable benchmark;
do not reuse a previous iteration's number and do not infer a result from code
inspection alone.

# Frozen requirements
{_render_req(req)}

# Working tree
{iter_dir}

# Authoritative measured data
- `{iter_dir}/perf_report.json`
- `{logs_dir}/perf_report.json` (external copy, if present)
- Parsed headline: {json.dumps(measured_perf, sort_keys=True) if measured_perf else "(missing)"}

# Earlier D correctness/code review
{correctness_review or "(none)"}

# Rules
1. Verify every arithmetic statement against `perf_report.json`.
2. Separate concurrency-1 latency gains from aggregate batching gains.
3. Use per-active-device telemetry; never interpret an average across idle
   visible devices as the utilization of the assigned device.
4. Attribute a gain only when measurement or profiling evidence supports it.
   Mark guesses as hypotheses.
5. Report benchmark errors/timeouts and whether comparisons are like-for-like.

# Knowledge base
{_notebooks_hint(req, "perf")}
Knowledge base path: {notebooks_dir}

{_language_contract(req)}

# Deliverable
Write `{logs_dir}/perf-review.md` with:
- verified headline metrics and deltas,
- correctness-preserving changes likely responsible,
- unsupported hypotheses or contradictory earlier claims,
- next profiling evidence needed,
- confidence 1-5.

Do not modify code or rewrite `perf_report.json`.
"""


# --------------------------------------------------------------------------- #
# C — Test runner prompt (only used when no test.sh exists yet)
# --------------------------------------------------------------------------- #


def write_test_harness_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
) -> str:
    return f"""You are the **TEST HARNESS WRITER** for MetaInfer iteration #{iteration}.

The implementer did not provide a usable `test.sh`. Write one at
`{iter_dir}/test.sh` that exercises the implementation in `{iter_dir}`
according to `{iter_dir}/test_spec.md`.

Output contract (MANDATORY — the orchestrator parses this):
- success → exactly one stdout JSON line:
  `{{"passed": true, "perf": {{"<metric>": <number>, ...}}, "notes": "..."}}`
- failure → exactly one stdout JSON line:
  `{{"passed": false, "error": "<short>", "traceback": "<last 4KB>"}}`
- exit 0 even on failure

# Knowledge base
{_notebooks_hint(req, "repair")}
Knowledge base path: {notebooks_dir}
"""


# --------------------------------------------------------------------------- #
# C — In-place repair (dispatched inside _do_test's repair loop)
# --------------------------------------------------------------------------- #


def c_repair_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    attempt: int,
    max_attempts: int,
    failure: Optional[str],
    logs_dir: Optional[Path] = None,
) -> str:
    """Prompt for the **C-step debugger**.

    Dispatched by ``_do_test``'s repair loop when the C step (oracle or
    test.sh) just failed with a LOGIC_FAIL. The debugger reads the failure
    reason, makes a MINIMAL fix in the existing code tree (which the
    orchestrator has already populated via copy-forward), and exits — the
    orchestrator re-runs the test itself. This is NOT a full re-implement
    pass; if the bug needs architectural rework, the repair budget will
    exhaust and D will close the iteration and route to A for a fresh plan.

    Key constraints (baked into the prompt below):
    * Edit existing files only — do not regenerate the tree.
    * One root cause per attempt. Fix one thing, then stop. The orchestrator
      re-runs the test; if it still fails, the next attempt tackles the next
      cause.
    * The standard implementer smoke-test mandate still applies — verify the
      import + server-boot after editing, so the re-run isn't wasted on a
      syntax error.
    """
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    remaining = max(0, max_attempts - attempt)
    return f"""You are the **C-STEP DEBUGGER** for MetaInfer iteration #{iteration},
repair attempt {attempt} of {max_attempts} ({remaining} attempt(s) remaining
after this one before the iteration closes and routes to A for a fresh plan).

The correctness test (C step) just FAILED with this reason:

```
{failure or "(no failure detail provided — read the diagnostic logs below)"}
```

Your job: **identify ONE root cause from the failure reason above, make a
MINIMAL fix to the existing code, verify it locally, and STOP.** The
orchestrator re-runs the test as soon as you exit — you do NOT re-run it
yourself beyond the import + boot smoke checks below.

# Task requirements (frozen)
{_render_req(req)}

# Working directory (pre-populated by orchestrator — do NOT re-copy)
{iter_dir}
The code tree from the previous step is already in place. **DO NOT run
`cp -r`, `rsync`, or any copy command.** Edit files in-place.

# Iteration mode: TARGETED REPAIR (do NOT regenerate)
This is NOT a fresh implementation pass. The previous agent already wrote
working code that just has a bug. Touch only the file(s) the failure reason
points at. Regenerating unrelated files re-introduces bugs that were
already fixed and burns your repair budget on noise.

# Failure-context diagnostic logs
The full oracle-report.json (or test log) for THIS iteration's failing C
step lives in:
  {logs_dir}/
Open the relevant files there for the full per-case verdict + server
stderr/stdout. The inline failure reason above is a summary; the file has
the detail (exact prompt, response, judge verdict, http status, traceback).
{_prev_logs_section(failure, prev_snap)}
{_inline_traceback_section(prev_snap)}

# Knowledge base
{_notebooks_hint(req, "repair")}
Knowledge base path: {notebooks_dir}

{_reference_and_access_rule(req)}

{_language_contract(req)}

# Discipline (MANDATORY — the iteration's survival depends on this)
1. **Diagnose first.** Re-read the failure reason above. Open the oracle
   report or test log if you need the exact failing input. Quote the
   specific symptom in your final message before editing.
2. **Fix ONE root cause per attempt.** If multiple things look wrong,
   pick the one the failure reason actually points at. The next repair
   attempt (if needed) can address the next symptom.
3. **Minimal diff.** Use Edit, not Write. Do not rewrite whole functions
   unless the bug is structural. Do not touch files unrelated to the
   failure.
4. **Escalate architecture gaps.** If the failure cannot be fixed by a
   bounded local edit, do not make an unrelated change just to consume a
   repair attempt. Write `REPLAN_REQUIRED: <reason>` in the repair log's
   Root cause section and stop. The orchestrator will return to A.
{_repair_verification_steps(req)}

{PROCESS_SAFETY_MANDATE}

# Deliverable
A minimal code change in `{iter_dir}` that addresses the root cause named
in the failure reason. No new files unless the fix genuinely requires one.

**MANDATORY — write a structured repair log** to:
  `{logs_dir}/c-repair-attempt{attempt}.md`

The orchestrator collects these logs into a per-iteration
`c-repairs.jsonl` for post-run forensics. Without this file the repair
is treated as "no diagnosis recorded" and the next attempt loses
context. Use EXACTLY this section structure (markdown):

```markdown
# C-step repair attempt {attempt}/{max_attempts}

## Error reason (input)
<one paragraph: what the failing C step reported — quote the specific
error message, traceback frame, or judge verdict that triggered this
repair. Do NOT paste the whole failure block verbatim; summarize the
operative symptom.>

## Root cause hypothesis
<one paragraph: what you believe is the underlying bug. Cite file:line.
If you considered and rejected alternative hypotheses, name them and
say why you picked this one.>

## Fix applied
<bullet list of concrete edits, each as `path/to/file.py:LINE — <what
changed>`. Only the files you actually touched. Examples:
- `engine.py:142` — guard against empty `layers` list before indexing
- `sampler.py:88` — cast temperature to float before comparison>

## Verification
<one paragraph: what local check(s) you ran (import smoke, server-boot
probe, manual request) and what you observed. If a check failed and you
fixed it before exiting, note that too.>

## Expected next-step outcome
<one sentence: "C re-run should now pass because <reason>" OR "C re-run
will likely still fail on <X>; the next attempt should focus on <Y>".>
```

End your turn after writing both the code fix and the repair log.
"""


def c_repair_followup_prompt(
    iteration: int,
    attempt: int,
    max_attempts: int,
    new_failure: Optional[str],
    logs_dir: Path,
) -> str:
    """Short follow-up prompt used for C-step repair turns 2..N.

    This prompt runs under ``ccb --resume``, so the agent already has the
    full prior context: the iteration's code tree (already loaded once),
    the prior failure analysis, and any edits it already made. Re-stating
    the whole bootstrap (knowledge base hints, framework rules, deliverable
    contract) would re-pay for those tokens at full input cost and drown
    the actual signal — the new failure that surfaced after the last fix.

    So this prompt is intentionally short: "your last fix didn't fully
    work, here's the new failure, apply the same discipline as before
    (one root cause, minimal diff, smoke test, write the .md)."
    """
    remaining = max(0, max_attempts - attempt)
    return f"""The C-step re-run after your previous fix still FAILED. This is
repair attempt {attempt} of {max_attempts} ({remaining} remaining after this).

**New failure from the re-run:**

```
{new_failure or "(no failure detail provided — open the diagnostic logs below)"}
```

You're running in a resumed session — everything you diagnosed, read, and
edited last turn is still in your context. Do NOT re-bootstrap; do NOT
re-read files you already read. The diagnostic files for this iteration
live at `{logs_dir}/` (open them only if the new failure needs detail you
don't already have).

Same discipline as before:
1. Identify the ONE root cause the new failure points at.
2. Make a minimal Edit (no rewrites of unrelated code).
3. Smoke-check with the native CMake target and a short `serve.sh` boot probe.
4. **MANDATORY**: overwrite `{logs_dir}/c-repair-attempt{attempt}.md`
   with the same 5-section structure as before (Error reason / Root cause
   hypothesis / Fix applied / Verification / Expected next-step outcome).

Be terse. Stop as soon as the smoke check passes and the .md is written.
If the new failure proves the plan or architecture is incomplete, write
`REPLAN_REQUIRED: <reason>` in the Root cause section and stop without an
unrelated code edit.
"""


# --------------------------------------------------------------------------- #
# E — Perf Test (only reached when C passed)
# --------------------------------------------------------------------------- #


def perf_test_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    review_feedback: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    """Prompt for the **perf-test** step (E).

    The implementation in ``iter_dir`` has passed the correctness oracle. Now
    we need an independent, heavier performance measurement than the latency
    numbers the correctness oracle happened to collect. The agent is asked
    to write ``perf.sh`` (a benchmark) and run it, emitting a perf_report.json
    the F step (and the next iteration's planner) can consume.

    Heavy benchmark guidance: more prompts, higher concurrency, warmup,
    longer sequences — see the prompt body for details.
    """
    return f"""You are the **PERF TESTER** for MetaInfer iteration #{iteration}.

The implementation in `{iter_dir}` has PASSED correctness (C step). Your job
is to measure its performance under a heavier, more realistic load than the
correctness oracle used.

# Task requirements (frozen)
{_render_req(req)}

# Working directory (code only — logs are NOT here)
{iter_dir}

# Knowledge base
{_notebooks_hint(req, "perf")}
Knowledge base path: {notebooks_dir}

{_reference_and_access_rule(req)}

{_review_feedback_section(review_feedback)}

# Deliverables

1. Write `{iter_dir}/perf.sh` (bash, executable). It MUST:
   - Boot `serve.sh` on a free localhost port (you can reuse the same
     startup logic the correctness oracle used). Pass the model path
     through: `MODEL_DIR="<target_model from requirements>" bash serve.sh $PORT`
     (the correctness oracle does this for you; perf.sh must do it itself).
   - Wait until `/v1/models` answers 200 (poll with stdlib only).
   - Run a HEAVIER benchmark than the correctness pass:
     * at least 16 distinct prompts (mix short + long output)
     * measure under concurrent requests (e.g. a thread/process pool of
       4-16 workers via `concurrent.futures`)
     * include a warmup phase (drop the first N requests before measuring)
     * tokenize output with the model's own tokenizer OR approximate with
       a len-of-text heuristic if the tokenizer isn't easily reachable
   - Kill the server cleanly and emit perf_report.json next to the script.

2. Run `perf.sh` yourself (you have shell access). Capture its output.

3. Write `{iter_dir}/perf_report.json` (or have perf.sh write it directly)
   with EXACTLY this shape (all numbers are floats):

   ```json
   {{
     "tokens_per_sec": <throughput across all concurrent reqs>,
     "ms_per_request": <mean end-to-end latency>,
     "p50_ms": <p50 latency>,
     "p99_ms": <p99 latency>,
     "total_tokens": <sum across requests>,
     "total_wall_s": <benchmark wall time>,
     "concurrency": <number of concurrent workers>,
     "num_requests": <number of measured requests>,
     "notes": "<short string: warmup policy, tokenizer source, etc.>"
   }}
   ```

   The orchestrator will parse this and feed it to the next agent (F) and
   to the next iteration's planner. Output that doesn't parse → INFRA_FAIL.

4. ALSO print exactly one JSON line to stdout (same shape) so the orchestrator
   can capture perf without re-parsing the file:

   `{{"passed": true, "perf": {{...}}, "notes": "..."}}`

   On failure: `{{"passed": false, "error": "<short>", "traceback": "<last 4KB>"}}`

Do NOT modify the implementation code — only write perf.sh and the report.
If the implementation is too broken to benchmark, emit the failure JSON.

{PROCESS_SAFETY_MANDATE}
"""


# --------------------------------------------------------------------------- #
# F — Perf Plan (only reached when E passed)
# --------------------------------------------------------------------------- #


def perf_plan_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    last_perf: Optional[Dict[str, float]] = None,
    review_feedback: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    """Prompt for the **perf-plan** step (F).

    Reads the perf_report.json produced by E (+ review.md from D, if any),
    then writes perf_plan.md describing concrete perf improvements the NEXT
    iteration's planner / implementer should execute. Does NOT change code.
    """
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    return f"""You are the **PERF PLANNER** for MetaInfer iteration #{iteration}.

E measured the current implementation's perf. Your job: read those numbers
(+ the D reviewer's notes if any) and write a concrete perf improvement
plan that the NEXT iteration's A (planner) and B (implementer) will execute.

**You do NOT modify code.** Planning only. The plan must be specific enough
that the next implementer can act on each item without re-analysis.

# Task requirements (frozen)
{_render_req(req)}

# Working directory (code only)
{iter_dir}

# Knowledge base
{_notebooks_hint(req, "perf")}
Knowledge base path: {notebooks_dir}
Look especially for kernel tuning notes, fused kernels, and memory layout tips
relevant to the target hardware.

{_reference_and_access_rule(req)}

{_language_contract(req)}

# Last measured performance (from E)
{json.dumps(last_perf) if last_perf else "(no perf data — read perf_report.json directly)"}

# Files to read first
- `{iter_dir}/perf_report.json`        — the numbers E just measured
- `{logs_dir}/review.md`               — D's correctness/code review
- `{logs_dir}/perf-review.md`          — G's review of E's measured results
{_prev_logs_section(None, prev_snap)}
{_review_feedback_section(review_feedback)}

# Deliverable
Write `{iter_dir}/perf_plan.md`. Structure:

1. **Current perf baseline** (1-2 lines: cite the key numbers from perf_report.json).
2. **Bottleneck analysis** — what's the likely limit? (kernel bound, memory
   bandwidth, scheduler overhead, Python overhead, KV cache layout, etc.)
   Cite a profiler / py-spy result if you can get one.
3. **Planned optimizations** — a prioritized list. For each item:
   - One-line summary (what to change)
   - Expected impact (e.g. "+15% tokens/sec")
   - Confidence (1-5)
   - Specific files / functions / kernels to touch
   - Risk to correctness (and how to mitigate)
4. **What NOT to do** — anti-suggestions (things that look tempting but
   won't help, or would break correctness).

Keep it terse. The next iteration's planner will read this verbatim.
"""


# --------------------------------------------------------------------------- #
# Deprecated: optimize_prompt
# --------------------------------------------------------------------------- #
#
# The D_optimize phase was split into E_perf_test + F_perf_plan in the 6-phase
# restructure. The function is kept here as a thin alias so any external
# callers don't break, but it is no longer used by the orchestrator.

def optimize_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    last_perf: Optional[Dict[str, float]] = None,
    prev_failure: Optional[str] = None,
    review_feedback: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    """Deprecated alias — the orchestrator no longer calls this. Kept only
    so external imports don't break. Prefer :func:`perf_plan_prompt`."""
    return perf_plan_prompt(
        req=req, iter_dir=iter_dir, notebooks_dir=notebooks_dir,
        iteration=iteration, last_perf=last_perf, review_feedback=review_feedback,
        logs_dir=logs_dir,
    )


# --------------------------------------------------------------------------- #
# Post-E retrospective (human-readable iteration summary)
# --------------------------------------------------------------------------- #
#
# Runs once at the end of E_perf_test. Produces a single Markdown file that
# the WebUI renders when the user clicks an iteration row. The point is to
# answer three questions at a glance:
#   1. What did this iteration actually change?
#   2. How did perf move vs the previous iteration (numbers)?
#   3. Why did perf move that way? (root-cause analysis from the code +
#      plan + review)
# Plus a boiled-down summary of the D-phase review.md so the reader doesn't
# need to open a second file.
#
# This is NOT a gating step — failure to produce the file is logged but
# does not change the transition out of E. It's purely an observability
# artifact for post-hoc browsing.


def _perf_table_block(this_perf: Optional[Dict[str, float]],
                      prev_perf: Optional[Dict[str, float]]) -> str:
    """Render this-iter vs prev-iter perf as a markdown table with a delta
    column. Returns "(no data yet)" lines when perf is missing so the
    retro agent still has something concrete to react to."""
    if not this_perf and not prev_perf:
        return "(no perf data for this iteration or the previous one)"
    keys: List[str] = []
    for k in (this_perf or {}):
        if k not in keys:
            keys.append(k)
    for k in (prev_perf or {}):
        if k not in keys:
            keys.append(k)
    lines = [
        "| metric | previous | this iteration | Δ |",
        "| --- | --- | --- | --- |",
    ]
    for k in keys:
        prev_v = prev_perf.get(k) if prev_perf else None
        this_v = this_perf.get(k) if this_perf else None
        prev_s = f"{prev_v:.4g}" if isinstance(prev_v, (int, float)) else "—"
        this_s = f"{this_v:.4g}" if isinstance(this_v, (int, float)) else "—"
        if isinstance(prev_v, (int, float)) and isinstance(this_v, (int, float)) and prev_v != 0:
            # Show signed percentage; guard divide-by-zero.
            pct = (this_v - prev_v) / abs(prev_v) * 100.0
            delta_s = f"{pct:+.1f}%"
        else:
            delta_s = "—"
        lines.append(f"| `{k}` | {prev_s} | {this_s} | {delta_s} |")
    return "\n".join(lines)


def retrospective_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    this_perf: Optional[Dict[str, float]] = None,
    prev_perf: Optional[Dict[str, float]] = None,
    review_feedback: Optional[str] = None,
    e_ok: bool = True,
    e_error: Optional[str] = None,
    logs_dir: Optional[Path] = None,
    goal: Optional[str] = None,
) -> str:
    """Prompt for the **post-E retrospective** writer.

    The agent reads the iteration's plan.md + code + perf_report.json + the
    D-phase review.md, then writes a single Markdown file
    (``{logs_dir}/retrospective.md``) that explains — in plain prose — what
    this iteration did and why perf moved vs the previous iteration. The
    WebUI renders this file when the user clicks the iteration row.

    This step does NOT modify code, does NOT gate the loop, and does NOT
    feed into the next iteration's prompts. It is pure observability.
    """
    perf_table = _perf_table_block(this_perf, prev_perf)
    review_section = (
        f"# D-phase review (already produced this iteration)\n"
        f"The reviewer wrote `{logs_dir}/review.md`. Read it, then SUMMARIZE\n"
        f"it below — do NOT copy it verbatim. The retrospective is meant to\n"
        f"be the single doc a reader opens; the raw review.md stays as the\n"
        f"long-form backup.\n"
        if logs_dir is not None
        else "# D-phase review\n(read `.metainfer-logs/review.md` if present)\n"
    )
    e_status_line = (
        "E (perf test) PASSED — perf numbers below are real measurements."
        if e_ok
        else f"E (perf test) FAILED — no fresh perf data. Error: `{e_error or 'unknown'}`"
    )
    goal_line = goal.strip() if goal and goal.strip() else "(no goal recorded for this iteration)"
    return f"""You are the **RETROSPECTIVE WRITER** for MetaInfer iteration #{iteration}.

The perf-test step (E) just finished. Your ONE job is to produce a single
Markdown file — `{logs_dir}/retrospective.md` — that lets a reader understand
this iteration at a glance: what changed, how perf moved vs the previous
iteration, and why.

You do NOT modify code. You do NOT gate anything. You only WRITE that one
Markdown file. Be concrete and specific — cite file paths, function names,
and metric numbers. Avoid hand-wavy claims like "the scheduler got faster";
say WHICH function and BY HOW MUCH.

# Task requirements (frozen context)
{_render_req(req)}

# Iteration goal (from planner)
{goal_line}

# E step status
{e_status_line}

# Perf comparison vs previous iteration
{perf_table}

# Working directory (the code this iteration produced)
{iter_dir}

Read for context (do NOT dump them in the output, synthesize):
  - `{iter_dir}/plan.md`           — what the planner set out to do
  - `{iter_dir}/perf_report.json`  — this iteration's raw perf numbers
  - `{iter_dir}/perf_plan.md`      — (if present) the prior cycle's perf plan
  - the code under `{iter_dir}/`    — diff mentally vs what a typical
                                      baseline would look like

{review_section}

# Knowledge base (only if you need background on a technique)
{_notebooks_hint(req, "retrospective")}
Knowledge base path: {notebooks_dir}

{_reference_and_access_rule(req)}

{_language_contract(req)}

# Deliverable — write `{logs_dir}/retrospective.md` with EXACTLY these sections

```markdown
# Iteration {iteration} retrospective

**Goal:** <one sentence paraphrasing the iteration goal above>
**E status:** <passed / failed — one short clause>
**Headline perf change:** <e.g. "tokens_per_sec +12.4% vs iter {max(iteration - 1, 1)}, driven by X">

## What this iteration changed
<2-4 bullet points naming the concrete code changes — file:function level.
Reference real paths under {iter_dir}. Don't paraphrase the plan verbatim;
describe what was ACTUALLY done, which may differ from the plan.>

## Perf vs previous iteration
<Restate the table above in prose. Call out the biggest mover(s). If a
metric regressed, say so explicitly — do not hide bad news.>

| metric | previous | this | Δ |
| --- | --- | --- | --- |
| ... | ... | ... | ... |

## Why perf moved
<The core analysis. Tie each notable delta to a specific code change.
"tokens_per_sec improved because `{iter_dir}/X.py:Y` switched from Z to W,
which removes N redundant ops per request." If you can't find a cause,
say "unclear — possibly noise" rather than inventing one.>

## Review summary
<3-6 bullet distilled points from review.md. Focus on actionable findings
the reader should remember: latent bugs flagged, perf cliffs predicted,
correctness risks. NOT a copy of the review.>

## Caveats / open questions
<Anything the reader should be skeptical about: noisy measurement, tiny
sample size, perf gained at the cost of correctness, features not yet
exercised by the benchmark. Bullet list, may be empty.>
```

Keep the whole file under ~600 words. The reader is skimming a dashboard,
not reading a paper. If a section has nothing to say, write "(none)" rather
than padding.
"""


def failure_retrospective_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    failure_reason: Optional[str] = None,
    failed_phase: Optional[str] = None,
    phase_attempts: Optional[int] = None,
    logs_dir: Optional[Path] = None,
    goal: Optional[str] = None,
) -> str:
    """Prompt for the **postmortem writer** of a FAILED iteration.

    A failed iteration never reaches E (perf test), so the regular
    retrospective_prompt — which centers on perf deltas — doesn't apply.
    Instead this prompt asks for a focused failure analysis: what was
    attempted, where it broke, what the agent tried to recover, why
    recovery didn't work, and what the next iteration should do
    differently.

    The agent reads the plan + the implementation that was tried + the
    failing phase's diagnostic artifacts (oracle report, debugger attempt
    events, c-repair notes, test stderr, server logs) and writes a single
    Markdown file (``{logs_dir}/retrospective.md``). The WebUI already
    renders this path — no UI changes needed.

    Non-gating: a failure retro that itself fails is logged and swallowed
    (a failure-within-a-failure shouldn't break the orchestrator loop).
    """
    goal_line = goal.strip() if goal and goal.strip() else "(no goal recorded for this iteration)"
    failure_line = (
        failure_reason.strip() if failure_reason and failure_reason.strip()
        else "(no failure_reason on the iteration record)"
    )
    phase_line = failed_phase or "(unknown)"
    attempts_line = (
        f"{phase_attempts} attempt(s) at this phase before the iteration gave up"
        if phase_attempts is not None
        else "(attempt count not available)"
    )
    return f"""You are the **FAILURE POSTMORTEM WRITER** for MetaInfer iteration #{iteration}.

This iteration FAILED. It never reached the perf-test (E) phase, so the
regular "perf moved because X" retrospective does not apply. Your job is to
write the postmortem: WHAT broke, WHERE in the loop, WHY recovery didn't
succeed, and WHAT the next iteration should do differently.

You produce ONE Markdown file: `{logs_dir}/retrospective.md`. The WebUI
renders it when the user clicks the iteration row — make it the one doc a
reader opens to understand why this iteration failed. Cite real file paths,
real function names, real error messages. Do NOT hand-wave "the test had
a bug"; quote the actual stderr line or the actual failing oracle case.

You do NOT modify code. You do NOT gate anything. You only WRITE that file.

# Task requirements (frozen context)
{_render_req(req)}

# Iteration goal (from planner)
{goal_line}

# Headline failure
- **failed phase:** `{phase_line}`
- **failure reason:** {failure_line}
- **{attempts_line}**

# Working directory (the code this iteration produced)
{iter_dir}

# Where to find failure evidence
Read for context (do NOT dump them in the output, synthesize):
  - `{iter_dir}/plan.md`           — what the planner set out to do
  - the code under `{iter_dir}/`    — what was actually implemented
  - `{iter_dir}/test.sh`           — the correctness harness (if present)
  - `{logs_dir}/oracle-report.json`— per-case results when oracle path was used
  - `{logs_dir}/c-repairs.jsonl`   — structured record of each repair attempt
  - `{logs_dir}/c-repair-attempt*.md`— per-attempt debugger notes
  - `{logs_dir}/iter{iteration}-c-debugger.attempt*.events.jsonl`
                                    — raw ccb events from each debugger turn
  - `{logs_dir}/server.stderr.log` — server logs (crashes, OOM, import errors)
  - any `*.log` / `*.stderr` files under `{logs_dir}/`

Many of these files may not exist (depends on which phase failed and
whether the repair loop ran). Read what exists; skip what doesn't.

# Knowledge base (only if you need background on a technique)
{_notebooks_hint(req, "retrospective")}
Knowledge base path: {notebooks_dir}

{_reference_and_access_rule(req)}

{_language_contract(req)}

# Deliverable — write `{logs_dir}/retrospective.md` with EXACTLY these sections

```markdown
# Iteration {iteration} postmortem (FAILED)

**Goal:** <one sentence paraphrasing the iteration goal>
**Failed phase:** `{phase_line}`
**Headline failure:** <one-sentence summary of the failure_reason, in plain language>

## What this iteration tried to do
<2-3 bullets: the concrete plan — what code change / new component was
attempted, citing plan.md and the actual files written under {iter_dir}.>

## Where it broke
<The single most load-bearing failure point. Be specific: "the oracle
returned LOGIC_FAIL on 3 of 10 cases; case 'X' failed because the model's
output didn't match the expected schema" — NOT "tests failed". Quote the
actual error line / stderr / failing case id. If multiple things broke,
lead with the one that aborted the iteration.>

## What recovery was attempted
<If the repair loop ran (c-repairs.jsonl exists), summarize each attempt:
what the debugger hypothesized, what it changed, why it still didn't pass.
If no repair was attempted (e.g. B INFRA_FAIL), say so and explain why.>

## Why recovery didn't work
<The honest analysis. Common patterns:
  - misdiagnosis (debugger fixed the wrong thing)
  - root cause deeper than the surface symptom
  - test harness itself buggy / oracle flake
  - infra failure (OOM, model timeout) that code can't fix
  - planner asked for the impossible
Quote evidence from the diagnostic files. If you genuinely can't tell,
write "unclear from the available logs" rather than inventing a story.>

## What the next iteration should do differently
<Concrete, actionable recommendations the next planner/implementer should
read. NOT "be more careful" — instead "the oracle schema check at line X
of oracle.py rejects empty outputs; the implementer should handle the
no-candidates case before formatting". Bullet list, 2-5 items.>

## Caveats
<Anything the reader should be skeptical about: missing logs (so the
analysis is partial), suspected oracle flakiness, infra-only failure that
might not reproduce. Bullet list, may be empty.>
```

Keep the whole file under ~600 words. The reader is skimming a dashboard,
not reading a paper. If a section has nothing to say, write "(none)" rather
than padding. Lead with the failure; do not bury the lede by walking through
the plan in detail before saying what went wrong.
"""
