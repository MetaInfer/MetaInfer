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


# Subdirectory (relative to iter_dir/.metainfer-logs/) where the previous
# iteration's diagnostic files land via the copy-forward rule in
# metainfer.iteration. Kept here as a constant so prompt text stays in sync
# with the path the orchestrator actually writes to.
PREV_ITER_LOGS_SUBDIR = "prev-iter"


# Hard prohibition on reading / referencing existing inference-framework
# source code. Past iterations were wasting time exploring vLLM / SGLang /
# TensorRT-LLM internals and producing derivative work; this rule keeps the
# implementer honest and the iterations fast.
#
# The ONLY exception is calling an operator (kernel) from such a library —
# e.g. `torch.nn.functional.softmax`, `flash_attn_func`, a CUTLASS op — as
# a leaf primitive. Anything that resembles a scheduler, KV cache manager,
# paged-attention allocator, block manager, request queue, ContinuousBatching
# loop, tokenizer-detokenizer stream, or HTTP server must be written from
# scratch inside the iteration directory.
NO_FRAMEWORK_REFERENCE_RULE = """# CRITICAL: no reading or copying from existing inference frameworks
You are building a framework FROM SCRATCH. The following are STRICTLY FORBIDDEN:

1. **Do not read, browse, grep, cat, or open** the source code of any
   existing inference engine — this includes (but is not limited to)
   vLLM, SGLang, TensorRT-LLM, TGI, ExLlamaV2, llama.cpp, LightLLM,
   DeepSpeed-MII, Aphrodite, OpenLLM, LMCache, or any of their forks.
   This applies whether the code is on disk, in a pip-installed package,
   on GitHub, or anywhere outside the current iteration directory.
2. **Do not import or call** non-operator code from those frameworks.
   No `from vllm import ...`, no `import sglang`, no copying their files
   into the iteration directory. The ONLY allowed dependency surface is
   - torch / numpy / scipy (the numerical primitives)
   - kernel libraries used as **leaf operators** (e.g. `flash_attn`,
     `xformers.ops`, a single CUTLASS / Triton kernel) — called directly
     for a math op, never extended or subclassed
   - HTTP server / JSON primitives for C++ (e.g. system packages such as
     Boost.Beast, cpp-httplib, nlohmann/json, or a small self-contained
     stdlib/POSIX HTTP shim) — used as transport/parsing, not copied from
     an inference framework
3. **Everything non-operator must be your own work**: the scheduler,
   the KV cache manager, the paged-attention allocator / block table,
   the request queue + continuous-batching loop, tokenizer-detokenizer
   streaming, the OpenAI-compatible request/response shaping, the weight
   loader, the dtype / quantization packaging. All of it — written from
   scratch inside the iteration directory.
4. **Stay inside the iteration directory and the notebooks/ knowledge base.**
   Do NOT read files outside the working directory. In particular, do NOT
   enumerate `/usr/lib/python*`, `site-packages/vllm/`, `site-packages/sglang/`,
   `~/.cache/`, or any third-party project tree. The ONLY reference material
   you may consult is:
     - the `notebooks/` directory pointed to below
     - files inside the current iteration directory

If you find yourself wanting to "see how vLLM does it" — STOP. That is
plagiarism. Re-read the relevant entry in `notebooks/` instead; if none
exists, design the component yourself from first principles. A clean,
slower, original implementation is always preferred over a fast copy."""

CPP_BUILD_TEST_MANDATE = """# MANDATORY C++ build smoke test BEFORE booting the server (gen-cpp-infer-framework)
A large fraction of C++ iterations fail before runtime: missing headers,
bad CMake target wiring, ABI/linker errors, unresolved symbols, or a binary
path mismatch between `build.sh` and `serve.sh`. These are fast failures if
you catch them before the full server boot.

Before you try to start the server, gate on this from the iteration directory:

1. Run:
   ```
   bash build.sh
   ```
   `build.sh` must configure and build the Release binary using CMake,
   typically:
   ```
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j$(nproc)
   ```

2. Then run:
   ```
   ./build/metainfer_cpp_server --help
   ```
   or `./build/metainfer_cpp_server --version` if you provide that flag.

3. If either command exits non-zero, you are not done. Fix the compile,
   link, or binary-path issue and repeat until both commands pass.

Only after the C++ build smoke test passes may you run the server-boot
smoke test below."""

SMOKE_TEST_MANDATE = """# MANDATORY self-test before declaring B done (gen-cpp-infer-framework)
Before you finish, you MUST verify your server actually starts and answers
an OpenAI-style request. Do NOT hand off to C with an unbooted server.

Minimum bar — perform ALL of these locally inside the iteration directory:

0. **C++ build smoke test FIRST** — see the CPP_BUILD_TEST_MANDATE section
   above. Do not run any of the steps below until `bash build.sh` and the
   binary help/version check exit 0.
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

Only after the build and server smoke tests pass may you declare B complete. The next
phase (C) is an immutable oracle that does almost exactly this — if you
skip the self-test, C will fail and you will be back here next iteration
reading this same prompt. Save everyone the round-trip."""


# GPU preflight mandate (gen-cpp-infer-framework).
#
# Every iteration's B-phase self-test boots serve.sh, which loads model
# weights into VRAM. If a previous crashed experiment (your own prior
# attempt that got SIGKILLed, or the C oracle's last run that left a
# zombie) stranded allocation in VRAM, the next boot collides with the
# orphan and fails with "out of memory" — which looks identical to a bug
# in YOUR code. Real example: an implementer burned 60 minutes debugging
# a phantom OOM that was actually leftover VRAM from a previous attempt.
#
# The orchestrator runs a preflight before C and E boots, but it CANNOT
# intercept the agent's own bash smoke tests during B. So the agent MUST
# run the same check itself before every `bash serve.sh` / manual engine run.
GPU_PREFLIGHT_MANDATE = """# MANDATORY: free the GPU before every boot (gen-cpp-infer-framework)
Before EVERY local command that loads model weights into VRAM — including
your B-phase smoke tests (`bash serve.sh $PORT`, `bash build.sh` if it
links/runs GPU probes, any manual `./build/metainfer_cpp_server ...` run
for debugging) — you MUST first verify the GPU is clean. If a previous
experiment crashed without freeing VRAM, the leftover allocation will
make your new boot fail with "out of memory" and you will waste the rest
of this iteration chasing a phantom bug.

Required preflight recipe (run it as a separate Bash turn BEFORE the boot):

  ```bash
  # NVIDIA
  if command -v nvidia-smi >/dev/null; then
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        | awk -F', ' '$2+0 >= 128 {print $1}' \
        | while read pid; do
            echo "killing orphan GPU pid=$pid"
            kill -TERM "$pid" 2>/dev/null || true
          done
      sleep 3
  fi

  # AMD ROCm (this host's primary platform)
  if command -v rocm-smi >/dev/null; then
      rocm-smi --showpids 2>/dev/null \
        | grep -oE '\\b[0-9]{4,}\\b' \
        | while read pid; do
            # Only kill processes you can attribute to python / ccb / metainfer
            cmd=$(ps -o comm= -p "$pid" 2>/dev/null || true)
            case "$cmd" in
              python*|ccb*|claude*) echo "killing orphan GPU pid=$pid ($cmd)"; kill -TERM "$pid" 2>/dev/null || true ;;
            esac
          done
      sleep 3
  fi

  # Also scan /proc/*/fd for any process holding /dev/dri/renderD* open
  # (catches orphans the tools miss).
  for f in /proc/[0-9]*/fd/*; do
    tgt=$(readlink "$f" 2>/dev/null || true)
    case "$tgt" in
      /dev/dri/renderD*|/dev/nvidia*)
        pid=$(echo "$f" | awk -F/ '{print $3}')
        cmd=$(ps -o comm= -p "$pid" 2>/dev/null || true)
        case "$cmd" in
          python*|ccb*|claude*) kill -TERM "$pid" 2>/dev/null || true ;;
        esac ;;
    esac
  done
  sleep 2

  # Verify clean
  if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; fi
  if command -v rocm-smi >/dev/null; then rocm-smi --showmemuse; fi
  ```

Rules:
1. **Run this every time before booting**, not just once at the start of
   B. Your own prior manual C++ server run may have crashed and left VRAM
   behind; the next boot will collide with it.
2. **Only kill python / ccb / claude processes.** Never kill X, your
   shell, or unrelated daemons that happen to hold the render node for
   display.
3. **If the preflight shows zero occupants, proceed immediately** — don't
   waste the turn. The point is to catch orphans, not to make you do
   ceremony.
4. **If `kill -TERM` doesn't free the VRAM after 3 seconds, escalate**:
   `kill -9 $pid`. CUDA/ROCm contexts sometimes need a hard kill.
5. **The orchestrator's C and E oracles already run this same check** —
   you don't need to add anything to serve.sh itself. The check is YOUR
   responsibility only during local B-phase debugging."""


# Process-safety mandate (gen-cpp-infer-framework).
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
PROCESS_SAFETY_MANDATE = """# MANDATORY: process & port safety (gen-cpp-infer-framework)
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
  - a process whose command line contains the iteration dir path
    (`ps -ef | grep $ITER_DIR`) — i.e. clearly your own
  - GPU orphans caught by the GPU_PREFLIGHT_MANDATE recipe (which
    restricts kills to `python*|ccb*|claude*` AND only after a model
    load — that's the ONLY pre-authorized blanket kill)

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

`kill -9` skips signal handlers, profiler flush, and ROCm/CUDA context
teardown. It strands VRAM (you'll OOM on the
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


# Profiling interface mandate (gen-cpp-infer-framework).
#
# The E-step perf oracle will, on every iteration where the implementation
# boots cleanly, set METAINFER_PROFILE=1 and an output dir, then re-run the
# benchmark. The framework MUST honor these env vars so we get a per-iter
# Chrome tracing artifact for the F-step perf planner to read. Without it,
# the perf planner is blind — it can only guess at bottlenecks from
# aggregate throughput numbers.
#
# The generic env-var contract is specified in
# notebooks/00_contracts/profiling_contracts.md. The C++ implementation can
# export native profiler traces, lightweight JSON events, or backend-specific
# artifacts; it must not pretend to use Python-only torch.profiler APIs.
PROFILING_INTERFACE_MANDATE = """# MANDATORY: reserve a profiling interface (gen-cpp-infer-framework)
Your framework MUST honor a small set of `METAINFER_PROFILE*` env vars so
the E-step perf oracle can capture a Chrome tracing artifact per iteration.
Without this hook, the F-step perf planner has no kernel-level visibility
and can only guess at bottlenecks.

Read the contract before writing the hook:
  - `notebooks/00_contracts/profiling_contracts.md`

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
   - Start a native C++ profiling hook immediately. It may be a lightweight
     Chrome-trace JSON writer, a backend profiler wrapper, or a no-op-safe
     adapter that records request/batch timing until a backend profiler is
     available.
   - Output either a `.json.gz` Chrome trace or a structured `.json` event
     file named `metainfer-profile-rank{LOCAL_RANK}-{unix_ts}.json[.gz]`.

3. **Lifecycle tied to server.** On SIGTERM / atexit, ALWAYS call
   `profiler.stop()` / `profiler.flush()` before the process exits. An unflushed trace file is
   zero bytes and useless to F. NEVER use `os._exit()` from a profiled
   server — it skips the export.

4. **One startup log line.** Print exactly one line at server boot:
   `[metainfer-profile] enabled=1 outdir=/path activities=CPU,CUDA duration_s=60`
   or `[metainfer-profile] enabled=0` so the perf oracle can confirm by
   grepping server stdout.

5. **Multi-rank safe.** When TP > 1, each rank writes its own trace file
   with its `LOCAL_RANK` in the name. Never hardcode a single filename.

Keep the profiler hook in ONE module (for example `src/profiler.cpp` plus
`include/profiler.h`). Do NOT scatter profiler calls across the codebase -
one call site around the batch/forward path is enough.

The smoke tests you already run (build + boot) MUST cover the profile
hook: temporarily `METAINFER_PROFILE=1 METAINFER_PROFILE_DURATION_S=2`
when booting the server for the smoke test, then verify a non-empty
`.json` or `.json.gz` appeared in the output dir. If the file is 0 bytes or missing,
your hook is broken — fix it before declaring B done."""


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
    The server's stderr capture — C++ exceptions, assertion failures, OOM
    messages, CUDA/HIP errors, "address already in use", etc. **For a
    crashed/hung server this is almost always where the root cause lives.**
  - {p_stdout}
    Server stdout — startup banner, model load progress, listening-port
    messages. Useful for confirming whether the server even started.
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

    The point is to surface the actual crash/build clue INLINE in the next
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
    _skip = {"task_type", "task_id", "raw_request", "answers"}
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
) -> str:
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    return f"""You are the **PLANNER** for MetaInfer iteration #{iteration}.

Your job: produce a concrete, file-level work plan for this iteration ONLY.
Subsequent agents in this iteration (implementer) will follow your plan
exactly, so be specific about file paths, function signatures, and test
commands.

# Task requirements (frozen)
{_render_req(req)}

# Working directory for this iteration
{iter_dir}
All code and artifacts you plan for must live INSIDE this directory.

# Knowledge base
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}

{NO_FRAMEWORK_REFERENCE_RULE}

# Previous iteration failures (if any)
{prev_failures or "(none — this is the first iteration)"}
{_prev_logs_section(prev_failures, prev_snap)}
{_perf_plan_section(perf_plan)}
{_review_feedback_section(review_feedback)}

# Deliverables
Write exactly two files inside `{iter_dir}`:

1. `plan.md` — your work plan. Must contain:
   - **Goal of this iteration** (1-2 sentences)
   - **File-by-file work items** (path → what to create/modify, key APIs)
   - **Test plan** (what `test.sh` should check; what "correct" means)
   - **Performance targets** (only if this iteration includes perf work)
   - **Risks** (anything that might block the implementer)
   - If `review_feedback` was provided above: a **"Review response"** section
     that says how the plan addresses each suggestion.
   - If a perf plan brief was provided above: a **"Perf plan response"**
     section mapping each planned optimization to a concrete file/function
     in this iteration's work items.

2. `test_spec.md` — a short spec for the test script the implementer will
   write. Describe inputs, expected outputs, and the shape of the JSON the
   test must print to stdout on success:
   `{{"passed": true, "perf": {{"tokens_per_sec": 123.4, ...}}, "notes": "..."}}`

Do NOT write code. Do NOT run tests. Planning only. Be terse — a good
plan fits in one screen. Do not over-explore the knowledge base; read
only the 3-4 notebook files most relevant to this iteration's task.
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

# Plan to follow
Read `{iter_dir}/plan.md` and `{iter_dir}/test_spec.md` (written by the
planner). Implement exactly what they specify — do not invent new scope.

# Knowledge base
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}

{NO_FRAMEWORK_REFERENCE_RULE}

{_iteration_mode_section(iteration, iter_dir)}

# Previous failure (if retrying after a failed test)
{prev_failure or "(none — fresh implementation)"}
If a previous failure is shown, your FIRST commit must address it.
{_inline_traceback_section(prev_snap)}{_prev_logs_section(prev_failure, prev_snap)}
{_perf_plan_section(perf_plan)}
{_review_feedback_section(review_feedback)}

# Deliverables
1. The code described in the plan, inside `{iter_dir}`.
{_deliverables_for_task(task_type, iter_dir, req)}

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

# Discipline
- Same minimal-diff rule: use Edit, not Write. Touch only the file(s)
  the failure points at.
- Run the C++ build + server-boot smoke tests from SMOKE_TEST_MANDATE
  before exiting — your last turn's exit without a passing smoke check
  is exactly why we're back here.
- Be terse. Stop as soon as smoke checks pass.

{PROCESS_SAFETY_MANDATE}

# Knowledge base
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}
"""


def _deliverables_for_task(task_type: str, iter_dir: Path, req: Dict[str, Any]) -> str:
    """Task-type-specific deliverable contract.

    For ``gen-cpp-infer-framework``, the correctness check is owned by an
    immutable oracle that boots your ``serve.sh`` and sends OpenAI-style
    HTTP requests. The implementation behind ``serve.sh`` must be C++.
    For other task types, the agent writes ``test.sh`` itself.
    """
    if task_type == "gen-cpp-infer-framework":
        target_model = req.get("target_model") or (req.get("answers") or {}).get("target_model") or "<path from requirements>"
        return f"""2. A C++ inference framework tree inside `{iter_dir}`. Required files:
   - `{iter_dir}/CMakeLists.txt`
   - `{iter_dir}/build.sh` (bash, executable)
   - `{iter_dir}/serve.sh` (bash, executable)
   - `{iter_dir}/include/` for public headers
   - `{iter_dir}/src/main.cpp`
   - `{iter_dir}/src/http_server.cpp`
   - `{iter_dir}/src/openai_api.cpp`
   - `{iter_dir}/src/engine.cpp`
   - `{iter_dir}/src/model_loader.cpp`
   - `{iter_dir}/src/tokenizer.cpp`
   - `{iter_dir}/src/scheduler.cpp`
   - `{iter_dir}/src/kv_cache.cpp`
   - `{iter_dir}/src/sampler.cpp`

3. `build.sh` must configure and build a Release C++ binary:
   - Use CMake.
   - Produce `build/metainfer_cpp_server`.
   - Exit non-zero on any compile/link failure.
   - Do not download dependencies from the network. Prefer C++17/20,
     POSIX sockets, and system-installed libraries. If an optional library
     is unavailable, choose a local fallback or fail with a clear CMake
     error that names the missing dependency.

4. A serving script at `{iter_dir}/serve.sh` (bash, executable) that:
   - Takes the port as `$1` (e.g. `./serve.sh 8080`).
   - Builds if needed by invoking `bash build.sh`.
   - Starts `./build/metainfer_cpp_server` on that port.
   - Must expose an **OpenAI-compatible API**:
     * `POST /v1/chat/completions`  (required)
     * `POST /v1/completions`      (recommended)
     * `GET  /v1/models`           (recommended; used for health-check)
   - Must BLOCK in the foreground (no daemonize, no `&`) — the orchestrator's
     oracle owns process lifecycle and will SIGTERM/SIGKILL when done.
   - Must honor deterministic decoding (temperature=0, fixed seed) where
     the framework supports it; the oracle sends temperature=0 requests.
   - Request/response bodies must follow the OpenAI schema. At minimum the
     chat completion response must contain
     `choices[0].message.content` as a string.
   - The HTTP serving process must be the C++ binary. Do not implement the
     server in Python with a C++ helper hidden underneath. Small Python
     snippets are allowed only for local smoke-test polling.

   ## Model path resolution (CRITICAL — read carefully)
   The real model weights live at: `{target_model}`
   The orchestrator's oracle launches your serve.sh as `bash serve.sh <port>`
   and sets the env var `MODEL_DIR={target_model}`. Your serve.sh and C++
   binary MUST resolve the model directory in this order:
     1. Positional argument `$2` (if provided)
     2. Env var `$MODEL_DIR`
     3. **A hardcoded default baked in from the requirements above** —
        i.e. literally `DEFAULT_MODEL_DIR="{target_model}"` in serve.sh
        and/or a C++ default used when neither `$2` nor `$MODEL_DIR` is set.

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

   You do NOT write the test, the prompts, or the judge. Only `serve.sh`
   plus the C++ framework and its build files.

{GPU_PREFLIGHT_MANDATE}

{PROFILING_INTERFACE_MANDATE}

{CPP_BUILD_TEST_MANDATE}

{SMOKE_TEST_MANDATE}
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
    if req.get("task_type") == "gen-cpp-infer-framework":
        return "CMakeLists.txt, build.sh, serve.sh, include/, and src/"
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
cases, correctness risks) and on what would speed the next iteration up.

{logs_section}

# Working directory (code only — logs are NOT here, see above)
{iter_dir}

# Plan and spec
Read `{iter_dir}/plan.md` and `{iter_dir}/test_spec.md` for context.

# Knowledge base
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}

{NO_FRAMEWORK_REFERENCE_RULE}

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
{NOTEBOOKS_HINT}
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
    exhaust and D will route back to B for a redo.

    Key constraints (baked into the prompt below):
    * Edit existing files only — do not regenerate the tree.
    * One root cause per attempt. Fix one thing, then stop. The orchestrator
      re-runs the test; if it still fails, the next attempt tackles the next
      cause.
    * The standard implementer smoke-test mandate still applies — verify the
      C++ build + server-boot after editing, so the re-run isn't wasted on a
      compile or link error.
    """
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    remaining = max(0, max_attempts - attempt)
    return f"""You are the **C-STEP DEBUGGER** for MetaInfer iteration #{iteration},
repair attempt {attempt} of {max_attempts} ({remaining} attempt(s) remaining
after this one before the iteration gives up and routes back to B for a full redo).

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
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}

{NO_FRAMEWORK_REFERENCE_RULE}

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
4. **Verify locally before exiting** (gen-cpp-infer-framework): run
   `bash build.sh` from the iteration directory first, then run
   `./build/metainfer_cpp_server --help` or `--version`. If either fails,
   fix the compile/link/binary-path error before doing anything else -
   the orchestrator's C re-run will fail instantly and waste this repair
   attempt. After the build check passes, do a quick local server-boot
   smoke (start `serve.sh` on a free port, poll `/v1/models`, kill it).
   See `CPP_BUILD_TEST_MANDATE` / `SMOKE_TEST_MANDATE` principles from
   the implementer's contract - same rules apply here.
5. **Do NOT re-run the full oracle / test.sh yourself.** That's the
   orchestrator's job. Stop after the local smoke check passes.

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
<bullet list of concrete edits, each as `path/to/file.cpp:LINE - <what
changed>`. Only the files you actually touched. Examples:
- `src/engine.cpp:142` - guard against empty `layers` before indexing
- `src/sampler.cpp:88` - cast temperature to float before comparison>

## Verification
<one paragraph: what local check(s) you ran (C++ build smoke, server-boot
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
3. Smoke-check: `bash build.sh`, binary help/version, then a quick server-boot probe.
4. **MANDATORY**: overwrite `{logs_dir}/c-repair-attempt{attempt}.md`
   with the same 5-section structure as before (Error reason / Root cause
   hypothesis / Fix applied / Verification / Expected next-step outcome).

Be terse. Stop as soon as the smoke check passes and the .md is written.
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
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}

{NO_FRAMEWORK_REFERENCE_RULE}

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
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}
Look especially for kernel tuning notes, fused kernels, and memory layout tips
relevant to the target hardware.

{NO_FRAMEWORK_REFERENCE_RULE}

# Last measured performance (from E)
{json.dumps(last_perf) if last_perf else "(no perf data — read perf_report.json directly)"}

# Files to read first
- `{iter_dir}/perf_report.json`        — the numbers E just measured
- `{iter_dir}/review.md`               — D's review (if any) — may already
  flag perf cliffs
{_prev_logs_section(None, prev_snap)}
{_review_feedback_section(review_feedback)}

# Deliverable
Write `{iter_dir}/perf_plan.md`. Structure:

1. **Current perf baseline** (1-2 lines: cite the key numbers from perf_report.json).
2. **Bottleneck analysis** — what's the likely limit? (kernel bound, memory
   bandwidth, scheduler overhead, C++ allocation/locking overhead, KV cache layout, etc.)
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
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}

{NO_FRAMEWORK_REFERENCE_RULE}

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
"tokens_per_sec improved because `{iter_dir}/src/X.cpp:Y` switched from Z to W,
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
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}

{NO_FRAMEWORK_REFERENCE_RULE}

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
read. NOT "be more careful" — instead "the OpenAI response formatter in
src/openai_api.cpp rejects empty outputs; the implementer should handle the
no-candidates case before serializing JSON". Bullet list, 2-5 items.>

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
