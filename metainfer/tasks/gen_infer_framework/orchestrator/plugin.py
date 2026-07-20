"""The TaskPlugin descriptor for gen-infer-framework."""

from metainfer.orchestrator.tasks.base import TaskPlugin


# Diagnostic files copied forward from iteration N-1's logs dir into
# iteration N's logs/prev-iter/ subdir at open time. Without this, the
# next agent has no visibility into WHY the previous step failed —
# the failure_reason text alone is rarely enough to debug a server
# crash or a judge verdict.
#
# Covers BOTH C_test and E_perf_test artifacts. When E fails (perf
# oracle produced no usable data), the planner needs to see the perf
# server logs and the retrospective to understand WHY — otherwise it
# blindly replans and the next iteration hits the same E failure.
DIAGNOSTIC_GLOBS = (
    # C_test (correctness oracle)
    "oracle-report.json",
    "server.stdout.log",
    "server.stderr.log",
    "judge.*",
    # E_perf_test (perf oracle)
    "perf_report.json",
    "perf-server.stdout.log",
    "perf-server.stderr.log",
    # Review analysis — the retrospective often contains better root
    # cause analysis than the raw oracle output.
    "retrospective.md",
    # Agent prompts + generic diagnostics
    "*-test.log",
    "test.log",
    "*.prompt.txt",
)


PLUGIN = TaskPlugin(
    task_type="gen-infer-framework",
    cli_module="metainfer.tasks.gen_infer_framework.orchestrator.cli",
    phases_module="metainfer.tasks.gen_infer_framework.orchestrator.phases",
    diagnostic_globs=DIAGNOSTIC_GLOBS,
)
