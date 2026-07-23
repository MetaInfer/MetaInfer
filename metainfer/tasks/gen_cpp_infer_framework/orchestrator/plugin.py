"""The TaskPlugin descriptor for gen-cpp-infer-framework."""

from metainfer.orchestrator.tasks.base import TaskPlugin


DIAGNOSTIC_GLOBS = (
    "oracle-report.json",
    "oracle-stages.json",
    "server.stdout.log",
    "server.stderr.log",
    "cpp-build.*.log",
    "numeric-test-report.json",
    "numeric-test.*.log",
    "*-test.log",
    "test.log",
    "judge.*",
    "*.prompt.txt",
    "*.status.json",
    "retrospective.md",
)


PLUGIN = TaskPlugin(
    task_type="gen-cpp-infer-framework",
    cli_module="metainfer.tasks.gen_cpp_infer_framework.orchestrator.cli",
    phases_module="metainfer.tasks.gen_cpp_infer_framework.orchestrator.phases",
    diagnostic_globs=DIAGNOSTIC_GLOBS,
)
