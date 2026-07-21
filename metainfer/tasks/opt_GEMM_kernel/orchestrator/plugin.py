"""Task descriptor for the independent GEMM kernel arena."""

from metainfer.orchestrator.tasks.base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="opt_GEMM_kernel",
    cli_module="metainfer.tasks.opt_GEMM_kernel.orchestrator.cli",
    phases_module="metainfer.tasks.opt_GEMM_kernel.orchestrator.phases",
    diagnostic_globs=(
        "feedback.json",
        "review.md",
        "*.prompt.txt",
        "*.stdout.log",
        "*.stderr.log",
    ),
)

