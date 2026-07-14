"""The TaskPlugin descriptor for gen-cpp-infer-framework."""

from ..base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="gen-cpp-infer-framework",
    name="C++ Inference Framework Optimizer",
    description=(
        "6-phase iteration loop (plan→implement→test→review→perf→"
        "perf_plan). Generates and optimizes C++ inference serving code "
        "for a target model on target hardware."
    ),
    cli_module="metainfer.orchestrator.tasks.gen_cpp_infer_framework.cli",
    phases_module="metainfer.orchestrator.tasks.gen_cpp_infer_framework.phases",
)
