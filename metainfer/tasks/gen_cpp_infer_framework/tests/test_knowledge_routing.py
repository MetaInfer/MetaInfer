"""Knowledge routing and prompt-injection contract tests."""

from __future__ import annotations

from pathlib import Path
import re

import yaml

from metainfer.tasks.gen_cpp_infer_framework.orchestrator.knowledge import (
    resolve_knowledge_route,
)
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.failure_routing import (
    classify_failure,
)
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.prompts import (
    c_repair_prompt,
    failure_retrospective_prompt,
    implement_redo_prompt,
    implement_prompt,
    perf_plan_prompt,
    perf_test_prompt,
    plan_prompt,
    retrospective_prompt,
    review_prompt,
    write_test_harness_prompt,
)


TASK_DIR = Path(__file__).parents[1]
NOTEBOOKS_DIR = TASK_DIR / "notebooks"


def _ids(documents):
    return [document.id for document in documents]


def _requirements(**overrides):
    req = {
        "task_type": "gen-cpp-infer-framework",
        "task_id": "routing-test",
        "raw_request": "Build a Qwen3 C++ inference framework",
        "target_model": "/models/qwen3-8b-q8_0.gguf",
        "target_hardware": "Hygon Z200",
        "features": [],
        "perf_target": "Throughput",
    }
    req.update(overrides)
    return req


def test_manifest_registers_every_knowledge_asset():
    manifest = yaml.safe_load(
        (NOTEBOOKS_DIR / "routing.yaml").read_text(encoding="utf-8")
    )
    registered = {
        Path(entry["path"])
        for entry in manifest["documents"].values()
    }
    actual = {
        path.relative_to(NOTEBOOKS_DIR)
        for path in NOTEBOOKS_DIR.rglob("*")
        if path.is_file() and path.name not in {"README.md", "routing.yaml"}
    }
    assert registered == actual


def test_relative_markdown_links_resolve_inside_knowledge_base():
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in NOTEBOOKS_DIR.rglob("*.md"):
        in_fence = False
        for line in document.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for raw_target in link_pattern.findall(line):
                target = raw_target.split("#", 1)[0].strip()
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (document.parent / target).resolve()
                assert resolved.is_file(), (
                    f"broken Markdown link in {document}: {raw_target}"
                )


def test_baseline_routes_are_role_specific_and_paths_exist():
    req = _requirements()
    planner = resolve_knowledge_route(req, NOTEBOOKS_DIR, role="planner")
    implementer = resolve_knowledge_route(req, NOTEBOOKS_DIR, role="implementer")

    assert _ids(planner.required) == [
        "implementation-blueprint",
        "capability-checklists",
        "qwen3-model-contract",
        "openai-http-server",
        "z200-operator-contract",
    ]
    assert "gguf-loader" in _ids(planner.optional)
    assert "z200-operator-contract" in _ids(planner.required)
    assert "gguf-loader" in _ids(implementer.required)
    assert "z200-operator-contract" in _ids(implementer.required)
    assert "z200-kernel-reference" in _ids(implementer.optional)
    assert "z200-numeric-tests" in _ids(implementer.optional)
    assert "tokenizer-source-reference" in _ids(implementer.optional)
    assert "implementation-blueprint" in _ids(implementer.optional)
    assert "implementation-sequence" in _ids(implementer.required)
    assert "capability-checklists" in _ids(implementer.required)
    assert "gguf-loader-template" in _ids(implementer.optional)
    assert "scheduler-block-manager-template" not in _ids(implementer.required)
    assert "framework-wiring-template" in _ids(implementer.optional)
    assert "numeric-harness-template" in _ids(implementer.optional)
    assert "verified-008-tp-paged-continuous" not in _ids(
        implementer.required
    )
    assert implementer.optional_limit == 4
    for document in (*planner.required, *planner.optional, *implementer.required):
        assert (NOTEBOOKS_DIR / document.path).is_file()


def test_feature_routes_are_independent_and_add_only_selected_contracts():
    continuous = resolve_knowledge_route(
        _requirements(features=["Continuous batching"]),
        NOTEBOOKS_DIR,
        role="implementer",
    )
    assert "continuous-batching" in _ids(continuous.required)
    assert "paged-kv-cache" not in _ids(continuous.required)
    assert "continuous-batching" not in _ids(continuous.optional)
    assert "paged-kv-cache" not in _ids(continuous.optional)

    paged = resolve_knowledge_route(
        _requirements(features=["Paged KV cache"]),
        NOTEBOOKS_DIR,
        role="planner",
    )
    assert "paged-kv-cache" in _ids(paged.required)
    assert "continuous-batching" not in _ids(paged.required)
    assert not paged.notes

    tp = resolve_knowledge_route(
        _requirements(
            target_model="/models/qwen3-8b-f16.gguf",
            features=["Tensor parallelism"],
        ),
        NOTEBOOKS_DIR,
        role="implementer",
    )
    assert "tensor-parallel" in _ids(tp.required)
    assert "paged-kv-cache" not in _ids(tp.required)
    assert "continuous-batching" not in _ids(tp.required)
    assert "gguf-loader" in _ids(tp.required)
    assert not tp.notes


def test_failure_context_adds_targeted_debugger_documents():
    route = resolve_knowledge_route(
        _requirements(),
        NOTEBOOKS_DIR,
        role="debugger",
        context="GGUF tensor shape mismatch while loading model metadata",
    )
    assert "gguf-loader" in _ids(route.required)
    assert "qwen3-model-contract" in _ids(route.required)
    assert "z200-numeric-tests" not in _ids(route.required)

    numeric = resolve_knowledge_route(
        _requirements(),
        NOTEBOOKS_DIR,
        role="debugger",
        context="RoPE logits contain NaN after hipBLAS",
    )
    assert "z200-operator-contract" in _ids(numeric.required)
    assert "z200-numeric-tests" in _ids(numeric.required)
    assert "z200-kernel-reference" in _ids(numeric.required)


def test_failure_playbook_forces_exact_debugger_documents_and_prompt(
    tmp_path: Path,
):
    req = _requirements(
        features=["Paged KV cache", "Continuous batching"]
    )
    failure = (
        "C0.1 numeric tests failed: missing required numeric case "
        "kv_capacity_contract"
    )
    classification = classify_failure(failure, req)
    forced = (
        *classification.required_documents,
        *classification.reference_templates,
    )
    route = resolve_knowledge_route(
        req,
        NOTEBOOKS_DIR,
        role="debugger",
        context=failure,
        required_document_ids=forced,
    )
    prompt = c_repair_prompt(
        req,
        tmp_path,
        NOTEBOOKS_DIR,
        iteration=1,
        attempt=1,
        max_attempts=3,
        failure=failure,
        logs_dir=tmp_path,
        failure_route=classification.to_dict(),
    )

    required_ids = _ids(route.required)
    assert "capability-checklists" in required_ids
    assert "scheduler-block-manager-template" in required_ids
    assert str(
        NOTEBOOKS_DIR / "reference" / "scheduler_block_manager_template.hpp"
    ) in prompt
    assert "failure playbook root-cause checks" in prompt
    assert "name-only PASS" in prompt
    assert "evidence required before exit" in prompt


def test_combination_knowledge_is_added_only_for_active_combinations():
    combined = resolve_knowledge_route(
        _requirements(
            target_model="/models/qwen3-8b-f16.gguf",
            features=[
                "Paged KV cache", "Continuous batching", "Tensor parallelism",
            ],
        ),
        NOTEBOOKS_DIR,
        role="implementer",
    )
    paged_only = resolve_knowledge_route(
        _requirements(features=["Paged KV cache"]),
        NOTEBOOKS_DIR,
        role="implementer",
    )
    paged_batching = resolve_knowledge_route(
        _requirements(
            features=["Paged KV cache", "Continuous batching"],
        ),
        NOTEBOOKS_DIR,
        role="implementer",
    )
    tp_batching = resolve_knowledge_route(
        _requirements(
            target_model="/models/qwen3-8b-f16.gguf",
            features=["Tensor parallelism", "Continuous batching"],
        ),
        NOTEBOOKS_DIR,
        role="implementer",
    )

    assert "paged-continuous-state-machine" in _ids(combined.required)
    assert "tp-paged-kv-contract" in _ids(combined.required)
    assert "tp-continuous-batching-contract" in _ids(combined.required)
    assert "verified-008-tp-paged-continuous" in _ids(combined.required)
    assert "paged-continuous-state-machine" not in _ids(paged_only.required)
    assert "tp-paged-kv-contract" not in _ids(paged_only.required)
    assert "verified-008-tp-paged-continuous" not in _ids(
        paged_only.required
    )
    assert "verified-008-tp-paged-continuous" not in _ids(
        paged_batching.required
    )
    assert "verified-008-tp-paged-continuous" not in _ids(
        tp_batching.required
    )
    assert "tp-continuous-batching-contract" in _ids(tp_batching.required)
    assert "tp-paged-kv-contract" not in _ids(tp_batching.required)


def test_required_routes_stay_bounded_after_contract_deduplication():
    cases = [
        _requirements(),
        _requirements(features=["Continuous batching"]),
        _requirements(
            target_model="/models/qwen3-8b-f16.gguf",
            features=["Tensor parallelism"],
        ),
        _requirements(
            target_model="/models/qwen3-8b-f16.gguf",
            features=[
                "Paged KV cache", "Continuous batching", "Tensor parallelism",
            ],
        ),
    ]
    limits = {"planner": 11, "implementer": 16, "reviewer": 10}
    for req in cases:
        for role, limit in limits.items():
            route = resolve_knowledge_route(req, NOTEBOOKS_DIR, role=role)
            assert len(route.required) <= limit, (role, _ids(route.required))


def test_independent_capability_contracts_do_not_require_paged_kv():
    continuous = (
        NOTEBOOKS_DIR / "runtime" / "continuous_batching.md"
    ).read_text(encoding="utf-8")
    tensor_parallel = (
        NOTEBOOKS_DIR / "distributed" / "tensor_parallel.md"
    ).read_text(encoding="utf-8")

    assert "生产路径必须先实现 Paged KV Cache" not in continuous
    assert "Continuous-only 使用每 sequence 独立的 contiguous" in continuous
    assert "TP-only 使用每 Rank 本地的 contiguous" in tensor_parallel


def test_failure_context_cannot_activate_disabled_capability_documents():
    base = resolve_knowledge_route(
        _requirements(features=[]),
        NOTEBOOKS_DIR,
        role="debugger",
        context="TP2 allreduce rank mismatch in paged KV batching scheduler",
    )
    assert "tensor-parallel" not in _ids(base.required)
    assert "paged-kv-cache" not in _ids(base.required)
    assert "continuous-batching" not in _ids(base.required)

    tp = resolve_knowledge_route(
        _requirements(
            target_model="/models/qwen3-8b-f16.gguf",
            features=["Tensor parallelism"],
        ),
        NOTEBOOKS_DIR,
        role="debugger",
        context="TP2 allreduce rank mismatch",
    )
    assert "tensor-parallel" in _ids(tp.required)


def test_debugger_reads_only_capability_documents_selected_by_failure():
    req = _requirements(
        target_model="/models/qwen3-8b-f16.gguf",
        features=[
            "Paged KV cache", "Continuous batching", "Tensor parallelism",
        ],
    )
    memory = resolve_knowledge_route(
        req,
        NOTEBOOKS_DIR,
        role="debugger",
        context="insufficient VRAM after KV cache allocation",
    )
    tp = resolve_knowledge_route(
        req,
        NOTEBOOKS_DIR,
        role="debugger",
        context="TP2 allreduce rank mismatch",
    )

    assert "tensor-parallel" not in _ids(memory.required)
    assert "paged-kv-cache" not in _ids(memory.required)
    assert "continuous-batching" not in _ids(memory.required)
    assert "single-sequence-runtime" in _ids(memory.required)
    assert "tensor-parallel" in _ids(tp.required)
    assert "paged-kv-cache" not in _ids(tp.required)
    assert "continuous-batching" not in _ids(tp.required)


def test_legacy_nested_features_are_routed():
    req = _requirements()
    req.pop("features")
    req["answers"] = {"features": ["Continuous batching"]}
    route = resolve_knowledge_route(req, NOTEBOOKS_DIR, role="reviewer")
    assert "continuous-batching" in _ids(route.required)


def test_embedded_frozen_contract_overrides_mutated_feature_fields():
    req = _requirements(features=[])
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator.capabilities import (
        resolve_capabilities,
    )
    req["resolved_requirements"] = resolve_capabilities(req)
    req["features"] = ["Continuous batching"]

    route = resolve_knowledge_route(req, NOTEBOOKS_DIR, role="implementer")
    assert "continuous-batching" not in _ids(route.required)


def test_prompts_render_required_and_bounded_optional_routes(tmp_path: Path):
    req = _requirements(features=["Continuous batching"])
    planner = plan_prompt(req, tmp_path, NOTEBOOKS_DIR, iteration=1)
    implementer = implement_prompt(req, tmp_path, NOTEBOOKS_DIR, iteration=1)

    for prompt, optional_limit in ((planner, 2), (implementer, 4)):
        assert "# Deterministic knowledge route (MANDATORY)" in prompt
        assert "## Required reading" in prompt
        assert f"## Optional reading (choose at most {optional_limit})" in prompt
        assert str(NOTEBOOKS_DIR / "runtime" / "continuous_batching.md") in prompt
        assert str(NOTEBOOKS_DIR / "runtime" / "paged_kv_cache.md") not in prompt
        assert "items do not count against the optional-reading limit" in prompt

    assert str(NOTEBOOKS_DIR / "formats" / "gguf" / "qwen3_loader.md") in implementer
    assert str(
        NOTEBOOKS_DIR / "reference" / "implementation_sequence.md"
    ) in implementer
    assert str(
        NOTEBOOKS_DIR / "reference" / "framework_wiring_template.hpp"
    ) in implementer
    assert str(
        NOTEBOOKS_DIR / "reference" / "numeric_harness_template.hpp"
    ) in implementer
    assert str(
        NOTEBOOKS_DIR / "case_studies" / "008_tp2_paged_continuous.md"
    ) not in implementer
    assert "at most ~4 Read calls" not in implementer
    assert "Overall architecture" in planner
    assert "Iteration roadmap" in planner
    assert "runtime evidence" in implementer
    assert "first applicable fix MUST audit `src/model_loader.cpp`" in implementer
    assert "data_offset + tensor.offset" in implementer


def test_verified_008_case_study_is_rendered_only_for_full_combination(
    tmp_path: Path,
):
    case_path = str(
        NOTEBOOKS_DIR / "case_studies" / "008_tp2_paged_continuous.md"
    )
    full = implement_prompt(
        _requirements(
            target_model="/models/qwen3-8b-f16.gguf",
            features=[
                "Paged KV cache", "Continuous batching", "Tensor parallelism",
            ],
        ),
        tmp_path,
        NOTEBOOKS_DIR,
        iteration=1,
    )
    partial = implement_prompt(
        _requirements(
            features=["Paged KV cache", "Continuous batching"],
        ),
        tmp_path,
        NOTEBOOKS_DIR,
        iteration=1,
    )

    assert case_path in full
    assert case_path not in partial


def test_implementer_prompts_stop_before_delegated_verification(tmp_path: Path):
    req = _requirements(features=["Continuous batching"])
    prompts = (
        implement_prompt(req, tmp_path, NOTEBOOKS_DIR, iteration=1),
        implement_redo_prompt(
            req,
            tmp_path,
            NOTEBOOKS_DIR,
            iteration=2,
            prev_failure="HTTP smoke check failed",
        ),
    )

    for prompt in prompts:
        assert "return immediately" in prompt
        assert "Agent, Task, TaskOutput, Explore" in prompt
        assert "orchestrator owns independent validation" in prompt.casefold()


def test_tp_planner_prompt_forbids_real_model_tp1_and_background_agents(
    tmp_path: Path,
):
    req = _requirements(
        features=["Tensor parallelism"],
        target_model="/models/Qwen3-8B-F16.gguf",
    )
    req["tp_size"] = 2

    prompt = plan_prompt(req, tmp_path, NOTEBOOKS_DIR, iteration=1)

    assert "MUST NOT be loaded as TP1/single-device" in prompt
    assert "reduced synthetic layer" in prompt
    assert "real-model loading, forward" in prompt
    assert "Do not create a bring-up roadmap" in prompt
    assert "base.single_sequence_generation" in prompt
    assert "Do not infer a hipBLAS row-major/column-major bug" in prompt
    assert "transA/transB/M/N/K/lda/ldb/ldc" in prompt
    assert "finite or non-zero embedding is" in prompt
    assert "data_offset = align_up(tensor_info_end, general.alignment)" in prompt
    assert "first recovery" in prompt
    assert "return immediately" in prompt
    assert "Do not launch a subagent, background verifier" in prompt


def test_hardware_profile_source_document_exists():
    profiles = yaml.safe_load(
        (TASK_DIR / "orchestrator" / "hardware_profiles.yaml").read_text(
            encoding="utf-8"
        )
    )
    source = profiles["profiles"]["Hygon Z200"]["source_notebook"]
    assert source.startswith("notebooks/")
    assert (TASK_DIR / source).is_file()


def test_every_agent_prompt_receives_a_deterministic_route(tmp_path: Path):
    req = _requirements(features=["Continuous batching"])
    prompts = [
        implement_redo_prompt(
            req, tmp_path, NOTEBOOKS_DIR, iteration=2,
            prev_failure="GGUF tensor mismatch",
        ),
        review_prompt(
            req, tmp_path, NOTEBOOKS_DIR, iteration=1,
            outcome="logic_fail", failure="HTTP response is malformed",
            logs_dir=tmp_path,
        ),
        write_test_harness_prompt(
            req, tmp_path, NOTEBOOKS_DIR, iteration=1,
        ),
        c_repair_prompt(
            req, tmp_path, NOTEBOOKS_DIR, iteration=1,
            attempt=1, max_attempts=3, failure="RoPE logits are NaN",
            logs_dir=tmp_path,
        ),
        perf_test_prompt(
            req, tmp_path, NOTEBOOKS_DIR, iteration=1,
            review_feedback="scheduler mutex limits throughput",
            logs_dir=tmp_path,
        ),
        perf_plan_prompt(
            req, tmp_path, NOTEBOOKS_DIR, iteration=1,
            last_perf={"tokens_per_sec": 1.0}, logs_dir=tmp_path,
        ),
        retrospective_prompt(
            req, tmp_path, NOTEBOOKS_DIR, iteration=1,
            this_perf={"tokens_per_sec": 1.0}, logs_dir=tmp_path,
        ),
        failure_retrospective_prompt(
            req, tmp_path, NOTEBOOKS_DIR, iteration=1,
            failure_reason="out of memory", logs_dir=tmp_path,
        ),
    ]
    for prompt in prompts:
        assert "# Deterministic knowledge route (MANDATORY)" in prompt
        assert str(NOTEBOOKS_DIR / "routing.yaml") not in prompt
        assert "## Required reading" in prompt
        assert "## Optional reading" in prompt


def test_failure_retrospective_preserves_frozen_capability_scope(tmp_path: Path):
    req = _requirements(features=["Continuous batching"])
    prompt = failure_retrospective_prompt(
        req,
        tmp_path,
        NOTEBOOKS_DIR,
        iteration=2,
        failure_reason="fixed token output followed by timeout",
        logs_dir=tmp_path,
    )

    assert "plan_manifest.json" in prompt
    assert "iter2-implementer.attempt*.events.jsonl" in prompt
    assert "Do NOT recommend postponing or dropping a required" in prompt
    assert "deferred_suites" in prompt
    assert "fixed, mock, non-finite, or input-independent" in prompt


def test_planner_prompt_freezes_manifest_schema_and_prior_deadlines(tmp_path: Path):
    req = _requirements(features=["Continuous batching"])
    prior_logs = tmp_path / "logs" / "001"
    prior_logs.mkdir(parents=True)
    (prior_logs / "iter1-implementer.attempt1.events.jsonl").write_text(
        '{"type":"assistant","message":{"content":[{"type":"thinking",'
        '"thinking":"Embedding is finite but final logits are NaN and token 33 repeats"}]}}\n'
        '{"type":"user","message":{"content":[{"type":"tool_result",'
        '"content":"HTTP output !!!!!!!!; first attention output all zeros; logits NaN"}]}}\n',
        encoding="utf-8",
    )
    prompt = plan_prompt(
        req,
        tmp_path,
        NOTEBOOKS_DIR,
        iteration=2,
        prev_failures=(
            "A plan validation failed: milestones[0].iteration is missing"
        ),
        logs_dir=tmp_path / "logs" / "002",
    )

    assert "read the inherited" in prompt
    assert "frozen upper bounds" in prompt
    assert "`iteration`, `capabilities`, `suites`, and `deliverables`" in prompt
    assert "legacy aliases such as `id` or `gating_suites`" in prompt
    assert "exactly one milestone" in prompt
    assert "A plan validation failed" in prompt
    assert "MACHINE-VALIDATOR REPAIR MODE" in prompt
    assert "Before reading source or diagnostics" in prompt
    assert "data_offset + tensor.offset" in prompt
    assert "finite or non-zero embedding alone does not prove" in prompt
    assert "Repair" in prompt
    assert "every listed error before returning" in prompt
    assert "A milestone assignment is a deadline, not evidence" in prompt
    assert "ended before C" in prompt
    assert "iter*-implementer.attempt*.events.jsonl" in prompt
    assert "MUST perform the bounded" in prompt
    assert "Reading only the" in prompt
    assert "Failure evidence" in prompt
    assert "no" in prompt
    assert "correctness suite may be described as already passing" in prompt
    assert "Bounded Implementer evidence digest" in prompt
    assert "final logits are NaN and token 33 repeats" in prompt
    assert "HTTP output !!!!!!!!" in prompt


def test_policy_failure_planner_keeps_newer_successful_tool_evidence(tmp_path: Path):
    req = _requirements(features=["Continuous batching"])
    prior_logs = tmp_path / "logs" / "004"
    prior_logs.mkdir(parents=True)
    (prior_logs / "iter4-implementer.attempt1.events.jsonl").write_text(
        '{"type":"assistant","message":{"content":[{"type":"thinking",'
        '"thinking":"The old oracle had logits NaN and truncated output"}]}}\n'
        '{"type":"user","message":{"content":[{"type":"tool_result",'
        '"content":"Content: Paris; finish_reason=stop; completion_tokens=2; '
        'max_observed_batch_size=2; SUMMARY passed=16 failed=0"}]}}\n',
        encoding="utf-8",
    )

    prompt = plan_prompt(
        req,
        tmp_path,
        NOTEBOOKS_DIR,
        iteration=5,
        prev_failures=(
            "B execution policy failed: "
            "iter4-implementer.attempt1.events.jsonl:229 "
            "[subagent-delegation] tool=Agent"
        ),
        logs_dir=tmp_path / "logs" / "005",
    )

    assert "Execution-policy failure evidence ordering" in prompt
    assert "direct tool-result evidence are newer" in prompt
    assert "Do not use `server*.log` files" in prompt
    assert "direct Implementer tool" in prompt
    assert "[tool-result]" in prompt
    assert "Paris" in prompt
    assert "max_observed_batch_size=2" in prompt
    assert "SUMMARY passed=16 failed=0" in prompt


def test_fresh_implementer_keeps_prior_runtime_evidence_after_a_pass(tmp_path: Path):
    req = _requirements(
        features=["Tensor parallelism"],
        target_model="/models/Qwen3-8B-F16.gguf",
    )
    prior_logs = tmp_path / "logs" / "001"
    prior_logs.mkdir(parents=True)
    (prior_logs / "iter1-implementer.attempt1.events.jsonl").write_text(
        '{"type":"assistant","message":{"content":[{"type":"thinking",'
        '"thinking":"Embedding is finite but attention output is all zeros; '
        'final logits are NaN"}]}}\n',
        encoding="utf-8",
    )

    prompt = implement_prompt(
        req,
        tmp_path,
        NOTEBOOKS_DIR,
        iteration=2,
        prev_failure=None,
        logs_dir=tmp_path / "logs" / "002",
    )

    assert "Prior Implementer runtime evidence" in prompt
    assert "attention output is all zeros" in prompt
    assert "final logits are NaN" in prompt
    assert "data_offset = align_up" in prompt
    assert "data_offset + tensor.offset" in prompt


def test_review_hypotheses_cannot_override_routed_contracts(tmp_path: Path):
    prompt = implement_prompt(
        _requirements(),
        tmp_path,
        NOTEBOOKS_DIR,
        iteration=2,
        prev_failure="chat output was truncated",
        review_feedback="Use a guessed prompt directive.",
    )

    assert "binding contracts and reference source as authoritative" in prompt
    assert "reviews and retrospectives are diagnostic hypotheses" in prompt
    assert "the routed contract/reference wins" in prompt
