from pathlib import Path
import tempfile
import unittest

from metainfer.orchestrator.tasks import get_task
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.perf import (
    _scan_profile_dir,
)
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.pipeline import Orchestrator
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.prompts import (
    _CPP_PHASE_NOTEBOOKS,
    c_repair_prompt,
    implement_prompt,
    perf_plan_prompt,
    plan_prompt,
    review_prompt,
)
from metainfer.server.forms import (
    list_task_types,
    load_form_schema,
    validate_submission,
)
from metainfer.server.registry import get as get_web_plugin


TASK_TYPE = "gen-infer-framework-cpp"


def _requirements():
    return {
        "task_id": "cpp-test",
        "task_type": TASK_TYPE,
        "target_model": "/models/test",
        "target_hardware": "NVIDIA H100",
        "accelerator_backend": "CUDA",
        "cpp_standard": "C++20",
        "build_type": "RelWithDebInfo",
        "features": ["Paged KV cache", "Continuous batching"],
    }


class CppFrameworkTaskTest(unittest.TestCase):
    def test_all_routed_cpp_notebooks_exist(self):
        notebooks = Path(__file__).resolve().parents[1] / "notebooks"
        routed = {
            path
            for paths in _CPP_PHASE_NOTEBOOKS.values()
            for path in paths
        }
        for path in routed:
            self.assertTrue((notebooks / path).is_file(), path)

    def test_card_schema_and_validation(self):
        task_ids = [entry["id"] for entry in list_task_types()]
        self.assertIn(TASK_TYPE, task_ids)

        schema = load_form_schema(TASK_TYPE)
        self.assertIsNotNone(schema)
        self.assertEqual(schema["label"], "构建 C++ 推理框架")
        field_keys = {field["key"] for field in schema["fields"]}
        self.assertTrue({
            "target_model",
            "model_family",
            "weight_dtype",
            "kv_cache_dtype",
            "target_hardware",
            "accelerator_backend",
            "assigned_devices",
            "tensor_parallel_size",
            "cpp_standard",
            "build_type",
        }.issubset(field_keys))
        hardware_field = next(field for field in schema["fields"] if field["key"] == "target_hardware")
        self.assertIn("Hygon Z200SM", {option["label"] for option in hardware_field["options"]})
        backend_field = next(field for field in schema["fields"] if field["key"] == "accelerator_backend")
        self.assertIn("Hygon DTK / HIP", {option["label"] for option in backend_field["options"]})

        self.assertTrue(validate_submission(TASK_TYPE, _requirements())["ok"])
        valid_devices = {**_requirements(), "assigned_devices": "0, 1,2,3"}
        self.assertTrue(validate_submission(TASK_TYPE, valid_devices)["ok"])
        invalid = validate_submission(TASK_TYPE, {})
        self.assertFalse(invalid["ok"])
        self.assertIn("cpp_standard", invalid["errors"])

    def test_plugin_owns_its_orchestrator(self):
        plugin = get_task(TASK_TYPE)
        self.assertEqual(
            plugin.cli_module,
            "metainfer.tasks.gen_infer_framework_cpp.orchestrator.cli",
        )
        self.assertEqual(
            plugin.phases_module,
            "metainfer.tasks.gen_infer_framework_cpp.orchestrator.phases",
        )

        web_plugin = get_web_plugin(TASK_TYPE)
        self.assertIsNotNone(web_plugin)
        self.assertEqual(web_plugin.detail_view_module, "app/cpp-gf-detail")
        self.assertEqual(web_plugin.extra_stylesheets, ["cpp-gf.css"])
        self.assertTrue(
            (web_plugin.frontend_dir / "cpp-gf-detail.js").is_file()
        )
        route_paths = {
            route.path for route in web_plugin.build_router(web_plugin).routes
        }
        self.assertTrue(
            {"/iterations", "/charts", "/state-graph", "/qa/start"}
            .issubset(route_paths)
        )

    def test_notebooks_live_inside_cpp_task_package(self):
        from metainfer.tasks.gen_infer_framework_cpp.orchestrator import (
            orchestrator as orchestrator_module,
        )

        notebooks = orchestrator_module._NOTEBOOKS_DIR
        self.assertEqual(notebooks.parent.name, "gen_infer_framework_cpp")
        self.assertTrue((notebooks / "00_contracts" / "cpp").is_dir())
        self.assertTrue((notebooks / "09_cpp_inference").is_dir())

    def test_cpp_task_uses_the_immutable_perf_oracle(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.req = {"task_type": TASK_TYPE}
        expected = ("ok", {"tokens_per_sec": 1.0}, None)
        orchestrator._do_perf_test_oracle = lambda *args: expected

        actual = orchestrator._do_perf_test(1, Path("/tmp/iter"), None, None)
        self.assertIs(actual, expected)

    def test_cpp_contract_is_present_across_agent_phases(self):
        req = _requirements()
        iter_dir = Path("/tmp/cpp-iteration")
        notebooks = Path("/tmp/notebooks")
        logs = Path("/tmp/logs")

        plan = plan_prompt(req, iter_dir, notebooks, 1, logs_dir=logs)
        implement = implement_prompt(req, iter_dir, notebooks, 1, logs_dir=logs)
        repair = c_repair_prompt(
            req, iter_dir, notebooks, 1, 1, 3, "compile failed", logs,
        )
        review = review_prompt(req, iter_dir, notebooks, 1, logs_dir=logs)
        perf_plan = perf_plan_prompt(req, iter_dir, notebooks, 1, logs_dir=logs)

        for prompt in (plan, implement, repair, review, perf_plan):
            self.assertIn("NON-NEGOTIABLE implementation language: C++ first", prompt)
            self.assertIn("LANGUAGE_BOUNDARY.md", prompt)

        self.assertIn("cmake -S . -B build", implement)
        self.assertIn("CMakeLists.txt", implement)
        self.assertIn("compiled native server binary", implement)
        self.assertNotIn('python3 -c "import server"', implement)
        self.assertIn("Python runtime substitution", review)
        self.assertIn("cmake --build build --parallel", repair)

    def test_cpp_notebook_route_and_hardware_contract_are_deterministic(self):
        req = _requirements()
        req.update({
            "target_hardware": "Hygon Z200SM",
            "accelerator_backend": "Hygon DTK / HIP",
            "model_family": "Qwen3 MoE",
            "features": [
                "Paged KV cache",
                "Continuous batching",
                "Tensor parallelism",
                "Streaming responses",
            ],
            "hardware_profile_path": "/state/hardware_profile.json",
            "hardware_profile": {
                "vendor_family": "hygon",
                "device_count": 4,
                "hip_architectures": ["gfx906"],
                "validation": {"status": "matched", "blockers": []},
            },
        })
        prompt = plan_prompt(req, Path("/tmp/cpp"), Path("/tmp/notebooks"), 1)
        self.assertIn("C++ knowledge-base route", prompt)
        self.assertIn("REQUIRED phase bundle", prompt)
        self.assertIn("COMPONENT guides", prompt)
        self.assertIn("read only the 2-5 files", prompt)
        self.assertIn("00_contracts/cpp/hardware_profile_contracts.md", prompt)
        self.assertIn("09_cpp_inference/07_paged_kv_cache.md", prompt)
        self.assertIn("09_cpp_inference/08_continuous_batching.md", prompt)
        self.assertIn("09_cpp_inference/09_tensor_parallelism.md", prompt)
        self.assertIn("09_cpp_inference/10_cpp_http_server.md", prompt)
        self.assertIn("09_cpp_inference/13_qwen3_moe_runtime.md", prompt)
        self.assertIn("NON-NEGOTIABLE detected hardware contract", prompt)
        self.assertIn('"hip_architectures": [', prompt)
        self.assertNotIn("Glob notebooks/**/*.md", prompt)

    def test_python_framework_prompt_does_not_inherit_cpp_contract(self):
        req = {
            "task_id": "python-test",
            "task_type": "gen-infer-framework",
            "target_model": "/models/test",
            "target_hardware": "Hygon K100AI",
        }
        iter_dir = Path("/tmp/python-iteration")
        notebooks = Path("/tmp/notebooks")

        plan = plan_prompt(req, iter_dir, notebooks, 1)
        implement = implement_prompt(req, iter_dir, notebooks, 1)

        self.assertNotIn("NON-NEGOTIABLE implementation language: C++", plan)
        self.assertNotIn("CMake/build checks", plan)
        self.assertNotIn("CMakeLists.txt", implement)
        self.assertIn('python3 -c "import server"', implement)
        self.assertIn("Glob notebooks/**/*.md", plan)

    def test_native_profiler_artifacts_are_discovered(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("trace.json.gz", "native.nsys-rep", "rocprof.csv"):
                (root / name).write_bytes(b"x" * 2048)
            (root / "ignored.bin").write_bytes(b"x" * 2048)

            artifacts = _scan_profile_dir(root)
            names = {Path(item["path"]).name for item in artifacts}
            self.assertEqual(
                names,
                {"trace.json.gz", "native.nsys-rep", "rocprof.csv"},
            )


if __name__ == "__main__":
    unittest.main()
