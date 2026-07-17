from pathlib import Path
import tempfile
import unittest

from metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.native_validation import (
    hardware_validation_errors,
    native_accelerator_errors,
    validate_cpp_artifacts,
)


class CppNativeValidationTest(unittest.TestCase):
    def test_native_delivery_tree_passes_static_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "include").mkdir()
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "CMakeLists.txt").write_text("project(native LANGUAGES CXX)\n", encoding="utf-8")
            (root / "LANGUAGE_BOUNDARY.md").write_text("No Python runtime.\n", encoding="utf-8")
            (root / "src" / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
            (root / "serve.sh").write_text("#!/bin/sh\nexec ./build/native_server \"$1\"\n", encoding="utf-8")
            self.assertEqual(validate_cpp_artifacts(root), [])

    def test_python_server_and_hardware_blocker_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "include").mkdir()
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "CMakeLists.txt").write_text("project(native)\n", encoding="utf-8")
            (root / "LANGUAGE_BOUNDARY.md").write_text("\n", encoding="utf-8")
            (root / "src" / "main.cpp").write_text("int main() {}\n", encoding="utf-8")
            (root / "serve.sh").write_text("python3 server.py\n", encoding="utf-8")
            errors = validate_cpp_artifacts(root)
            self.assertTrue(any("Python" in item for item in errors))

        req = {"hardware_profile": {"validation": {"blockers": ["no /dev/kfd access"]}}}
        self.assertEqual(hardware_validation_errors(req), ["no /dev/kfd access"])

        gpu_req = {
            "target_hardware": "Hygon Z200SM",
            "accelerator_backend": "Hygon DTK / HIP",
        }
        self.assertTrue(native_accelerator_errors(
            gpu_req, {"loaded_accelerator_libraries": []},
        ))
        library_only = native_accelerator_errors(
            gpu_req,
            {"loaded_accelerator_libraries": ["/opt/dtk/lib/libamdhip64.so"]},
        )
        self.assertTrue(any("device FD" in item for item in library_only))

        fd_only = native_accelerator_errors(
            gpu_req, {"gpu_device_fds": ["/dev/kfd"]},
        )
        self.assertTrue(any("runtime library" in item for item in fd_only))

        self.assertEqual(native_accelerator_errors(
            gpu_req,
            {
                "loaded_accelerator_libraries": ["/opt/dtk/lib/libamdhip64.so"],
                "gpu_device_fds": ["/dev/kfd"],
            },
        ), [])

    def test_accelerator_evidence_must_belong_to_same_native_process(self):
        gpu_req = {
            "target_hardware": "Hygon Z200SM",
            "accelerator_backend": "Hygon DTK / HIP",
        }
        split_evidence = {
            "loaded_accelerator_libraries": ["/opt/dtk/lib/libamdhip64.so"],
            "gpu_device_fds": ["/dev/kfd"],
            "processes": [
                {
                    "exe_in_iteration": True,
                    "is_python": False,
                    "accelerator_libraries": ["/opt/dtk/lib/libamdhip64.so"],
                    "gpu_device_fds": [],
                },
                {
                    "exe_in_iteration": False,
                    "is_python": False,
                    "accelerator_libraries": [],
                    "gpu_device_fds": ["/dev/kfd"],
                },
            ],
        }
        errors = native_accelerator_errors(gpu_req, split_evidence)
        self.assertTrue(any("same native server process" in item for item in errors))

        split_evidence["processes"][0]["gpu_device_fds"] = ["/dev/kfd"]
        self.assertEqual(native_accelerator_errors(gpu_req, split_evidence), [])

    def test_tensor_parallelism_requires_native_rank_processes(self):
        req = {
            "target_hardware": "Hygon Z200SM",
            "accelerator_backend": "Hygon DTK / HIP",
            "features": ["Tensor parallelism"],
            "tensor_parallel_size": "2",
            "assigned_devices": "0,1",
        }
        process = {
            "exe_in_iteration": True,
            "is_python": False,
            "accelerator_libraries": ["/opt/dtk/lib/libamdhip64.so"],
            "gpu_device_fds": ["/dev/kfd", "/dev/dri/renderD128"],
        }
        evidence = {
            "loaded_accelerator_libraries": ["/opt/dtk/lib/libamdhip64.so"],
            "gpu_device_fds": ["/dev/kfd", "/dev/dri/renderD128"],
            "processes": [process],
        }

        errors = native_accelerator_errors(req, evidence)

        self.assertTrue(any("requested 2 native ranks" in item for item in errors))
        evidence["processes"].append(dict(process))
        self.assertEqual(native_accelerator_errors(req, evidence), [])


if __name__ == "__main__":
    unittest.main()
