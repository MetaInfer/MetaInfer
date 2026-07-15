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
        self.assertEqual(native_accelerator_errors(
            gpu_req,
            {"loaded_accelerator_libraries": ["/opt/dtk/lib/libamdhip64.so"]},
        ), [])


if __name__ == "__main__":
    unittest.main()
