from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ..orchestrator.build import (
    BuildConfigError,
    BuildProfile,
    SubmissionManifest,
    SystemBuilder,
)


def triton_profile() -> BuildProfile:
    return BuildProfile(
        backend="triton",
        kernel_language="Triton",
        target_hardware="test-gpu",
        detected_hardware=None,
        gpu_arch="test-arch",
        detected_gpu_arch=None,
        compiler=sys.executable,
        compiler_version=subprocess.run(
            [sys.executable, "--version"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=True,
        ).stdout.strip(),
        cxx_compiler=None,
        cxx_compiler_version=None,
        cmake=None,
        cmake_version=None,
        generator=None,
        build_tool=None,
        build_tool_version=None,
        fixed_flags=["-O3"],
        allowed_options=[],
    ).with_fingerprint()


def test_system_builder_owns_profile_and_build_script(tmp_path):
    profile = triton_profile()
    system = tmp_path / "system"
    builder = SystemBuilder(profile, system)
    assert (system / "build_profile.json").is_file()
    assert (system / "build.sh").is_file()
    assert json.loads((system / "build_profile.json").read_text())["fingerprint"] == profile.fingerprint

    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "kernel.py").write_text("def kernel():\n    return 1\n", encoding="utf-8")
    (submission / "submission.yaml").write_text(
        "schema_version: 1\nsources: [kernel.py]\nentrypoint: kernel.py\n",
        encoding="utf-8",
    )
    result = builder.build(submission, tmp_path / "build")
    assert result.passed
    assert result.report["build_fingerprint"] == profile.fingerprint
    assert not (submission / "__pycache__").exists()


def test_manifest_rejects_commands_and_unknown_build_options(tmp_path):
    profile = triton_profile()
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "kernel.py").write_text("pass\n", encoding="utf-8")
    (submission / "submission.yaml").write_text(
        """schema_version: 1
sources: [kernel.py]
entrypoint: kernel.py
requested_build_options:
  arbitrary_nvcc_flags: --disable-system-checks
""",
        encoding="utf-8",
    )
    try:
        SubmissionManifest.load(submission, profile)
    except BuildConfigError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("unknown compiler option should be rejected")


def test_frozen_build_profile_detects_mutation(tmp_path):
    profile = triton_profile()
    system = tmp_path / "system"
    builder = SystemBuilder(profile, system)
    data = json.loads((system / "build_profile.json").read_text(encoding="utf-8"))
    data["gpu_arch"] = "different"
    (system / "build_profile.json").write_text(json.dumps(data), encoding="utf-8")
    try:
        builder.verify()
    except BuildConfigError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("mutated build profile should be rejected")


def test_cuda_cmake_is_generated_from_frozen_profile(tmp_path):
    cmake = shutil.which("cmake")
    assert cmake
    cmake_version = subprocess.run(
        [cmake, "--version"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=True,
    ).stdout.strip()[:2000]
    compiler_version = subprocess.run(
        [sys.executable, "--version"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=True,
    ).stdout.strip()
    cxx = shutil.which("c++")
    make = shutil.which("make")
    assert cxx and make
    profile = BuildProfile(
        backend="cuda",
        kernel_language="CUDA C++",
        target_hardware="test-gpu",
        detected_hardware=None,
        gpu_arch="90",
        detected_gpu_arch=None,
        compiler=sys.executable,
        compiler_version=compiler_version,
        cxx_compiler=cxx,
        cxx_compiler_version=subprocess.run(
            [cxx, "--version"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=True,
        ).stdout.strip()[:2000],
        cmake=cmake,
        cmake_version=cmake_version,
        generator="Unix Makefiles",
        build_tool=make,
        build_tool_version=subprocess.run(
            [make, "--version"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=True,
        ).stdout.strip()[:2000],
        fixed_flags=["-O3"],
        allowed_options=["fast_math", "max_registers"],
    ).with_fingerprint()
    system = tmp_path / "system"
    SystemBuilder(profile, system)
    cmake_text = (system / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'CUDA_ARCHITECTURES "90"' in cmake_text
    assert "--use_fast_math" in cmake_text
    assert "--maxrregcount" in cmake_text


@pytest.mark.skipif(not shutil.which("nvcc"), reason="nvcc is not installed")
def test_native_cuda_submission_builds_with_system_cmake(tmp_path):
    nvcc = shutil.which("nvcc")
    cmake = shutil.which("cmake")
    cxx = shutil.which("c++")
    make = shutil.which("make")
    assert nvcc and cmake and cxx and make
    version = lambda argv: subprocess.run(  # noqa: E731
        argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()[:2000]
    profile = BuildProfile(
        backend="cuda",
        kernel_language="CUDA C++",
        target_hardware="compile-only",
        detected_hardware=None,
        gpu_arch="80",
        detected_gpu_arch=None,
        compiler=nvcc,
        compiler_version=version([nvcc, "--version"]),
        cxx_compiler=cxx,
        cxx_compiler_version=version([cxx, "--version"]),
        cmake=cmake,
        cmake_version=version([cmake, "--version"]),
        generator="Unix Makefiles",
        build_tool=make,
        build_tool_version=version([make, "--version"]),
        fixed_flags=["-O3"],
        allowed_options=["fast_math", "max_registers"],
    ).with_fingerprint()
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "kernel.cu").write_text(
        'extern "C" __global__ void gemm_candidate(float* out) { out[0] = 1.0f; }\n',
        encoding="utf-8",
    )
    (submission / "submission.yaml").write_text(
        """schema_version: 1
sources: [kernel.cu]
requested_build_options:
  fast_math: false
  max_registers: 128
""",
        encoding="utf-8",
    )
    harness_source = Path(__file__).resolve().parents[1] / "harness" / "user_gemm" / "evaluate_native.cpp"
    result = SystemBuilder(
        profile, tmp_path / "system", harness_source=harness_source
    ).build(
        submission, tmp_path / "build"
    )
    assert result.passed, result.report.get("stderr_tail")
    assert result.report["artifacts"]
    assert (tmp_path / "build" / "metainfer_gemm_harness").is_file()


def test_empty_optional_tool_paths_use_discovered_defaults():
    profile = BuildProfile.from_requirements({
        "kernel_language": "Triton",
        "target_hardware": "test-gpu",
        "gpu_arch": "test-arch",
        "compiler_path": "",
        "allowed_build_options": [],
    })
    assert profile.compiler == str(Path(sys.executable).resolve())
    assert profile.allowed_options == []


def test_backend_rejects_unsupported_allowed_option():
    with pytest.raises(BuildConfigError, match="unsupported"):
        BuildProfile.from_requirements({
            "kernel_language": "Triton",
            "target_hardware": "test-gpu",
            "gpu_arch": "test-arch",
            "allowed_build_options": ["Fast math"],
        })
