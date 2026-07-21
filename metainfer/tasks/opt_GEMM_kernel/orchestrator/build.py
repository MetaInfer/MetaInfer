"""System-owned build profiles and native GEMM candidate compilation.

The optimizer controls source code and a narrow ``submission.yaml`` manifest.
It never controls CMake, the compiler executable, architecture flags, or the
build command.  A profile is materialized once per task and fingerprinted so
baseline and challenger binaries are directly comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from metainfer.orchestrator.requirements import req_field


class BuildConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BuildProfile:
    backend: str
    kernel_language: str
    target_hardware: str
    detected_hardware: Optional[str]
    gpu_arch: str
    detected_gpu_arch: Optional[str]
    compiler: str
    compiler_version: str
    cxx_compiler: Optional[str]
    cxx_compiler_version: Optional[str]
    cmake: Optional[str]
    cmake_version: Optional[str]
    generator: Optional[str]
    build_tool: Optional[str]
    build_tool_version: Optional[str]
    fixed_flags: List[str] = field(default_factory=list)
    allowed_options: List[str] = field(default_factory=list)
    schema_version: int = 1
    fingerprint: str = ""

    @classmethod
    def from_requirements(
        cls, req: Dict[str, Any], hardware_profile: Optional[Dict[str, Any]] = None,
    ) -> "BuildProfile":
        hardware_build = dict((hardware_profile or {}).get("build") or {})
        language = (
            "HIP" if hardware_profile and hardware_profile.get("backend") == "hip"
            else str(req_field(req, "kernel_language", "")).strip()
        )
        lower = language.lower()
        if "triton" in lower:
            backend = "triton"
            default_compiler = sys.executable
        elif "hip" in lower or "composable" in lower or lower == "ck":
            backend = "hip"
            default_compiler = shutil.which("hipcc") or ""
        else:
            backend = "cuda"
            default_compiler = shutil.which("nvcc") or ""

        if hardware_build:
            compiler = _first_executable(hardware_build.get("compiler_candidates") or [])
        else:
            compiler = str(req_field(req, "compiler_path", "") or "").strip() or default_compiler
        if not compiler:
            raise BuildConfigError(f"no compiler found for backend {backend!r}")
        compiler_path = Path(compiler).expanduser().resolve()
        if not compiler_path.is_file():
            raise BuildConfigError(f"compiler does not exist: {compiler_path}")

        requested_arch = str(req_field(req, "gpu_arch", "")).strip()
        profile_arch = str((hardware_profile or {}).get("gpu_arch") or "").strip()
        gpu_arch = _normalize_arch(backend, profile_arch or requested_arch)
        if not gpu_arch:
            raise BuildConfigError("gpu_arch is required; examples: 90, 89, gfx942")
        target_hardware = str(req_field(req, "target_hardware", "")).strip()
        if not target_hardware:
            raise BuildConfigError("target_hardware is required")
        detected_hardware, detected_arch = _probe_gpu(backend)
        if detected_arch and detected_arch != gpu_arch:
            raise BuildConfigError(
                f"requested gpu_arch {gpu_arch!r} does not match detected {detected_arch!r}"
            )

        cmake_path: Optional[Path] = None
        cxx_path: Optional[Path] = None
        generator: Optional[str] = None
        cmake_version: Optional[str] = None
        build_tool_path: Optional[Path] = None
        build_tool_version: Optional[str] = None
        if backend in {"cuda", "hip"}:
            cmake_value = (
                _first_executable(hardware_build.get("cmake_candidates") or [])
                if hardware_build else
                str(req_field(req, "cmake_path", "") or "").strip()
                or shutil.which("cmake") or ""
            )
            if not cmake_value:
                raise BuildConfigError("cmake is required for native GEMM kernels")
            cmake_path = Path(cmake_value).expanduser().resolve()
            if not cmake_path.is_file():
                raise BuildConfigError(f"cmake does not exist: {cmake_path}")
            requested_generator = str(req_field(req, "cmake_generator", "")).strip()
            generator = (
                str(hardware_build.get("generator")) if hardware_build
                else requested_generator or ("Ninja" if shutil.which("ninja") else "Unix Makefiles")
            )
            if generator not in {"Ninja", "Unix Makefiles"}:
                raise BuildConfigError("cmake_generator must be Ninja or Unix Makefiles")
            cmake_version = _version([str(cmake_path), "--version"])
            cxx_value = (
                _first_executable(hardware_build.get("host_compiler_candidates") or [])
                if hardware_build else
                str(req_field(req, "cxx_compiler_path", "") or "").strip()
                or shutil.which("c++") or shutil.which("g++") or ""
            )
            if not cxx_value:
                raise BuildConfigError("a host C++ compiler is required for native kernels")
            cxx_path = Path(cxx_value).expanduser().resolve()
            if not cxx_path.is_file():
                raise BuildConfigError(f"host C++ compiler does not exist: {cxx_path}")
            if generator == "Ninja":
                tool_value = (
                    _first_executable(hardware_build.get("build_tool_candidates") or [])
                    if hardware_build else shutil.which("ninja") or ""
                )
                version_args = [tool_value, "--version"] if tool_value else []
            elif generator == "Unix Makefiles":
                tool_value = shutil.which("make") or shutil.which("gmake") or ""
                version_args = [tool_value, "--version"] if tool_value else []
            else:
                tool_value = ""
                version_args = []
            if tool_value:
                build_tool_path = Path(tool_value).resolve()
                build_tool_version = _version(version_args)

        requested_options = [] if hardware_build else req_field(req, "allowed_build_options", []) or []
        if isinstance(requested_options, str):
            requested_options = [requested_options]
        option_names = {
            "Fast math": "fast_math",
            "Max registers": "max_registers",
            "fast_math": "fast_math",
            "max_registers": "max_registers",
        }
        unknown_requested = sorted(
            str(value) for value in requested_options if str(value) not in option_names
        )
        if unknown_requested:
            raise BuildConfigError(f"unknown allowed_build_options: {unknown_requested}")
        allowed_options = [
            option_names[str(value)] for value in requested_options
            if str(value) in option_names
        ]
        backend_supported = (
            {"fast_math", "max_registers"} if backend == "cuda"
            else {"fast_math"} if backend == "hip"
            else set()
        )
        unsupported = sorted(set(allowed_options) - backend_supported)
        if unsupported:
            raise BuildConfigError(
                f"requested build options are unsupported by {backend}: {unsupported}"
            )

        profile = cls(
            backend=backend,
            kernel_language=language,
            target_hardware=target_hardware,
            detected_hardware=detected_hardware,
            gpu_arch=gpu_arch,
            detected_gpu_arch=detected_arch,
            compiler=str(compiler_path),
            compiler_version=_version([str(compiler_path), "--version"]),
            cxx_compiler=str(cxx_path) if cxx_path else None,
            cxx_compiler_version=(
                _version([str(cxx_path), "--version"]) if cxx_path else None
            ),
            cmake=str(cmake_path) if cmake_path else None,
            cmake_version=cmake_version,
            generator=generator,
            build_tool=str(build_tool_path) if build_tool_path else None,
            build_tool_version=build_tool_version,
            fixed_flags=list(map(str, hardware_build.get("release_flags") or ["-O3"])),
            allowed_options=sorted(set(allowed_options)),
        )
        return profile.with_fingerprint()

    def with_fingerprint(self) -> "BuildProfile":
        data = asdict(self)
        data["fingerprint"] = ""
        digest = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return BuildProfile(**{**data, "fingerprint": digest})

    def materialize(self, root: Path, harness_source: Optional[Path] = None) -> None:
        root.mkdir(parents=True, exist_ok=True)
        profile_path = root / "build_profile.json"
        if profile_path.exists():
            existing = json.loads(profile_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != self.fingerprint:
                raise BuildConfigError("frozen BuildProfile fingerprint changed")
        else:
            profile_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        if self.backend in {"cuda", "hip"}:
            cmake_text = _render_cmake(self, harness_source)
            cmake_path = root / "CMakeLists.txt"
            if cmake_path.exists() and cmake_path.read_text(encoding="utf-8") != cmake_text:
                raise BuildConfigError("system-owned CMakeLists.txt changed")
            cmake_path.write_text(cmake_text, encoding="utf-8")
        build_sh = root / "build.sh"
        script = _render_build_script(profile_path, root, harness_source)
        if build_sh.exists() and build_sh.read_text(encoding="utf-8") != script:
            raise BuildConfigError("system-owned build.sh changed")
        build_sh.write_text(script, encoding="utf-8")
        build_sh.chmod(0o755)

    @classmethod
    def load(cls, path: Path) -> "BuildProfile":
        data = json.loads(path.read_text(encoding="utf-8"))
        profile = cls(**data)
        expected = profile.with_fingerprint().fingerprint
        if profile.fingerprint != expected:
            raise BuildConfigError("invalid BuildProfile fingerprint")
        return profile


@dataclass(frozen=True)
class SubmissionManifest:
    sources: List[str]
    include_dirs: List[str] = field(default_factory=list)
    entrypoint: Optional[str] = None
    requested_build_options: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def load(cls, submission_dir: Path, profile: BuildProfile) -> "SubmissionManifest":
        path = submission_dir / "submission.yaml"
        if not path.is_file():
            raise BuildConfigError("submission/submission.yaml is required")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict) or raw.get("schema_version", 1) != 1:
            raise BuildConfigError("submission.yaml must use schema_version=1")
        sources = raw.get("sources")
        if not isinstance(sources, list) or not sources:
            raise BuildConfigError("submission.yaml requires a non-empty sources list")
        include_dirs = raw.get("include_dirs") or []
        if not isinstance(include_dirs, list):
            raise BuildConfigError("include_dirs must be a list")
        entrypoint = raw.get("entrypoint")
        options = raw.get("requested_build_options") or {}
        if not isinstance(options, dict):
            raise BuildConfigError("requested_build_options must be a mapping")
        unknown = sorted(set(options) - set(profile.allowed_options))
        if unknown:
            raise BuildConfigError(f"build options are not allowed by the profile: {unknown}")
        _validate_options(options, profile)
        manifest = cls(
            sources=[str(v) for v in sources],
            include_dirs=[str(v) for v in include_dirs],
            entrypoint=str(entrypoint) if entrypoint else None,
            requested_build_options=options,
        )
        manifest.resolve(submission_dir, profile)
        return manifest

    def resolve(self, submission_dir: Path, profile: BuildProfile) -> Dict[str, List[Path]]:
        allowed_suffixes = {
            "cuda": {".cu", ".cc", ".cpp", ".cxx"},
            "hip": {".hip", ".cc", ".cpp", ".cxx"},
            "triton": {".py"},
        }[profile.backend]
        sources = [
            _safe_path(submission_dir, value, must_exist=True)
            for value in self.sources
        ]
        bad = [path.name for path in sources if path.suffix.lower() not in allowed_suffixes]
        if bad:
            raise BuildConfigError(f"source suffix is not allowed for {profile.backend}: {bad}")
        includes = [
            _safe_path(submission_dir, value, must_exist=True, require_dir=True)
            for value in self.include_dirs
        ]
        if profile.backend == "triton":
            if not self.entrypoint:
                raise BuildConfigError("Triton submission requires entrypoint")
            entry = _safe_path(submission_dir, self.entrypoint, must_exist=True)
            if entry not in sources:
                raise BuildConfigError("Triton entrypoint must also appear in sources")
        return {"sources": sources, "include_dirs": includes}


@dataclass
class BuildResult:
    passed: bool
    artifact_dir: Path
    report: Dict[str, Any]
    failure: Optional[str] = None
    infra_failure: bool = False


class SystemBuilder:
    def __init__(
        self,
        profile: BuildProfile,
        system_dir: Path,
        harness_source: Optional[Path] = None,
    ) -> None:
        self.profile = profile
        self.system_dir = system_dir
        self.harness_source = harness_source.resolve() if harness_source else None
        profile.materialize(system_dir, self.harness_source)

    def verify(self) -> None:
        on_disk = BuildProfile.load(self.system_dir / "build_profile.json")
        if on_disk.fingerprint != self.profile.fingerprint:
            raise BuildConfigError("BuildProfile no longer matches the active task")
        if _version([self.profile.compiler, "--version"]) != self.profile.compiler_version:
            raise BuildConfigError("compiler version changed after BuildProfile was frozen")
        if self.profile.cmake and _version([self.profile.cmake, "--version"]) != self.profile.cmake_version:
            raise BuildConfigError("CMake version changed after BuildProfile was frozen")
        if self.profile.cxx_compiler and _version(
            [self.profile.cxx_compiler, "--version"]
        ) != self.profile.cxx_compiler_version:
            raise BuildConfigError("host C++ compiler changed after BuildProfile was frozen")
        if self.profile.build_tool and _version(
            [self.profile.build_tool, "--version"]
        ) != self.profile.build_tool_version:
            raise BuildConfigError("CMake build tool changed after BuildProfile was frozen")
        detected_hardware, detected_arch = _probe_gpu(self.profile.backend)
        if self.profile.detected_gpu_arch and detected_arch != self.profile.detected_gpu_arch:
            raise BuildConfigError("detected GPU architecture changed after BuildProfile was frozen")
        if self.profile.detected_hardware and detected_hardware != self.profile.detected_hardware:
            raise BuildConfigError("detected GPU device changed after BuildProfile was frozen")
        if self.profile.backend in {"cuda", "hip"}:
            expected = _render_cmake(self.profile, self.harness_source)
            if (self.system_dir / "CMakeLists.txt").read_text(encoding="utf-8") != expected:
                raise BuildConfigError("system CMakeLists.txt was modified")
        expected_script = _render_build_script(
            self.system_dir / "build_profile.json", self.system_dir, self.harness_source
        )
        if (self.system_dir / "build.sh").read_text(encoding="utf-8") != expected_script:
            raise BuildConfigError("system build.sh was modified")

    def build(self, submission_dir: Path, build_dir: Path) -> BuildResult:
        started = time.time()
        try:
            self.verify()
            _validate_no_symlinks(submission_dir)
            manifest = SubmissionManifest.load(submission_dir, self.profile)
            resolved = manifest.resolve(submission_dir, self.profile)
        except (BuildConfigError, OSError, ValueError) as exc:
            return BuildResult(False, build_dir, {}, str(exc), False)

        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        if self.profile.backend == "triton":
            argv = [self.profile.compiler, "-m", "py_compile", *map(str, resolved["sources"])]
            return self._run_build(
                argv, submission_dir, build_dir, started,
                env_overrides={"PYTHONPYCACHEPREFIX": str(build_dir / "pycache")},
            )

        cache = build_dir / "submission.cmake"
        cache.write_text(_render_submission_cmake(resolved, manifest), encoding="utf-8")
        configure = [
            str(self.profile.cmake),
            "-S", str(self.system_dir),
            "-B", str(build_dir),
            "-G", str(self.profile.generator),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DMETAINFER_SUBMISSION_FILE={cache}",
        ]
        configured = self._run_command(configure, build_dir, "configure")
        if configured[0] != 0:
            report = {
                "passed": False,
                "build_fingerprint": self.profile.fingerprint,
                "backend": self.profile.backend,
                "gpu_arch": self.profile.gpu_arch,
                "compiler": self.profile.compiler,
                "compiler_version": self.profile.compiler_version,
                "command": configure,
                "duration_s": time.time() - started,
                "artifacts": [],
                "stdout_tail": configured[1][-2000:],
                "stderr_tail": configured[2][-2000:],
            }
            (build_dir / "compile-report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            return BuildResult(
                False, build_dir, report, "CMake configure failed",
                "infrastructure failure" in configured[2],
            )
        targets = ["metainfer_gemm_candidate"]
        if self.harness_source is not None:
            targets.append("metainfer_gemm_harness")
        build = [str(self.profile.cmake), "--build", str(build_dir), "--target", *targets]
        result = self._run_build(build, build_dir, build_dir, started)
        result.report["configure_command"] = configure
        result.report["build_command"] = build
        result.report["fixed_flags"] = list(self.profile.fixed_flags)
        (build_dir / "compile-report.json").write_text(
            json.dumps(result.report, indent=2), encoding="utf-8"
        )
        return result

    def _run_build(
        self, argv: List[str], cwd: Path, build_dir: Path, started: float,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> BuildResult:
        rc, stdout, stderr = self._run_command(
            argv, cwd, "build", build_dir, env_overrides=env_overrides
        )
        artifacts = [
            str(path) for path in build_dir.rglob("*")
            if path.is_file() and (
                path.name.startswith("libmetainfer_gemm_candidate")
                or path.name == "metainfer_gemm_harness"
                or path.suffix in {".so", ".dll", ".dylib", ".pyc"}
            )
        ]
        report = {
            "passed": rc == 0,
            "build_fingerprint": self.profile.fingerprint,
            "backend": self.profile.backend,
            "gpu_arch": self.profile.gpu_arch,
            "compiler": self.profile.compiler,
            "compiler_version": self.profile.compiler_version,
            "command": argv,
            "duration_s": time.time() - started,
            "artifacts": artifacts,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }
        (build_dir / "compile-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return BuildResult(
            rc == 0, build_dir, report, None if rc == 0 else "system build failed",
            rc == 127 and "infrastructure failure" in stderr,
        )

    def _run_command(
        self,
        argv: List[str],
        cwd: Path,
        label: str,
        log_dir: Optional[Path] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> tuple[int, str, str]:
        target = log_dir or cwd
        env = dict(os.environ)
        env.update(env_overrides or {})
        try:
            proc = subprocess.run(
                argv, cwd=str(cwd), text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=1800, check=False, env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 127, "", f"{label} infrastructure failure: {exc}"
        (target / f"{label}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (target / f"{label}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        return proc.returncode, proc.stdout or "", proc.stderr or ""


def _version(argv: List[str]) -> str:
    try:
        proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildConfigError(f"cannot query tool version: {argv[0]}: {exc}") from exc
    if proc.returncode != 0:
        raise BuildConfigError(f"tool version command failed: {argv[0]}")
    return (proc.stdout or "unknown").strip()[:2000]


def _first_executable(candidates: List[Any]) -> str:
    for raw in candidates:
        value = str(raw)
        if "/" in value:
            path = Path(value).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
        else:
            found = shutil.which(value)
            if found:
                return str(Path(found).resolve())
    return ""


def _normalize_arch(backend: str, value: str) -> str:
    if backend == "cuda":
        value = value.lower().removeprefix("sm_").removeprefix("compute_").replace(".", "")
        if not re.fullmatch(r"[0-9]{2,3}[a-z]?", value):
            return ""
    elif backend == "hip":
        value = value.lower()
        if not re.fullmatch(r"gfx[0-9a-f]+", value):
            return ""
    else:
        value = value.strip()
    return value


def _probe_gpu(backend: str) -> tuple[Optional[str], Optional[str]]:
    if backend == "cuda" and shutil.which("nvidia-smi"):
        name = _probe_line([
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
        ])
        raw_arch = _probe_line([
            "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader",
        ])
        return name, _normalize_arch("cuda", raw_arch or "") or None
    if backend == "hip" and shutil.which("rocminfo"):
        try:
            proc = subprocess.run(
                ["rocminfo"], text=True, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, timeout=20, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None, None
        match = re.search(r"\b(gfx[0-9a-f]+)\b", proc.stdout or "", re.IGNORECASE)
        return None, match.group(1).lower() if match else None
    return None, None


def _probe_line(argv: List[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    return lines[0] if lines else None


def _safe_path(
    root: Path, value: str, *, must_exist: bool, require_dir: bool = False,
) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise BuildConfigError(f"submission path must be relative and contained: {value!r}")
    resolved_root = root.resolve()
    path = (root / relative).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise BuildConfigError(f"submission path escapes the root: {value!r}")
    if must_exist and not path.exists():
        raise BuildConfigError(f"submission path does not exist: {value!r}")
    if path.is_symlink():
        raise BuildConfigError(f"submission path may not be a symlink: {value!r}")
    if require_dir and not path.is_dir():
        raise BuildConfigError(f"submission include path is not a directory: {value!r}")
    if not require_dir and must_exist and not path.is_file():
        raise BuildConfigError(f"submission source is not a file: {value!r}")
    return path


def _validate_no_symlinks(root: Path) -> None:
    if not root.is_dir():
        raise BuildConfigError(f"submission directory does not exist: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise BuildConfigError(
                f"submission may not contain symlinks: {path.relative_to(root)}"
            )


def _validate_options(options: Dict[str, Any], profile: BuildProfile) -> None:
    if "fast_math" in options and not isinstance(options["fast_math"], bool):
        raise BuildConfigError("fast_math must be true or false")
    if "max_registers" in options:
        value = options["max_registers"]
        if not isinstance(value, int) or isinstance(value, bool) or not 16 <= value <= 255:
            raise BuildConfigError("max_registers must be an integer in [16, 255]")
        if profile.backend != "cuda":
            raise BuildConfigError("max_registers is only available for CUDA")


def _cmake_quote(path: Path) -> str:
    return '"' + str(path).replace("\\", "/").replace('"', '\\"') + '"'


def _render_submission_cmake(
    resolved: Dict[str, List[Path]], manifest: SubmissionManifest,
) -> str:
    sources = "\n  ".join(_cmake_quote(path) for path in resolved["sources"])
    includes = "\n  ".join(_cmake_quote(path) for path in resolved["include_dirs"])
    options = manifest.requested_build_options
    return (
        f"set(METAINFER_SOURCES\n  {sources}\n)\n"
        f"set(METAINFER_INCLUDE_DIRS\n  {includes}\n)\n"
        f"set(METAINFER_FAST_MATH {'ON' if options.get('fast_math') else 'OFF'})\n"
        f"set(METAINFER_MAX_REGISTERS {int(options.get('max_registers', 0))})\n"
    )


def _render_cmake(profile: BuildProfile, harness_source: Optional[Path] = None) -> str:
    compiler_var = "CMAKE_CUDA_COMPILER" if profile.backend == "cuda" else "CMAKE_HIP_COMPILER"
    language = "CUDA" if profile.backend == "cuda" else "HIP"
    arch_property = "CUDA_ARCHITECTURES" if profile.backend == "cuda" else "HIP_ARCHITECTURES"
    option_block = ""
    if profile.backend == "cuda":
        option_block = """
if(METAINFER_FAST_MATH)
  target_compile_options(metainfer_gemm_candidate PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
if(METAINFER_MAX_REGISTERS GREATER 0)
  target_compile_options(metainfer_gemm_candidate PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=${METAINFER_MAX_REGISTERS}>)
endif()
"""
    else:
        option_block = """
if(METAINFER_FAST_MATH)
  target_compile_options(metainfer_gemm_candidate PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>)
endif()
"""
    host_line = (
        f"set(CMAKE_CUDA_HOST_COMPILER {_cmake_quote(Path(profile.cxx_compiler))} CACHE FILEPATH \"\" FORCE)"
        if profile.backend == "cuda" and profile.cxx_compiler else ""
    )
    harness_block = ""
    if harness_source is not None:
        backend_define = "METAINFER_USE_HIP=1" if profile.backend == "hip" else "METAINFER_USE_CUDA=1"
        harness_block = f"""
set(METAINFER_HARNESS_SOURCE {_cmake_quote(harness_source)})
set_source_files_properties(${{METAINFER_HARNESS_SOURCE}} PROPERTIES LANGUAGE {language})
add_executable(metainfer_gemm_harness ${{METAINFER_HARNESS_SOURCE}})
target_compile_definitions(metainfer_gemm_harness PRIVATE {backend_define})
set_target_properties(metainfer_gemm_harness PROPERTIES
  {arch_property} "{profile.gpu_arch}"
  CXX_STANDARD 17
  {language}_STANDARD 17
)
target_compile_options(metainfer_gemm_harness PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:{' '.join(profile.fixed_flags)}>
  $<$<COMPILE_LANGUAGE:{language}>:{' '.join(profile.fixed_flags)}>
)
target_link_libraries(metainfer_gemm_harness PRIVATE ${{CMAKE_DL_LIBS}})
"""
    fixed_options = " ".join(profile.fixed_flags)
    return f"""cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_COMPILER {_cmake_quote(Path(profile.cxx_compiler))} CACHE FILEPATH "" FORCE)
{host_line}
set({compiler_var} {_cmake_quote(Path(profile.compiler))} CACHE FILEPATH "" FORCE)
project(metainfer_gemm_candidate LANGUAGES CXX {language})

if(NOT DEFINED METAINFER_SUBMISSION_FILE)
  message(FATAL_ERROR "METAINFER_SUBMISSION_FILE is required")
endif()
include(${{METAINFER_SUBMISSION_FILE}})

add_library(metainfer_gemm_candidate SHARED ${{METAINFER_SOURCES}})
target_include_directories(metainfer_gemm_candidate PRIVATE ${{METAINFER_INCLUDE_DIRS}})
set_target_properties(metainfer_gemm_candidate PROPERTIES
  {arch_property} "{profile.gpu_arch}"
  CXX_STANDARD 17
  {language}_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(metainfer_gemm_candidate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:{fixed_options}>
  $<$<COMPILE_LANGUAGE:{language}>:{fixed_options}>
)
{option_block}
{harness_block}
"""


def _render_build_script(
    profile_path: Path,
    root: Path,
    harness_source: Optional[Path] = None,
) -> str:
    harness_arg = (
        f" --harness-source {json.dumps(str(harness_source))}" if harness_source else ""
    )
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [ \"$#\" -ne 2 ]; then echo 'usage: build.sh SUBMISSION_DIR BUILD_DIR' >&2; exit 2; fi\n"
        f"exec {json.dumps(sys.executable)} -m "
        "metainfer.tasks.opt_GEMM_kernel.orchestrator.build "
        f"--profile {json.dumps(str(profile_path))} --system-dir {json.dumps(str(root))} "
        f"--submission \"$1\" --build-dir \"$2\"{harness_arg}\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a frozen GEMM system build")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--system-dir", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--harness-source", type=Path)
    args = parser.parse_args(argv)
    result = SystemBuilder(
        BuildProfile.load(args.profile), args.system_dir, harness_source=args.harness_source
    ).build(
        args.submission, args.build_dir
    )
    if result.failure:
        print(result.failure, file=sys.stderr)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
