# Submission contract

The evaluator bundle defines the concrete ABI. Preserve the initial
submission's entry points, filenames and callable signatures unless the public
task explicitly permits changing them.

Every submission must contain `submission.yaml`:

```yaml
schema_version: 1
sources:
  - kernel.cu
  - binding.cpp
include_dirs:
  - include
requested_build_options:
  fast_math: false
  max_registers: 128
```

CUDA permits only `fast_math` and `max_registers`; HIP permits only
`fast_math`; Triton accepts no compiler options. Unknown options, absolute
paths, parent traversal, symlinks, and backend-incompatible source suffixes
are rejected before CMake runs.

General rules:

- keep all candidate sources and build metadata under `submission/`;
- do not use symlinks;
- do not read evaluator paths, private cases or previous raw system reports;
- do not write outside `submission/` during implementation;
- do not create CMakeLists.txt or build.sh; the system owns both;
- do not invoke a different nvcc/hipcc, host C++ compiler, CMake generator, or
  change the frozen GPU architecture;
- keep compilation reproducible and avoid downloading dependencies;
- place a short description of the current change in `CHANGELOG.md`.

The evaluator runs with these environment variables for its own runners:

```text
METAINFER_SUBMISSION_DIR
METAINFER_REPORT_PATH
METAINFER_EVALUATION_PHASE
METAINFER_EVALUATION_ROLE
METAINFER_EVALUATOR_BUNDLE
METAINFER_BUILD_ARTIFACT_DIR
METAINFER_BUILD_FINGERPRINT
METAINFER_BENCHMARK_PROTOCOL
```

They are an evaluator-runner interface, not a candidate interface. Candidate
code should not depend on their values.
