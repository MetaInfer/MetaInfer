# K100 / gfx928 fixed profiling route

The WebUI's `Hygon K100` selection resolves one system-owned execution
profile. Agents do not select tools or construct commands.

The compilation route is equivalent to the C++ framework task's hardware
binding:

```text
cmake -S system_build -B ITER_BUILD -G Ninja \
  -DMETAINFER_SUBMISSION_FILE=ITER_BUILD/submission.cmake
cmake --build ITER_BUILD --target \
  metainfer_gemm_candidate metainfer_gemm_harness
```

The generated CMake freezes the resolved DTK `hipcc`, Release `-O3`, C++/HIP
17, and `HIP_ARCHITECTURES=gfx928`. Exact resolved paths, versions, commands,
flags and the profile fingerprint are written to the compile report.

E first consumes the Harness GPU-event benchmark for every weighted shape.
It then invokes the Harness as `profile CASE_ID` for M=1, M=16 and M=4096 of
the public `wq_b TP=4` workload. On rocprofv3, the system command has this
fixed shape:

```text
rocprofv3 --pmc <FROZEN_GROUP> --output-format csv json \
  --output-directory <SYSTEM_OUTPUT> \
  --kernel-include-regex w8a8_scaled_ -- \
  metainfer_gemm_harness profile <CASE_ID>
```

For a DTK installation that provides legacy rocprof, the fixed fallback is:

```text
rocprof -i <SYSTEM_COUNTER_FILE> -o <SYSTEM_OUTPUT.csv> \
  --timestamp on metainfer_gemm_harness profile <CASE_ID>
```

The available-counter query is performed once when the profile is frozen;
unsupported names are removed from the whitelist rather than guessed. Tool
path, version, counter groups and representative shapes are fingerprinted.
Profiler failure is an E-stage infrastructure failure for this K100 profile.

The hardware profile is diagnostic evidence for F. Champion promotion remains
owned by correctness plus the complete weighted multi-shape event benchmark;
the three profiler cases do not replace or reweight that score.

# Interpretation checklist

Record the target GPU and exact compiler flags before interpreting profiler
data. Useful signals include achieved occupancy, waves/SM or waves/CU, register
and shared-memory pressure, memory transaction efficiency, cache hit rate,
tensor-core/MFMA utilization, synchronization stalls and launch count.

Do not optimize a single profiler counter in isolation. A lower occupancy
kernel may still win through better instruction-level parallelism or data
reuse. Conversely, a headline speedup smaller than run-to-run noise is not a
promotion.

Use public per-shape results to identify the class that changed. Held-out
results are deliberately summarized so implementation choices generalize
rather than overfit case IDs.
