#!/usr/bin/env python3
"""Summarize dsh-mini-smoke worker_0 iteration outcomes."""
import json
import os
import sys

base = sys.argv[1] if len(sys.argv) > 1 else (
    "/workspace/MetaInfer/nodes/worker29/workspaces/dsh-mini-smoke/"
    "workers/worker_0/iterations/tp8_wqkv_a_m32"
)
print("=== iteration results ===")
if not os.path.isdir(base):
    print("no iterations dir:", base)
    sys.exit(0)
rows = []
for d in sorted(os.listdir(base), key=lambda x: int(x[9:]) if x[9:].isdigit() else 0):
    f = os.path.join(base, d, "iteration.json")
    if not os.path.exists(f):
        continue
    j = json.load(open(f))
    acc = j.get("acceptance") or {}
    rows.append((
        d,
        j.get("build_success"),
        j.get("correctness_passed"),
        acc.get("accepted"),
        acc.get("best_us"),
        acc.get("candidate_us"),
        j.get("baseline_us"),
        str(j.get("failure_reason"))[:70],
    ))
for r in rows:
    print(
        f"{r[0]}: build={r[1]} correct={r[2]} accepted={r[3]} "
        f"best={r[4]} cand={r[5]} bl={r[6]} err={r[7]}"
    )
