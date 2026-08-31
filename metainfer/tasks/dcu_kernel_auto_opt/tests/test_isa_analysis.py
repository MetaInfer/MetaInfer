from __future__ import annotations

from ..orchestrator.isa_analysis import (
    analyze_inline_asm_source,
    evaluate_inline_asm_gate,
    resolve_dtk_llvm_bin,
    summarize_kernel_symbols,
    summarize_isa_text,
)


def test_isa_summary_counts_memory_wait_and_mmac():
    summary = summarize_isa_text("""
global_load_dwordx4 v[0:3], v[4:5], off
s_waitcnt vmcnt(0)
ds_write_b128 v0, v[0:3]
s_barrier
v_mmac_i32_16x16x32_i8 a[0:3], v[0:1], v[2:3], a[0:3]
""")

    assert summary["instruction_counts"]["global_load"] == 1
    assert summary["instruction_counts"]["ds_write"] == 1
    assert summary["instruction_counts"]["waitcnt"] == 1
    assert summary["instruction_counts"]["mmac"] == 1
    assert summary["load_wait0_ds_write_windows"] == 1


def test_isa_summary_attributes_resources_to_profiled_symbols():
    partial = "_Z20w8a8_partial_kernelv"
    combine = "_Z20w8a8_combine_kernelv"
    disassembly = f"""
0000000000001000 <{partial}>:
global_load_dwordx4 v[0:3], v[4:5], off
v_mmac_i32_16x16x32_i8 a[0:3], v[0:1], v[2:3], a[0:3]
0000000000002000 <{combine}>:
global_load_dword v0, v[1:2], off
global_store_dword v[1:2], v0, off
"""
    metadata = f"""
amdhsa.kernels:
  - .args:
    .group_segment_fixed_size: 10240
    .name:           {partial}
    .sgpr_count:     27
    .vgpr_count:     40
    .wavefront_size: 64
  - .args:
    .group_segment_fixed_size: 0
    .name:           {combine}
    .sgpr_count:     38
    .vgpr_count:     16
    .wavefront_size: 64
"""

    summary = summarize_kernel_symbols(
        disassembly,
        metadata,
        kernel_names=[partial, combine],
        primary_kernel_name=partial,
    )

    assert summary["kernel_name"] == partial
    assert summary["resources"]["vgpr_count"] == 40
    assert summary["resources"]["sgpr_count"] == 27
    assert summary["resources"]["lds_bytes"] == 10240
    assert len(summary["profiled_kernels"]) == 2
    assert summary["profiled_kernels"][1]["resources"]["sgpr_count"] == 38


def test_inline_asm_source_distinguishes_compiler_barrier():
    analysis = analyze_inline_asm_source(r'''
asm volatile("" ::: "memory");
__asm__ __volatile__("s_waitcnt vmcnt(0)" ::: "memory");
''')

    assert analysis["asm_block_count"] == 2
    assert analysis["compiler_barrier_count"] == 1
    assert analysis["raw_instruction_asm_count"] == 1
    assert analysis["raw_instruction_fingerprints"] == [
        "s_waitcnt vmcnt(0)"
    ]


def test_new_inline_asm_requires_plan_and_trusted_isa():
    before = analyze_inline_asm_source("")
    after = analyze_inline_asm_source(
        'asm volatile("s_waitcnt vmcnt(0)" ::: "memory");'
    )

    rejected = evaluate_inline_asm_gate(
        before=before,
        after=after,
        proposal={},
        isa_evidence={"available": False},
    )
    accepted = evaluate_inline_asm_gate(
        before=before,
        after=after,
        proposal={
            "isa_optimization": {
                "strategy": "inline_asm",
                "target_instructions": ["s_waitcnt vmcnt(0)"],
            }
        },
        isa_evidence={"available": True},
    )

    assert rejected["required"] is True
    assert rejected["passed"] is False
    assert accepted["passed"] is True


def test_inline_asm_gate_enforces_control_phase_and_verified_targets():
    before = analyze_inline_asm_source("")
    after = analyze_inline_asm_source(
        'asm volatile("s_waitcnt vmcnt(0)" ::: "memory");'
    )
    proposal = {
        "isa_optimization": {
            "strategy": "inline_asm",
            "target_instructions": ["s_waitcnt vmcnt(0)"],
        }
    }

    closed = evaluate_inline_asm_gate(
        before=before,
        after=after,
        proposal=proposal,
        isa_evidence={"available": True},
        raw_inline_asm_allowed=False,
    )
    wrong_target = evaluate_inline_asm_gate(
        before=before,
        after=after,
        proposal=proposal,
        isa_evidence={"available": True},
        raw_inline_asm_allowed=True,
        verified_target_instructions=["v_pk_add_u16"],
    )
    open_gate = evaluate_inline_asm_gate(
        before=before,
        after=after,
        proposal=proposal,
        isa_evidence={"available": True},
        raw_inline_asm_allowed=True,
        verified_target_instructions=["s_waitcnt vmcnt(0)"],
    )

    assert closed["passed"] is False
    assert wrong_target["passed"] is False
    assert open_gate["passed"] is True


def test_resolve_dtk_llvm_bin_accepts_aillvm_layout(tmp_path):
    toolchain = tmp_path / "aillvm" / "bin"
    toolchain.mkdir(parents=True)
    for name in (
        "llvm-objcopy",
        "clang-offload-bundler",
        "llvm-readobj",
        "llvm-objdump",
    ):
        (toolchain / name).touch()

    assert resolve_dtk_llvm_bin(toolchain) == toolchain
