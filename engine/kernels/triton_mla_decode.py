"""Simplified Triton MLA decode kernel for DeepSeek-V2.

Adapted from vLLM's triton_decode_attention.py with:
- MLA-only path (IS_MLA=True, V = trans(K))
- No FP8, no logit_cap, no paged block_table
- Contiguous buffer indexing (Req_to_tokens = identity)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_decode_stage1(
    Q,
    K_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """Stage 1: split-KV attention with MLA (V = trans(K))."""
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    # Load Q (first Lk dims)
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    # Load Q rope portion (extra BLOCK_DPE dims)
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        qpe = tl.load(Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        if BLOCK_DPE > 0:
            base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # Contiguous indexing: kv_loc = position (identity mapping)
            kv_loc = tl.load(
                Req_to_tokens + stride_req_b * cur_batch + offs_n,
                mask=offs_n < split_kv_end, other=0,
            )

            # Load K (c_kv || k_pe, 192 dims)
            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0, cache_modifier=".cg",
            )

            # Q @ K^T for first BLOCK_DMODEL dims (q_nope_proj @ c_kv)
            qk = tl.dot(q, k.to(q.dtype))

            # Q_pe @ K_pe for rope dims
            if BLOCK_DPE > 0:
                offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0, cache_modifier=".cg",
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))

            qk *= sm_scale
            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # MLA: V = trans(K), reuse c_kv (first Lv dims of K)
            v = tl.trans(k)

            # Online softmax + accumulate
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # Store partial results
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_lse = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )
        tl.store(Att_Out + offs_mid_lse, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
def _mla_decode_stage2(
    Mid_O,
    o,
    lse,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    """Stage 2: reduce partial results across KV splits."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(o + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / e_sum, mask=mask_d)
    tl.store(lse + cur_batch * stride_lse_bs + cur_head, e_max + tl.log(e_sum))


def triton_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_len: int,
    num_kv_splits: int,
    sm_scale: float,
    lv: int = 128,
) -> torch.Tensor:
    """Triton MLA decode attention.

    Args:
        q: [batch=1, num_heads, Lk=192] — projected query (q_nope_proj || q_pe)
        kv_cache: [batch=1, max_seq_len, 1, Lk=192] — contiguous KV cache (c_kv || k_pe)
        kv_len: number of valid cached tokens
        num_kv_splits: number of KV splits for parallel reduction
        sm_scale: softmax scale (typically 1/sqrt(Lk))
        lv: V dimension = kv_lora_rank (default 128 for DeepSeek-V2-Lite)

    Returns:
        o: [batch=1, num_heads, Lv] — attention output in latent space
    """
    batch, num_heads, Lk = q.shape
    Lv = lv  # kv_lora_rank: the compressed latent dimension

    # Ensure kv_cache is 4D: [batch, seq_len, kv_heads=1, dim]
    if kv_cache.dim() == 3:
        kv_cache = kv_cache.unsqueeze(2)

    # Create contiguous index mapping [0, 1, 2, ...]
    req_to_tokens = torch.arange(kv_cache.shape[1], device=q.device, dtype=torch.int32)
    b_seq_len = torch.tensor([kv_len], device=q.device, dtype=torch.int32)

    # Allocate output and intermediate buffers
    o = torch.zeros(batch, num_heads, Lv, device=q.device, dtype=q.dtype)
    lse = torch.zeros(batch, num_heads, device=q.device, dtype=torch.float32)

    BLOCK_DV = triton.next_power_of_2(Lv)
    att_out = torch.zeros(
        batch, num_heads, num_kv_splits, BLOCK_DV + 1,
        device=q.device, dtype=torch.float32,
    )

    # Determine BLOCK_DMODEL and BLOCK_DPE
    if Lk == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    elif Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0

    BLOCK = 32
    BLOCK_H = 16
    kv_group_num = num_heads  # MLA: kv_heads=1 per rank, all Q heads share

    # Stage 1: split-KV attention
    grid1 = (batch, triton.cdiv(num_heads, min(BLOCK_H, kv_group_num)), num_kv_splits)
    _mla_decode_stage1[grid1](
        q, kv_cache, sm_scale, req_to_tokens, b_seq_len, att_out,
        req_to_tokens.stride(0),
        q.stride(0), q.stride(1),
        kv_cache.stride(1), kv_cache.stride(2),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=num_heads,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=num_kv_splits,
        Lk=Lk,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )

    # Stage 2: reduce across splits
    grid2 = (batch, num_heads)
    _mla_decode_stage2[grid2](
        att_out, o, lse, b_seq_len,
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        o.stride(0), o.stride(1), lse.stride(0),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )

    return o
