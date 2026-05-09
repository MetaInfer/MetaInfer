# Task11 DeepSeek TP Real Test Debugging Experience

This document records a complete troubleshooting process of `tests/test_deepseek_tp_real.py`, aiming to explain the source of "output degradation/garbled text" and provide reproducible fixes and verification steps.

## 1. Problem Phenomenon

- In the original `torchrun --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -q -s` output:
  - `output[0]` has short sentence repetition.
  - `output[1..4]` has大量 line breaks or degraded text.
- On the surface, looks like two types of problems:
  1. RoPE/position encoding implementation deviation;
  2. Some ranks (1/2/3) didn't correctly load weights.

## 2. First排除 "rank1/2/3 Not Loaded Weights" Hypothesis

Did TP vs HF same-prompt logits comparison (4 cards each printing) and found:

- Each rank's `max_diff/mean_diff` are completely identical;
- Each rank's top-1 token is also identical;
- Explanation: not "some card's weights missing" causing random rank corruption.

Conclusion: More like **implementation-level systematic deviation** (attention/RoPE/numerical path), not个别 rank weights not copied in.

## 3. This Round's Code Fixes

### 3.1 RoPE Numerical Path Changed to fp32 for Trig Functions

File: `engine/models/deepseek_v2.py`

- Original implementation in old RoPE helper cast `emb` to bf16 first then did `cos/sin`.
- Changed to:
  - Frequency/phase computed in fp32 for `cos/sin`;
  - Then cast back to activation dtype.

Effect: Reduces bf16 RoPE phase error accumulation causing repetition/degradation risk.
**Subsequent** pure TP complete fix (**YaRN + GPT-J style rotate**) see Section 8.

### 3.2 Added DeepSeek TP Test HF Debug Switch

File: `tests/test_deepseek_tp_real.py`

- Added environment variable switch:
  - When `DEEPSEEK_TP_HF_DEBUG=1`, calls
    `load_weights_from_hf_model(..., use_hf_logits_debug=True)`.

Effect: Quickly verify "multi-card pipeline and generation loop" correctness, isolating the problem to TP numerical alignment itself.

### 3.3 Fixed HF-Debug Multi-Card Broadcast Deadlock and Device Issues

File: `engine/models/deepseek_v2.py`

- Fix point 1: rank0's HF model forced to current GPU device (avoid CPU embedding + CUDA index error).
- Fix point 2: HF-debug branch lets **all ranks participate in broadcast**:
  - rank0 computes logits;
  - Other ranks create same-shaped tensor;
  - Together execute `torch.distributed.broadcast`.
- Fix point 3: Before broadcast, cast rank0 logits to unified dtype (avoid dtype inconsistency causing sync异常).

## 4. Verification Commands and Results

Command (recommend explicit master port to avoid 29500 conflict):

```bash
DEEPSEEK_TP_HF_DEBUG=1 torchrun --master_port 29631 --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -q -s
```

Key observed output (rank0):

- `output[0]='苏州园林...追求自然美...'`
- `output[1]='张量并行是一种分布式计算技术...'`
- `output[2]='夏天傍晚，夕阳如火...'`
- `output[3]='大语言模型是一种基于大量文本数据训练...'`
- `output[4]='面包是一种非常受欢迎的食品...'`

And appeared:

- `1 passed`

说明 under HF-debug broadcast path, results have recovered to normal Chinese semantic output.

## 5. Conclusion and Subsequent Suggestions

1. **Confirmed**: Not rank1/2/3 weights not loaded causing random garbled text.
2. **HF-debug ground truth broadcast** can serve as pipeline and communication regression baseline (not for delivery, only for troubleshooting).
3. **Pure TP forward** after补齐 **YaRN + vLLM DeepSeek-consistent RoPE style** in `engine/models/deepseek_v2.py`, without enabling `DEEPSEEK_TP_HF_DEBUG`, multi-card real test can get normal Chinese semantic output (see Section 8).
4. If subsequent strict per-token logits consistency with HF is needed, can still do per-layer hidden/logits diff; general engineering acceptance prioritizes "pure TP readable output + stable reproduction".

## 6. Common Troubleshooting Command Memo

```bash
# Regular TP real path
torchrun --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -q -s

# HF-debug baseline path (recommend with master_port)
DEEPSEEK_TP_HF_DEBUG=1 torchrun --master_port 29631 --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -q -s
```

## 7. Agent Reusable Experience Supplement (Phenomenon -> Root Cause -> Solution)

This part uses Agent language that can be directly executed when writing code next time.

### 7.1 What These Phenomena Usually Mean

- Phenomenon A: `output[0]` has short sentence repetition
  - High probability is **numerical/position-related path has systematic deviation** (RoPE, Norm, attention scale, mask/position semantics), not "model completely broken".
- Phenomenon B: `output[1..4]` lots of blank lines/degradation
  - Common when logits distribution is squashed or misaligned, sampling/argmax falls on abnormal high-frequency tokens (like line breaks) region.
- Phenomenon C: Different ranks look like they're running, but output quality poor
  - Can't directly judge "some rank didn't load weights", must first do **quantitative evidence**.

### 7.2 How I排除ed False Root Causes (Evidence Chain)

1. **First verify if rank problem holds**
   - Do 4-card rank-level `HF vs TP` logits comparison (same prompt, same position).
   - Result: Each rank `max_diff/mean_diff` identical, top-1 identical.
   - Conclusion: Not rank1/2/3 dropped weights; belongs to systematic deviation.

2. **Then verify "framework pipeline" is normal**
   - Add `DEEPSEEK_TP_HF_DEBUG=1`, let rank0 use HF ground truth logits, broadcast to other ranks.
   - If output recovers normal Chinese semantics, then说明:
     - Distributed communication main pipeline works;
     - Decode loop works;
     - Problem concentrated in "pure TP forward numerical alignment".

3. **Locate to high-risk subsystem**
   - Prioritize checking RoPE, Norm, attention numerical path; don't优先 guess scheduler/sampler.

### 7.3 How This Round's "True Cause" is Expressed in Engineering

- This round confirmed main cause is not "weights not loaded", but **pure TP forward has systematic numerical deviation with HF**.
- Confirmed and fixed direct issues:
  - RoPE trig functions computed prematurely in bf16, causing phase precision loss risk;
  - HF-debug branch multi-card broadcast protocol incomplete (non-rank0 not participating causes hang);
  - rank0 HF model device and input device inconsistency (CPU/GPU mixed use error).
- After fixing these, HF-debug path recovers correct output, proving debug baseline established successfully.

### 7.4 What Agent Should Do Next Time (Execution Checklist)

When再次 encountering "TP output repetition/blank lines/garbled", follow this fixed order:

1. **Do evidence first, don't guess first**
   - Run rank-level `HF vs TP` logits diff (max/mean/top1).
2. **Establish HF-debug baseline**
   - Add switch letting rank0 use HF logits broadcast, confirm pipeline works.
3. **Fix protocol issues**
   - Broadcast must have all ranks participate; dtype and shape must be identical; model and input must be on same device.
4. **Fix numerical issues**
   - RoPE `freq/cos/sin` fp32 first then cast;
   - Norm/softmax precision strategy consistent with HF.
5. **Last thing: dig into model implementation differences**
   - DeepSeek's `rope_scaling/yarn`, attention details, per-layer hidden diff.

### 7.5 Constraints for Future Code Generation (For Agent Itself)

- Don't directly attribute "text degradation" to某个 rank weights not loaded.
- First establish "HF ground truth broadcast" usable baseline, then do pure TP convergence.
- Without per-layer alignment evidence, don't claim "TP correctly implemented".
- After each fix,保留 reproducible commands and对照 output, ensure next time can auto-regress.

## 8. Pure TP Fixes (Especially YaRN)

This section records key fixes **not dependent on HF forward**, only on self-developed TP path, for next time writing `DeepseekForCausalLMTP` or similar models with `rope_scaling` to directly reference.

### 8.1 Root Cause Summary

- DeepSeek V2's `config.json` has `rope_scaling.type` as **`yarn`**, with `factor`, `beta_fast`, `beta_slow`, `original_max_position_embeddings`, `mscale`, `mscale_all_dim` fields.
- Early custom implementation if written as "standard Neox RoPE + only `rope_theta`", **without applying YaRN's frequency interpolation and extrapolation混合**, will cause Q/K phase mismatch with pre-trained weights, manifesting as short sentence repetition,大量 line breaks, garbled-like output.
- Additionally, vLLM / DeepSeek uses **`is_neox_style=False` GPT-J style interleaved rotation** (`rotate_gptj`) for `qk_rope_head_dim` rotation, different from Neox's "split-half then cat" rotation; mixing will further misalign.

### 8.2 Code Locations (File: `engine/models/deepseek_v2.py`)

1. **Configuration**
   - In `DeepseekV2TPConfig` added `rope_scaling: dict | None`, in `_load_deepseek_v2_tp_config` read `cfg.rope_scaling` from HuggingFace `AutoConfig`.

2. **YaRN's `inv_freq`**
   - When `rope_scaling.get("type") == "yarn"`, for `qk_rope_head_dim` even dimensions construct `1/(theta^(2i/d))`, then consistent with HF/vLLM同类 logic:
     - Interpolation branch `1 / (factor * pos_freqs)` and extrapolation branch `1/pos_freqs` **linear混合** (`yarn_find_correction_range` + `yarn_linear_ramp_mask` idea).

3. **YaRN amplitude scaling**
   - Multiply `cos/sin` by `yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)` (consistent with vLLM `DeepseekScalingRotaryEmbedding`).

4. **Attention `scaling`**
   - When `rope_type` is yarn, multiply `self.scaling = (qk_head_dim ** -0.5)` by **`yarn_get_mscale(factor, mscale_all_dim) ** 2** (aligning with vLLM `DeepseekV2Attention`'s `mscale` usage).

5. **Rotation operator**
   - For `q_pe`, `k_pe`, use **GPT-J style** `rotate_half` + `cos/sin` `repeat_interleave(2, dim=-1)` layout, not Neox's `cat(freqs, freqs)` + half-split rotation.
   - `cos/sin` still computed in fp32, then `to(activation dtype)`, avoiding bf16 trig functions being too粗糙.

### 8.3 Pure TP Verification Command (Don't Set `DEEPSEEK_TP_HF_DEBUG`)

```bash
torchrun --master_port 29641 --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -q -s
```

(`--master_port` can be used per environment to avoid 29500 occupied port; not related to specific port.)

### 8.4 Agent Memo (One Sentence)

- See **`rope_scaling: yarn`** and output looks like "runnable but like broken model", **first fix RoPE by HF/vLLM's DeepSeek YaRN + correct rotate style**, then talk about MoE/EP tuning.
