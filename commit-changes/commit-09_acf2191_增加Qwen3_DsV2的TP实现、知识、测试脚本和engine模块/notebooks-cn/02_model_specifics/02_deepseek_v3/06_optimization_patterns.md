# DeepSeek V3 — 优化模式

## FP8 量化

DeepSeek V3 在训练阶段即面向 FP8，推理可借此显著加速。

### Attention 中的 FP8（与 MLA 相关）
```python
# 吸收后的 W_UK 可量化为 FP8 以加速 GEMM
# Q_absorbed = Q @ W_UK^T  →  FP8 GEMM
q_absorbed = fp8_gemm(q_nope, w_uk_fp8, scale_q, scale_w)
```

### MoE 中的 FP8
专家权重可存为 FP8、在 FP8 上计算：
```python
fused_moe_fp8(hidden_fp16, expert_w1_fp8, expert_w2_fp8,
                scale_w1, scale_w2, topk_weights, topk_ids)
```

## DeepEP（弹性专家并行）

在 EP 模式下，高效 all-to-all 的一种工程化组织：

```
1. 本地路由 → 定 token 应去哪些 GPU/专家
2. All-to-all → 把 token 发到目标专家所在 GPU
3. 专家计算 → 各卡只算本地专家
4. All-to-all → 把结果发回
5. 合并 → 按专家权重加和
```

DeepEP 对 2、4 步常做：

- 通信与计算重叠
- 弹性 batch 以缓解负载不均
- 跨节点时配合 RDMA

## 对代码生成有意义的优化小结

为 DeepSeek V3 生成推理代码时，可依次考虑：

1. **MLA 吸收模式**：预计算/折叠吸收后的投影，降低每 token attention 成本
2. **FP8 GEMM**：线性层与专家计算用 FP8 核
3. **融合 MoE**：256 专家规模下几乎必需
4. **分组 Top-K**：保证多组专家都有贡献
5. **KV 压缩存储**：只存潜向量 + RoPE 相关 key
6. **MTP 投机**：延迟敏感时打开内建投机
