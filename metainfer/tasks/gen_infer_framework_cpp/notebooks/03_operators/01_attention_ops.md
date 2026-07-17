# Attention算子：Prefill与Paged Decode

先读：`00_contracts/attention_kv_contracts.md`和
`01_framework_design/03_kv_cache.md`。

## 1. 两条接口

```cpp
Status PrefillAttention(const PrefillAttentionArgs&, TensorView output,
                        const OperatorContext&);
Status PagedDecodeAttention(const PagedDecodeArgs&, TensorView output,
                            const OperatorContext&);
```

Prefill处理Ragged Token和Causal Mask；Decode直接读取Block Table和KV Pool。二者
输出语义Shape相同，但Metadata和访问模式不同，不能用一个含大量隐式分支的接口
掩盖Shape错误。

## 2. Reference与优化阶梯

Tiny Reference为QK GEMM、Scale/Mask、Stable Softmax和PV GEMM，仅用于小Shape。
Prefill优化采用Tiled/Online Softmax，避免物化完整Score。Decode Kernel遍历物理
Block并在线归约，禁止每步Gather完整连续KV。

## 3. 数值不变量

- Scale只应用一次；
- Softmax先减Row Max，FP32累加；
- Causal判断使用逻辑Position；
- Padding Query/KV不读写Slot；
- 全Mask或非法KV Length返回错误而不是NaN；
- GQA Query Head到KV Head映射使用模型/TP后的Local配置。

## 4. Dispatch

Implementation ID由Backend、Architecture、DType、Head Dim、Layout、Prefill/Decode
和能力Probe决定。未命中优化Kernel时使用已验证的原生Reference/基线，不能切换
到CPU或Python。

## 5. 测试

覆盖长度1、非整Block、跨Block、Ragged Batch、GQA、不同Head Dim、长Context、
逻辑Position与物理Offset分离、NaN/Inf和OOM。每个优化Kernel与Reference比较
Output、LogSumExp或等价中间量，并记录Tolerance和支持范围。

