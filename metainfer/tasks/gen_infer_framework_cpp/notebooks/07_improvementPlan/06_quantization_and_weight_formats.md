# 改进计划：量化与权重格式

状态：proposed  
来源：早期8B量化Checkpoint兼容失败；SRC-LLAMA、SRC-MLC、SRC-TRT、SRC-SAFE。  
前置Contract：`00_contracts/weight_loading_contracts.md`、
`00_contracts/qwen3_model_contracts.md`、`02_model_specifics/02_qwen3/04_model_loader.md`。

## 1. 问题定义

“INT4模型”不是完整格式。至少需要明确Container、Tensor名称、Bit Packing、Group Size、
Scale/Zero Point、Symmetric/Asymmetric、Quant Axis、Outlier、Activation DType、Accumulation
DType和对应GEMM Kernel。只根据文件名或`quantization_config`中的一个字段选择Kernel，
很容易得到能加载但数值完全错误的框架。

## 2. 目标和非目标

目标是建立可扩展的Quantization Descriptor、严格Loader和经过DTK验证的Weight-only
执行路径。未支持格式必须在分配大块设备内存前明确拒绝。

非目标：第一阶段不承诺所有GGUF、AWQ、GPTQ、FP8或Vendor私有格式；不在生产请求中
动态转换整个模型；不使用反量化到FP16后仍宣称量化计算加速。

## 3. Canonical描述

```text
container              safetensors | gguf | converted_native
scheme                 none | weight_only | weight_activation
storage_bits           16 | 8 | 4 | ...
signedness
group_size
quant_axis
scale_dtype
zero_point_mode
packing_order
activation_dtype
accumulation_dtype
kernel_layout_id
source_format_version
```

每个量化Tensor拥有Descriptor或引用模型级Descriptor并声明例外。Runtime Manifest同时
记录Checkpoint原始格式和实际Device Storage，避免“请求INT4，实际FP16”不透明。

## 4. 实施阶段

### Q0：识别与拒绝

- 读取Config、Safetensors Metadata/Index或GGUF Metadata，不从文件名猜格式；
- 建立Format Registry和版本化Descriptor Parser；
- 校验每个Tensor Shape、Packed Byte Count、Group整除和Scale/Zero Shape；
- 输出支持矩阵；未知字段、混合格式和缺少量化辅助Tensor明确Unsupported；
- Full-precision Qwen3路径保持不变，作为质量Reference。

### Q1：离线转换边界

- 如需转换，使用一次性Offline Tool产生`converted_native`和Conversion Manifest；
- Manifest包含源文件Hash、Converter版本、Descriptor、每TensorOffset/Hash和License；
- 转换可使用脚本工具，但长驻Server和请求热路径保持原生C++；
- Runtime重新验证Manifest与文件，不信任Converter成功退出码；
- 转换不改变Tokenizer、Config、Chat Template或模型Architecture语义。

### Q2：Weight-only INT8基线

- 先选择目标DTK已Probe的Datatype和GEMM API；
- 没有量化GEMM时，可实现Tiny Native Reference用于验证，但不能作为高性能完成标准；
- Scale应用、Accumulation和输出转换集中在Operator Adapter；
- Embedding、Norm和LM Head是否量化分别声明，不使用“一刀切”规则；
- 对每个Linear Layer比较反量化Reference和Quantized Kernel输出。

### Q3：Weight-only INT4

- 冻结Nibble Packing、Group Layout、Scale/Zero和Kernel Tile Layout；
- Converter Layout与Kernel Layout不同时显式Repack并记录；
- M/N/K尾部、非整Group和Alignment必须测试；
- 设备不支持有效INT4计算时保持Unsupported，不用逐Token全量反量化冒充支持；
- Qwen3 Gate/Up、QKV和LM Head分别评估精度与收益。

### Q4：Activation量化和混合精度

- 只有Weight-only稳定后再进入；
- 明确Static/Dynamic Scale、Calibration数据、样本数和版本；
- FP8或其他格式必须同时通过Storage、Compute和Accumulation Probe；
- 每层输出Requantization和Residual DType写入Model Contract扩展；
- 不把CUDA可用的FP8路径推导为DTK支持。

## 5. 正确性和质量验收

格式级：

- Tiny人工Packed Fixture逐元素解码；
- Offset、Truncated File、错误Group、缺Scale、Endian和Overflow拒绝；
- Offline Conversion重复执行得到相同Manifest和Tensor Hash；
- Loader发布失败时不留下部分Device Weights。

数值级：

- 每类Linear对比Full-precision、显式Dequant Reference和Quantized Kernel；
- 单Layer、Prefill Logits、Decode Logits和完整Token逐级定位误差；
- 使用固定文本集合报告Logit Error、Token Agreement和适当的语言质量指标；
- Greedy偶然输出一致不能代替中间数值和统计质量验证；
- Sampling分布在量化后没有不可解释的塌缩。

## 6. 性能验收

报告磁盘字节、Host峰值、Device Weight字节、Load Time、Prefill/Decode吞吐、Kernel时间和
临时Dequant字节。必须区分：

```text
storage compression only
load-time dequantization
on-the-fly dequantization
native quantized GEMM
```

只有Device Storage和Kernel路径都与Descriptor一致时才宣称量化推理。质量阈值由任务
需求冻结，不能为了通过性能门槛临时放宽。

## 7. 风险与停止条件

- GGUF量化类型众多，优先支持明确请求且有DTK Kernel证据的子集。
- 上游格式可能演进，Source Version和Converter ABI必须固定。
- 量化节省显存但可能降低小Batch吞吐，结果按模型和Shape报告。
- 缺少真实量化Checkpoint或Reference输出时状态为blocked，不生成伪Fixture宣称完成。
