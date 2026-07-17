# 改进计划：Paged KV与Prefix Cache

状态：proposed  
来源：旧任务Iteration 2/3；SRC-VLLM、SRC-TRT、SRC-SGLANG。  
前置Contract：`00_contracts/attention_kv_contracts.md`、
`00_contracts/memory_contracts.md`、`01_framework_design/03_kv_cache.md`。

## 1. 当前证据

旧任务曾按照Qwen3训练最大长度40960一次性分配约4.38 GiB KV，并在权重加载后的
碎片化设备堆上OOM。后续把Serving上限降为4096并合并分配后，服务才能启动。这个
修复解决了单请求启动，但不等于具备Paged KV、多请求弹性分配或Prefix Reuse。

## 2. 目标和边界

第一目标是固定大小Block Pool和可事务回滚的Request Block Table；第二目标才是
Full-block Prefix Cache。二者必须分阶段验收，不能为了命中缓存破坏KV正确性。

非目标：第一版不做跨进程KV Transfer、PD Disaggregation、Host Swap或跨租户默认
共享。连续KV基线保留为Tiny Reference，不进入生产热路径。

## 3. 基础Paged KV阶段

### K0：预算和布局冻结

- 从每Rank真实空闲显存扣除Local Weights、Workspace、Collective和Safety Margin；
- 用Layer、Local KV Head、Head Dim、DType和Block Token计算每Block字节；
- 把Layout ID、Block Size、总Block和预算写入Runtime Manifest；
- Overflow、Alignment和设备最大分配必须在任何`hipMalloc`之前检查。

### K1：Block Pool与事务

- 初始化时创建物理Storage和Block Metadata，不在Decode Step分配；
- Request使用Logical Position到Physical Block ID的Block Table；
- Reserve返回RAII Transaction，全部Layer/Rank成功后才能Commit；
- Cancel、OOM、Kernel错误和Rank失败必须Rollback且Free Count守恒；
- 使用Generation Counter检测释放后重用导致的Stale Handle。

### K2：Paged Attention接入

- Kernel输入显式携带Block Table、KV Length、Block Size、Layout和Stream；
- 新K/V只写当前Slot，Attention只读取已提交Token；
- Block边界、非整Block尾部和GQA Head映射与连续Reference比较；
- Prefill可以先用Gather Reference，但生产Decode禁止每步Gather完整KV。

### K3：Scheduler联动

- Admission按Prompt和保留Decode Token计算所需Block；
- Scheduler只使用Pool的原子资源快照，不自己维护第二份Free Count；
- Block不足时选择延后、拒绝或有证据的Preemption，不允许越界提交；
- 每个Step记录Reserved、Committed、Freed和Cached Block变化。

## 4. Prefix Cache阶段

SRC-VLLM表明成熟Prefix Cache至少需要Full Block Hash、Parent Hash、Block Tokens、
Reference Count、Free/Eviction Queue和租户隔离。C++计划采用以下Canonical Key：

```text
model_revision
tokenizer_revision
adapter_or_lora_id
kv_dtype_and_layout_id
parent_block_hash
exact_block_token_ids
tenant_cache_salt
```

只有Full Block进入Cache；Partial Tail属于请求。Hash碰撞必须通过完整Key比较或安全
Hash策略处理。多租户默认不共享，只有相同Salt/Trust Domain才能复用。

### P0：只读命中

- 先实现已完成Block的Hash、Lookup、Ref Count和命中指标；
- Cache命中后Block进入In-use状态，不能同时被Evict；
- 新请求未命中部分正常Reserve，不修改已命中的Block；
- 关闭Prefix Cache时输出与开启但零命中时完全一致。

### P1：Eviction和重复Block

- 维护O(1) Free/Eviction队列，策略和状态转换有单Owner；
- 释放请求时先减少Ref Count，再决定进入可Evict队列；
- 允许短期重复Hash Block，但必须有确定的去重或最终回收规则；
- Cache条目失效和物理Block释放是一个事务，禁止悬空映射。

### P2：调度收益

- Scheduler用Cached Token数减少Prefill预算，但仍验证Block可用；
- 命中只跳过已证明等价的Prefix Compute，不跳过Chat Template和Tokenizer验证；
- 收集Lookup、Hash、Hit Token、Saved Prefill和Eviction指标；
- 只有重复Prompt负载证明收益后才默认启用。

## 5. 本地验收

正确性：

- Block Size 1、非整尾部、跨多个Block和最大Context边界；
- Reserve中第N个Layer失败后所有Free Count和Mapping恢复；
- 两请求共享Prefix后取消任一请求不会释放另一请求仍引用的Block；
- 不同Model Revision、Tokenizer、Adapter、KV DType和Salt永不误命中；
- Cache Eviction后Stale Generation访问被拒绝；
- 连续Reference、Paged无缓存、Paged有缓存三条路径Logits在容差内一致；
- 一千轮分配、取消、复用后无泄漏或Metadata增长。

性能：报告KV有效利用率、内部碎片、命中Token比例、Prefill节省、Hash CPU时间和
Paged Kernel GPU Event。只报告“缓存命中率”不能证明端到端收益。

## 6. 风险与停止条件

- Block太小增加Table和Kernel开销，太大增加内部碎片，必须通过Sweep选择。
- Prefix Cache引入隐私侧信道，未实现Salt和权限策略前保持默认关闭。
- DTK Kernel若无法高效跨Block读取，先保留正确Paged Baseline再优化访问，不回退为
  无界连续分配。
- Hybrid/Sliding-window模型需要独立KV Group策略，不能套用Dense Full Attention Key。
