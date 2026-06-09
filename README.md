# MetaInfer

> **克隆即用**：`git clone` master 分支 → Claude Code 打开本目录 → Agent 按知识图谱自主生成推理引擎。
>
> 本目录是一个自包含的**知识包**（prior knowledge package），包含领域知识图谱 `inference_blueprint.json`、多智能体执行 SOP `AGENT_SKILL.md`、28 个不可变测试合约 `scripts/`。用 Claude Code 打开后，主 Agent 会自动读取这三层先验知识，驱动三角色对抗式子代理（implementer → spec-reviewer → verification），按 11 个 Phase 管线自下而上生成适用于目标模型与硬件配置的**单路径推理框架**（约 4,500 行 Python）。
>
> **启动时只需提供两个路径**：模型权重目录 `MODEL_DIR` 和 Python 环境 `PYTHON_PATH`（含 torch、flash_attn、vLLM 的 conda/venv）。Agent 会先验证环境，再根据模型 `config.json` 自动路由架构（Dense / MLA+MoE(更新中)），然后逐 Phase 生成代码并通过 scripts/ 门禁验收。详见下文技术文档。

# MetaInfer 技术文档

> 本文档为 MetaInfer 项目的技术报告，系统记录从问题定义、系统设计、性能评估到未来规划的全过程。
>
> 

# **摘要**

现有的开源大语言模型（LLM）推理框架（如 vLLM、SGLang）为保持通用性，常在单一静态代码库中引入大量适配层：涵盖数十种模型架构、多种并行策略组合、多种量化后端、多硬件平台适配以及数千个环境变量开关。这种“大而全”的软件分发模式带来了沉重的维护负担，且在实际推理任务中，一旦模型与机器环境确定，推理服务仅需一种特定组合——通用框架中的冗余抽象和动态派发反而限制了推理效率与快速迭代能力。大语言模型在代码生成、内核优化及 Verilog 生成等领域已展现了强大潜力，然而在推理框架本身的自动化构建领域仍缺少实质性进展。为应对上述挑战，本文提出下一代生成式推理框架——MetaInfer。该设计摒弃传统的通用静态软件分发模式，创新性地引入“**大模型即编译器**”（LLM\-as\-Compiler）的新范式。在系统设计上，MetaInfer 提取大模型推理所需的核心算子接口、并行策略模式、调度状态机与内存管理规约，构建为硬件感知的领域知识图谱；当用户以自然语言或规范文档声明其特定场景的优化约束（目标模型、并行策略、硬件平台、性能指标），MetaInfer 即可通过规范驱动的多智能体协同，动态“编译”并组装出单路径的定制推理引擎，并根据每次定制的结果自动反馈并补充知识图谱，形成闭环。

作为 MetaInfer 范式的原型验证，我们首先通过人工方式在 Nvidia A800 上针对 Qwen3\-8B（Dense）构建了一个精简原型推理引擎 meta0，积累了从 TP 切分、kernel 替换到通信优化的完整工程经验。*随后，我们将*这些经验抽象为结构化的领域知识图谱与标准操作流程，设计了**实施者/规范审查者/验收者**（implementer / spec\-reviewer / verification） 三角色对抗式多智能体协同架构。在此基础上，Agent 在完全不接触 meta0 源码的白板条件下，自主生成了功能等价的新推理框架。与通用推理框架 vLLM 同等关闭 CUDA Graph 的对比表明，生成引擎：

> **代码量缩减约 99%**：核心推理引擎仅约 4,500 行 Python 代码，vLLM 相应模块数万行；
> 
> **端到端数值完全对齐**：相同输入、相同采样参数（temperature=0）下输出一致；
> 
> 

当前，MetaInfer 的全部 11 个 Phase已通过多智能体串行审查验收，Agent 在白板条件下自主生成的推理框架实现了单卡与四卡张量并行推理输出字字对齐，证明了”按需生成式推理框架”在 AI 基础设施领域取代巨型静态软件系统的可行性。在 Qwen3\-8B, TP=4, Batch=1 配置下，与vLLM eager 43\.9 tok/s相比，生成引擎 MetaInfer 初始吞吐率较低为 18\.8 tok/s，但仅通过agent对cpu侧进行简单的cpu开销修复得到的 MetaInfer\-Optimized，吞吐率最高可达到 62\.0 tok/s，vLLM eager 相比提升 41\.23 %，证明了”按需生成式推理框架”在 AI 基础设施领域取代巨型静态软件系统的可行性。

# **2 MetaInfer 系统设计概览**

## **2\.1 核心范式：LLM\-as\-Compiler**

传统推理框架以“软件库”形态分发——开发者下载一个包含所有模型、所有平台、所有优化策略的巨型代码库，再通过数千个环境变量和命令行参数“裁剪”出自己需要的路径。MetaInfer 修改了这一范式：

> 传统模式：  通用代码库 → 裁剪/配置 → 特定推理引擎（大量死代码残留）
> 
> MetaInfer：  领域知识图谱 \+ 用户约束 → 多智能体编译 → 单路径推理引擎（零冗余）
> 
> 

用户无需阅读大型框架的源码，只需以自然语言或规范文档声明其场景约束（例如：“Qwen3\-8B Dense，TP=4，A800，纯 eager 模式”），MetaInfer 即可从知识图谱中检索出该场景所需的全部契约，驱动多智能体系统生成仅包含目标路径的推理代码。

## 2\.2 **三层系统架构**

```Python
┌──────────────────────────────────────────────────────┐
│                    用户约束                           │
│  模型架构 + 并行策略 + 硬件平台 + 性能目标              │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│              第一层：领域知识图谱                       │
│  ┌────────────┬────────────┬────────────┬──────────┐ │
│  │ kernel     │ TP 层接口   │ 模型维度    │ 调度模式  │ │
│  │ 契约       │ 契约        │ 参数        │ 状态机    │ │
│  ├────────────┼────────────┼────────────┼──────────┤ │
│  │ 权重映射   │ 内存管理    │ 采样协议    │ 通信原语  │ │
│  └────────────┴────────────┴────────────┴──────────┘ │
│              inference_blueprint.json                 │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│           第二层：多智能体协同 SOP                     │
│                                                      │
│  AGENT_SKILL.md  +  CLAUDE.md                        │
│                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐  │
│  │implementer│ → │spec-reviewer │   →│verification│  │
│  │ 写代码    │    │ 蓝图契约核验  │    │ 测试+证据   │ │
│  │ (不跑测试)│    │ (Shell隔离)  │    │ (Shell隔离) │ │
│  └──────────┘    └──────────────┘    └────────────┘ │
│       ▲                                  │           │
│       └──────── 任一 ❌ 打回 ─────────────┘           │
│                                                      │
│  11 Phase 生成管线：数值基元 → TP通信 → 线性层 →        │
│  Embedding → Attention → Decoder → 权重加载 →         │
│  框架外壳 → 引擎集成 → E2E 验收 → 性能优化              │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│            第三层：生成的推理引擎                       │
│                                                      │
│  agent-infer/engine/   (~4,500 行)                   │
│  ┌──────────┬──────────┬──────────┬──────────────┐  │
│  │ models/  │tp_layers/│kernels/  │scheduler.py  │  │
│  │ qwen.py  │linear.py │vllm_     │block_        │  │
│  │ deepseek │embed.py  │wrappers  │manager.py    │  │
│  │ _v2.py   │custom_   │.py       │memory_       │  │
│  │          │ar.py     │          │pool.py       │  │
│  └──────────┴──────────┴──────────┴──────────────┘  │
│                                                      │
│  特性：单路径、无冗余分支、仅目标模型+目标并行策略       │
└──────────────────────────────────────────────────────┘
```

## **2\.3 与传统框架的本质区别**

|维度|传统框架（vLLM/SGLang）|MetaInfer|
|---|---|---|
|分发形态|静态代码库|知识图谱 \+ 生成 SOP|
|代码路径|多分支动态派发|编译期确定，无分支|
|适配新模型|手写数百行模型适配代码|知识图谱中补充模型维度参数|
|功能裁剪|环境变量 \+ 命令行参数|生成时确定，不生成无关代码|
|性能调优|全局开关影响所有路径|场景专属定制|
|可维护性|修改一处可能影响 N 个模型|每个引擎独立，回归范围可控|
|代码量|数十万行|单场景 \~4,500 行|

# **3 领域知识图谱的构建**

## **3\.1 知识来源与提取流程**

MetaInfer 的领域知识图谱（以下简称蓝图）并非凭空设计，而是从真实工程实践中逐步提取和验证的。构建过程分为三个阶段：

**阶段一：人工构建原型引擎（meta0）:**

为获得推理框架的一手工程经验，我们首先以人工方式在 Nvidia A800 上针对 Qwen3\-8B 构建了 TP=4 的精简推理引擎。整个构建过程分为三个子阶段，每个阶段使用了不同的 LLM API 作为编程助手：

|子阶段|内容|使用的 LLM API|
|---|---|---|
|**1\.1 最小可用框架**|Scheduler、BlockManager、KVMemoryPool、Sampler、Sequence、LLMEngine 等推理框架必备最小组件|**Cursor \+ Claude Opus 4\.6**|
|**1\.2 模型张量并行（TP）支持**|QKVColumnParallelLinear、RowParallelLinear、VocabParallelEmbedding、QwenAttentionTP、QwenDecoderLayerTP、所有权重切分和 HF key 映射|**Claude Code \+ Gemini 2\.5 Pro API**|
|**1\.3 推理性能优化**|7 个 vLLM 黑盒 kernel 替换（RMSNorm、RoPE、SiLU、FlashAttention）、P2P CustomAR 通信（8\.7× 加速）、KV Cache paged 格式、torch\.compile 集成|**Claude Code \+ DeepSeek V4 Pro API**|

**阶段二：经验抽象为结构化知识:**

从 meta0 的代码和文档中，提取了以下知识类别并填入\.json格式的知识图谱中：

|知识类别|JSON 路径|内容说明|
|---|---|---|
|Kernel 契约|qwen3\_kernel\_contracts|7 个 vLLM 黑盒 kernel 的签名、import 路径、调用前置条件|
|TP 通信契约|tp\_distributed\_runtime|3 种 collective 的 custom\_op 注册、fake 实现、IPC 初始化状态机|
|TP 层接口契约|tp\_linear\_layers|4 种 Linear 的 forward pseudocode、per\-rank 维度推导公式|
|模型维度参数|qwen3\_8b\_model\_dims|Qwen3\-8B 物理 config\.json 的精确值|
|权重映射表|qwen\_hf\_key\_mapping|12 个 HF key → 内部属性名的映射、Q\-K\-V/Gate\-Up 拼接规则|
|Attention 契约|qwen3\_tp\_model\_interfaces\.attention|paged KV cache 格式、block\_table/slot\_mapping、flash\_attn 调用|
|MLP/Decoder 契约|qwen3\_tp\_model\_interfaces\.mlp|gate\_up → silu\_and\_mul → down 完整数据流|
|调度器接口|components\[0\] Scheduler|schedule/postprocess 完整 pseudocode、REJECTED 机制|
|引擎组装链|scheduler\_tp\_runner\_bridge|block\_size 双轨注入、num\_free 来源路由、BlockManager 降级|
|失败模式库|failure\_mode\_library|双重切片、RoPE 风格错配、KV head 复制遗漏等高发故障|
|全局约束|global\_primitives\_constraints|RMSNorm 精度法则、fused\_add\_rms\_norm 跨层依赖禁令|

**阶段三：多智能体 SOP 设计：**

Agent根据搭建 meta0 时的经验，排出自下而上的工程构建顺序（数值基元 → 通信 → 层 → 模型 → 引擎集成），设计了执行铁律与三角色对抗式子代理协作协议（详见第四章）。

## **3\.2 知识图谱的核心设计原则**

1\. **契约优先**：所有实现必须受蓝图约束。Agent 禁止在未找到对应蓝图契约的情况下脑补实现。

2\. **三级引用链**：每个蓝图节点通过 \`ref\_docs\` 指向文档知识、通过 \`ref\_code\` 指向可执行参考开源框架源码行号。Agent 在生成代码前必须走完节点→ ref\_docs → ref\_code 三级知识链路。

3\. **伪代码自包含**：蓝图中的伪代码必须能在不依赖外部文档的前提下被Agent直接抄入。若伪代码不够完整，视为蓝图信息断裂。

4\. **维度参数从物理 config\.json 动态读取**：蓝图中的数值（如 \`max\_position\_embeddings=40960\`）标记为“示例值，禁止硬编码”，Agent 必须从实际模型文件动态获取。

## **3\.3 Human\-in\-the\-loop 迭代模式**

meta0 与知识图谱并非一次性完成，而是通过 **Human\-in\-the\-loop 迭代闭环**逐轮推进。每轮迭代遵循固定的五步流程（完整过程文件见附录 A）：

```Python
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 人类分析     │     │ Agent 制定  │     │ Agent 执行   │     │ 人类审查     │     │ 记录归档     │
│ Profiling   │ ──→ │ 修改计划     │ ──→ │ 代码修改     │ ──→ │ Benchmark   │ ──→ │ 提取知识     │
│ 提出方向     │     │             │     │             │     │ 对比         │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**人类的角色：**

（1）方向制定者：通过torch\.profiler分析瓶颈，指定优化目标

（2）质量审查者：对比修改前后 Benchmark——达标则通过，不达标则提供反馈重新迭代

（3）知识裁判：决定哪些经验值得入库，避免噪声污染知识图谱

**Agent 的角色**：

（1）读取现有代码和文档，生成修改计划

（2）执行代码修改，输出 Benchmark 数据

（3）整理每次迭代的产物（计划、算子契约、踩坑记录、Benchmark 对比）

冷启动的完整过程文件清单、每轮迭代摘要见附录 A。

## 3\.4 知识图谱验证机制

知识图谱本身作为 Agent 的最重要知识来源，其自身的**正确性**（图谱契约是否与物理工程实现一致）和**完备性**（图谱是否足以在 meta0 销毁假设下指导 Agent 重建引擎）同样需要质量保障。

为此，我们在知识图谱构建中引入了两类独立审计子代理（二者通过 Shell claude \-p 实现 PID 隔离），对图谱进行系统性审查：

**正确性审计**（Trinity Audit）：以独立三方审计官身份，对每个蓝图节点执行 ref\_docs → ref\_code → impl\_code 三方交叉印证，产出 Passed / Critical / Warning 三级审计报告。v1\-v5 共五轮审计发现并修复 3 个 CRITICAL 违规，将 FM 覆盖率从 40% 提升至 100%。

**完备性审计**（Blind Reconstructability Audit）：假设 meta0 源码已物理销毁，仅凭蓝图 \+ 参考文档 \+ 参考工程尝试逐组件重建，当蓝图信息不足以支撑实现决策时记录为 Fatal Gap \(FG\) 或 Override Warning \(OW\)。v1\-v17 共十七轮盲测审计发现并闭合全部 FG。

两个审计子代理的输出随后由构建 meta0 的主agent接管，执行物理 Tracing 采集 → 蓝图修复 → AGENT\_SKILL\.md 同步 → 再审计的闭环迭代，确保被发现的缺陷追溯至蓝图根因节点并回灌修复。两种审计互补：正确性审计确保”图谱说的是真话”，完备性审计确保”图谱说全了该说的话”。

# 4 **多智能体协同生成流程**

## **4\.1 为什么需要对抗式多智能体**

单 Agent 在长距离代码生成任务中面临三个致命问题：

1\. **确认偏误**：Agent 在生成代码时形成的错误假设，在自检时会被同样“合理化”。

2\. **上下文遗忘**：长对话中 Agent 会丢失蓝图早期约束。

3\. **测试盲区**：Agent 自写的测试倾向于验证“我以为我做了什么”而非“我应该做什么”。

MetaInfer 的解决方案是将代码生成拆分为三个物理隔离的角色，彼此互不信任。此设计灵感来源于**Superpowers**（https://github\.com/obra/superpowers）的子代理对抗协作模式——Superpowers 中 \`subagent\-driven\-development\` 技能定义了 implementer / spec\-reviewer / verification 三角色架构，并强制执行 "PRs with no evidence of human involvement will be closed" 的硬性质量门禁。MetaInfer 将这一模式适配到推理框架生成场景：spec\-reviewer 和 verification 通过 Shell \`claude \-p\` 启动独立 OS 进程（新 PID、无父进程记忆），而 implementer 通过 Agent 工具 spawn（需要完整工具链读写文件和查阅外部源码）。具体挂载方式如下：

|implementer         <br>\(Agent 工具\)         |spec\-reviewer <br>\(Shell claude \-p\)|verification<br>\(Shell claude \-p\)|
|---|---|---|
|只写代码  <br>不跑测试  <br>自读 diff <br>提交状态=SUBMITTED |独立读代码<br>逐条对照蓝图契约<br>不读 implementer 报告<br>输出 SPEC\_REVIEW\_REPORT|只跑测试<br>不读实现者/审查者输出<br>不读实现者/审查者输出<br>输出 VERIFICATION\_REPORT|

## **4\.2 三角色串行审查协议**

（1）implementer（生成代码）：

主 Agent 使用 Agent 工具 spawn implementer 子代理。子代理读取蓝图中当前 Phase 的契约节点 → 按 AGENT\_SKILL\.md §2\.0\.1 三步知识链路打开 ref\_docs 和 ref\_code → 生成代码。实现者只写代码、自读 diff，不运行任何 scripts/ 测试。返回状态始终为 SUBMITTED，绝不自行宣判 PASS。

（2）spec\-reviewer（契约核验）：

主 Agent 通过 Shell \`claude \-p\` 启动真正进程隔离的审查者。spec\-reviewer 不读 implementer 的任何输出——只读代码文件本身，逐条对照蓝图中的契约。审查结果写入 \`SPEC\_REVIEW\_REPORT\.md\`。任一契约违反 →  FAIL → 直接打回 implementer，verification 不启动。

（3）verification（测试与证据）：

仅当 spec\-reviewer PASS 后才启动。verification 同样通过 Shell 隔离，只跑 scripts/ 的固定测试脚本 \+ profiler \+ HCU 监控。不读任何其他子代理的报告。全部通过则 Phase 交付，任一 FAIL 则打回 implementer。

## **4\.3 当前多Phase管线生成情况**

管线按物理依赖排列，不可重排：

|Phase|内容|scripts/ 门禁|当前状态|
|---|---|---|---|
|Phase 1|数值基元（kernel wrapper）|2 个脚本|✅ Agent 可生成|
|Phase 2|TP 通信（CustomAR \+ all\_reduce）|2 个脚本|✅ Agent 可生成|
|Phase 3|TP 线性层（QKV/Column/Row）|2 个脚本|✅ Agent 可生成|
|Phase 4|TP Embedding \+ LM Head|2 个脚本|✅ Agent 可生成|
|Phase 5|Attention \+ KV Cache|3 个脚本|✅ Agent 可生成|
|Phase 6|MLP \+ Decoder Layer|4 个脚本|✅ Agent 可生成|
|Phase 7|权重加载|3 个脚本|✅ Agent 可生成|
|Phase 8|框架外壳（Scheduler \+ BlockManager）|2 个脚本|✅ Agent 可生成|
|Phase 9|引擎集成（LLMEngine \+ ModelRunner）|2 个脚本|✅ Agent 可生成|
|Phase 10|E2E 验收（对齐 \+ Benchmark \+ Profiler）|4 个脚本|✅ Agent 可生成|
|Phase 11|性能优化（pre\-alloc \+ view \+ contiguous 消除）|2 个脚本|✅ Agent 可生成|

## **4\.4 防退化机制**

**（1）scripts不可变**：测试脚本是先验知识，Agent 无权修改。测试不过 → 改代码，不降测试。

（2）**逐 Phase 门禁**：每个 Phase 的全部 scripts/ 必须 PASS 后才能进入下一 Phase。禁止跳过。

（3）**L0 防假 PASS**：verification 在运行任何测试前，必须先验证 import 的代码源路径——确认 engine 模块来自本地目录而非外部泄漏（如 PYTHONPATH 漏入真实 meta0 路径）。此项不通过 → 整个验收直接 FAIL。

（4）**步骤 3\.5 抽查**：verification 报告全部 PASS 后，主 Agent 随机抽取 1 个 Phase 脚本亲自重跑，比对 verification 报告中的原始 stdout。输出不一致 → 整个验收作废，重新 spawn verification。

（5）**跨 Phase 回归**：Phase 3\+ 的 verification 必须重跑所有前序 Phase 的 scripts/。任一回滚 → 打回。

（6）**E2E 证据链**：Phase 10 验收不仅要求输出正确，还必须附带 profiler trace（确认无 torch\.compile/CUDA Graph 痕迹）\+ HCU/VRAM 监控（确认 4 卡真实并行）。

（7）**反假推理**：严禁硬编码文本冒充真实推理输出。必须有 profiler trace 文件 \+ HCU 监控记录。

# **5 生成引擎的性能验证**

> **环境**: Qwen3\-8B, TP=4 \(4× NVIDIA A800\), prompt="苏州园林的特点是", 12 output tokens, temperature=0
> 
> **MetaInfer**: agent经过11 phase自动生成。
> 
> **MetaInfer\-Optimized**: MetaInfer 由agent自动优化, 消除部分cpu事件开销，CustomAR P2P 通信, 
> 
> **vLLM Eager**: vLLM 0\.15\.1 \`enforce\_eager=True\` — 无 CUDA Graph, 有 torch\.compile, NCCL ring 通信
> 
> **vLLM CUDA\-Graph**: vLLM 0\.15\.1 默认模式 — 有 CUDA Graph \(VllmBackend FULL\_AND\_PIECEWISE\), NCCL ring 通信
> 
> **Profiler 设置**: MetaInfer 使用 \`with\_stack=False\` \(避免 \`torch\.library\.custom\_op\` 三层 profiler 事件放大 CPU overhead\)
> 
> 

## 5\.1 MetaInfer 的简单优化

针对agent自动生成引擎吞吐率偏低情况，设计了一个 SKILL，通过比对 MetaInfer 与 Meta0 的 tracing ，通过多轮 loop 让 agent 自动优化并总结措施。实际操作中，MetaInfer 仅通过一轮简单的 Python 侧 CPU 调用优化（\+16/\-8 行）即得到 MetaInfer\-Optimized，具体修改如下：

|**修改**|**省掉的 CPU 开销**|
|---|---|
|@torch\.inference\_mode\(\)|autograd 元数据追踪、version counter 递增、tensor 生命周期管理|
|rms\_norm kernel 直接调用 \+ 预分配 buffer|empty\_like 触发的 cudaDeviceGetAttribute \+ nn\.Module\.call Python 调用链|
|MLP\_silu\_out 预分配|同上，empty\_like 的隐式 GPU 运行时查询|
|prefill KV 直接索引赋值|index\_copy\_ 的 dispatch 开销（decode 保留 index\_copy\_ 是为了 CUDA graph 兼容）|
|\_block\_table torch\.arange 初始化|per\-step 动态 arange \+ 索引填充|
|get\_num\_free\_blocks \(\) 去 \.item \(\)|GPU→CPU 同步等待（每次 decode step 一个隐式 cudaStreamSynchronize）|
|decode s\.kv\_len \+= 1 去 \.item \(\)|同上|

优化前后的gpu profiling:

|指标|MetaInfer|MetaInfer\-Optimized|加速比|说明|
|---|---|---|---|---|
|Self CUDA|207\.1ms|99\.4ms|**2\.08×**|GPU 时间减半|
|CustomAR|135\.8ms|29\.3ms|**4\.63×**|通信核心优化|
|Compute kernels|68\.8ms |68\.9ms|**1\.00×**|完全一致|

**零 GPU kernel 替换：**Compute kernel在多轮测试中完全一致，差异仅在 CustomAR 通信时间。MetaInfer 的 \`\.item\(\)\` GPU→CPU 同步点打断了 4\-GPU 流水线，导致 CustomAR kernel 在 NCCL barrier 处等待其他 GPU导致整体耗时加长——profiler 将此 wait 时间计入 CustomAR。

## 5\.2 **GPU 时间分解（四配置对比）**

### **5\.2\.1 Prefill \+ Decode 综合**

|类别|MetaInfer|MetaInfer\-Optimized|vLLM Eager|vLLM CUDA\-Graph|
|---|---|---|---|---|
|Compute \(GEMM/FA/norm/rope/silu\)|68\.8ms|68\.9ms|\~70ms|~~\~\~入图\~\~~~|
|Comm \(AllReduce\)|135\.8ms \(CustomAR P2P\)|29\.3ms \(CustomAR P2P\)|333\.4ms \(NCCL ring\)|19\.4ms\*\* \(NCCL in graph\)|
|Self CUDA \(profiler 去重汇总\)|207\.1ms|99\.4ms|390\.8ms|71\.9ms|
|Self CPU|497\.7ms|365\.0ms|453\.5ms|79\.7ms|
|Clean wall \(无 profiler\)|330\.7ms|239\.4ms|275\.3ms|69\.5ms|
|Throughput|36\.3 tok/s|50\.1 tok/s|43\.6 tok/s|172\.7 tok/s|

> **Self CUDA 说明**: MetaInfer profiler 的 \`self\_device\_time\_total\` 是去重后的纯 GPU kernel 时间。MetaInfer 的 Self CUDA \(207\.1ms\) ≈ Compute \(68\.8ms\) \+ Comm \(135\.8ms\) = 204\.6ms（差异在测量噪声范围）。MetaInfer\-Optimized 同理：99\.4ms ≈ 68\.9ms \+ 29\.3ms = 98\.2ms。vLLM 的 Self CUDA 包含 \`execute\_context\` \(torch\.compile/CUDA Graph 封装层\) 与通信 kernel 的嵌套时间。
> 
> **吞吐率说明**：本次 profiling 中 MetaInfer\-Optimized 吞吐率偏低，可能与GPU状态有关，但仍高于 vLLM Eager
> 
> 

### **5\.2\.2 CustomAR vs NCCL 通信对比**

|通信方式|耗时|调用次数|每次耗时|
|---|---|---|---|
|**CustomAR P2P  \(MetaInfer\-Optimized\)**|**29\.3ms**|876 |33\.4μs|
|**CustomAR P2P**\(MetaInfer\)|135\.8ms|876|155\.0μs|
|**NCCL ring**\(vLLM Eager\)|333\.4ms|876|380\.6μs|
|**NCCL in graph**\(vLLM CUDA\-Graph\)|**19\.4ms**|876|22\.1μs|

CustomAR vs NCCL ring: 29\.3ms vs 333\.4ms — CustomAR 快** 11\.4 **倍。这是 MetaInfer 即使 CPU dispatch 更高，wall time 仍快于 vLLM Eager 的根本原因。

CustomAR vs NCCL in graph: 29\.3ms vs 19\.4ms — NCCL 入图后反超 CustomAR 1\.5×。CUDA Graph 将 NCCL 通信也从 333ms 降至 19ms（融合进 execute\_context），说明 **CustomAR 仍有入图优化空间**。

MetaInfer CustomAR 不稳定性: 根因是 \`\.item\(\)\` sync 导致的 GPU 间随机去同步。

## 5\.2 **Top GPU Kernel 对比**

|**排名**|**MetaInfer**|**MetaInfer 对应 Self CUDA**|**MetaInfer\-Optimized**|**MetaInfer\-Optimized 对应 Self CUDA**|**vLLM Eager**|**vLLM Eager 对应 Self CUDA**|**vLLM CUDA\-Graph**|**vLLM CUDA\-Graph 对应 Self CUDA**|
|---|---|---|---|---|---|---|---|---|
|1|**\_C\_custom\_ar::all\_reduce**|137\.8ms|aten::mm \(GEMM total\)|37\.3ms|execute\_context\_0 \(compile wrapper\)|416\.2ms|execute\_context\_0 \(graph replay\)|63\.1ms|
|2|aten::mm \(GEMM total\)|37\.3ms|**\_C\_custom\_ar::all\_reduce**|31\.2ms|**NCCL AllReduce**|333\.4ms|**NCCL AllReduce**|19\.4ms|
|3|cutlass gemm\_relu|13\.8ms|cutlass gemm\_relu|13\.9ms|aten::mm \(GEMM total\)|37\.1ms|cutlass gemm\_relu|13\.3ms|
|4|rms\_norm|12\.8ms|rms\_norm|12\.8ms|cutlass gemm\_relu|14\.0ms|**flash\_fwd\_splitkv**|3\.4ms|
|5|rotary\_embedding|2\.8ms|rotary\_embedding|2\.8ms|**gemvx::kernel**|6\.3ms|**flash\_fwd\_splitkv\_combine**|2\.7ms|
|6|silu\_and\_mul|1\.6ms|silu\_and\_mul|1\.6ms|**\_vllm\_fa2\_C::varlen\_fwd**|4\.4ms|aten::mm \(prefill\)|2\.2ms|
|7|flash\_attn\_with\_kvcache|0\.5ms|flash\_attn\_with\_kvcache|0\.5ms|rms\_norm|3\.5ms \(fused\)|triton fused kernel \(rms\)|1\.7ms|
|8|—|—|—|—|rotary\_embedding|3\.2ms|**reshape\_and\_cache\_flash**|1\.4ms|

1）通信 kernel 是分水岭: MetaInfer / MetaInfer\-Optimized 使用 CustomAR P2P（\`\_C\_custom\_ar::all\_reduce\`），vLLM 使用 NCCL ring（\`ncclDevKernel\_AllReduce\_Sum\_bf16\_RING\_LL\`）。CustomAR 在干净 GPU 状态下达 29ms，NCCL ring 则需要 333ms——差距 11\.4×。

2）MetaInfer 的 flash\_attn \(0\.5ms\) 远低于 vLLM Eager \(4\.4ms\): 这是因为 vLLM 的 \`\_vllm\_fa2\_C::varlen\_fwd\` 包含 prefill 阶段的 variable\-length flash attention overhead（12 token prefill \+ 432 decode calls），而 MetaInfer 的 short\-prompt 场景下 prefill 开销极低。

## 5\.3 CPU 时间对比

### 5\.3\.1 ** CPU 总时间对比**

|指标|MetaInfer|MetaInfer\-Optimized|vLLM Eager|vLLM CUDA\-Graph|
|---|---|---|---|---|
|Self CPU|497\.7ms|365\.0ms|453\.5ms|**79\.7ms**|
|Self CUDA|207\.1ms|99\.4ms|390\.8ms|71\.9ms|
|Wall \(clean\)|330\.7ms|239\.4ms|275\.3ms|**69\.5ms**|

> vLLM CUDA\-Graph 的 Self CPU 仅 79\.7ms，而 MetaInfer 最优仍为 365ms——**4\.6×** 差距。CUDA Graph 将 6000\+ kernel launch 替换为 48 次 \`cudaGraphLaunch\`，kernel dispatch 几乎归零。这是 MetaInfer 下一步优化的核心方向。
> 
> 

## 5\.3 规模对比

|维度|AutoLLM 生成引擎|vLLM|精简倍数|
|---|---|---|---|
|核心推理引擎|\~3,550 行|几十万行|\~100x\+|
|模型支持|1 个（按需生成）|50\+ 个|—|
|并行策略|1 种（TP）|4 种（TP/PP/DP/EP）|—|
|量化后端|0（仅 bf16）|10\+ 种|—|
|调度器|87 行|数千行|\~20x\+|
|多模态|无|完整支持|—|

精简不是“功能缺失”，而是为确定场景消除不必要的抽象层。vLLM 为支持 50\+ 模型 × 4 并行策略 × 10\+ 量化后端 × 多硬件平台的乘积组合，必须引入层层抽象和动态派发。MetaInfer 生成的是该乘积中的一个确定点——所有抽象层在生成时就已经塌缩为直接的数据搬运和 kernel 调用。

# **6 跨平台适配性**

MetaInfer 的知识图谱驱动生成范式不应局限于单一硬件平台。为验证生成框架的跨平台迁移能力，我们在 Apple Silicon（M5 Pro, macOS）平台上进行了独立的适配实验——Agent 在不接触 NVIDIA 平台 meta0 源码、仅凭同一套知识图谱和 SOP 的条件下，自主生成了基于 MLX 后端的 mac\-engine 推理引擎。

## **6\.1 实验背景与目标**

> **硬件平台**：Apple M5 Pro（20 GPU cores, 48 GB 统一内存, \~273 GB/s 带宽），macOS 26\.4\.1  
> 
> **后端框架**：MLX 0\.31\.2 \+ mlx\-lm 0\.31\.3（Apple Silicon 上最具竞争力的推理框架）  
> 
> **测试模型**：Qwen3\-8B（bf16, safetensors），temperature=0\.0（greedy）  
> 
> **核心问题**：mac\-engine 自研引擎与 mlx\_lm 参考框架相比，在不同上下文长度下性能差距有多大？
> 
> 

Agent 在生成 mac\-engine 时，将蓝图中的 CUDA 专属 kernel（如 vLLM 黑盒 kernel、CustomAR P2P）替换为 MLX 等效算子（mx\.fast\.rms\_norm、mx\.fast\.scaled\_dot\_product\_attention 等），并基于 MLX lazy evaluation 特性重新组织了模型前向图。整个适配过程由 Agent 自主完成，Human 仅提供测试反馈。

## **6\.2 核心性能对比**

测试矩阵覆盖 8 个场景，两个维度：Prompt 长度（11/256/1021/2041 tokens）× 生成长度（256/1024/2048 tokens），采集 TTFT、TPOT、吞吐、内存 4 类指标。

### **6\.2\.1 Decode 性能（TPOT）：两框架持平**

|Prompt 长度|mlx\_lm TPOT|mac\-engine TPOT|差距|
|---|---|---|---|
|≤256 tokens|55\.8–57\.9ms|56\.3–57\.2ms|\<1\.5ms|
|1021 tokens|56\.5–56\.8ms|57\.4–59\.5ms|\~1–3ms|
|2041 tokens|57\.1ms|59\.6ms|\~2\.5ms|

两框架的 decode 性能在短 prompt 下几乎完全一致。长 prompt 下 mac\-engine 略慢 2–3ms，根因是 KV cache 动态增长的 realloc 开销（step=256），可通过预分配 KV cache 消除。

### **6\.2\.2 Prefill 性能（TTFT）：mac\-engine 全面领先**

|Prompt 长度|mlx\_lm TTFT|mac\-engine TTFT|加速比|
|---|---|---|---|
|11 tokens|246ms|107ms |**2\.3×**|
|256 tokens|314ms|167ms|**1\.9×**|
|1021 tokens|659ms |567ms|**1\.2×**|
|2041 tokens|1180ms|1164ms|**1\.0×**|

mac\-engine 在短/中 prompt 下 TTFT 快 1\.9–2\.3×，证明 Agent 生成的 prefill 路径更精简——去除了 mlx\_lm 额外的 tokenizer 预处理和 prompt template 开销。长 prompt 下计算主导，两框架持平。

### **6\.2\.3 端到端吞吐**

|条件|mac\-engine|mlx\_lm|mac/mlx|
|---|---|---|---|
|理想（p=11, g=256）|17\.4 tok/s|17\.0 tok/s|**1\.024×**|
|中 prompt（p=256）|17\.6 tok/s|17\.6 tok/s |1\.000×|
|长 prompt（p=1021 ）|16\.8 tok/s|17\.0 tok/s |0\.988×|
|最差（p=1021, g=1024）|16\.7 tok/s |17\.4 tok/s|0\.960×|

mac\-engine 在所有场景下的吞吐为 mlx\_lm 的 96–102%，平均 \~99%。

### **6\.2\.4 内存占用**

|框架|RSS|
|---|---|
|mlx\_lm|13\.2 GB|
|mac\-engine|13\.9 GB|

多出的 0\.7 GB 来自 Agent 生成的额外 Python 对象和 MLX graph cache。在 48GB 设备上影响可忽略。

## **6\.3 Decode 瓶颈微观分析**

基于 Agent 自带的逐算子纳秒级计时（`deep_profile.py`），单步 decode 的 55\.0ms 时间分配如下：

|组件|耗时|占比|说明|
|---|---|---|---|
|MLP 线性投影（gate/up/down × 36 层）|42\.1ms|**76\.5%**|绝对瓶颈|
|Attention 线性投影（q/k/v/o × 36 层）|10\.1ms|18\.4%|—|
|LM Head（4096→151936）|4\.5ms|8\.2%|—|
|RMSNorm（×73）|4\.4ms|8\.0%|—|
|其他（RoPE/SDPA/KV/残差）|3\.0ms|5\.5%|—|
|Python 开销（采样/tokenizer）|0\.4ms|0\.7% |—|

**带宽分析**：

|每步读取权重|耗时|
|---|---|
|实测有效带宽|**14\.5 GB**（bf16）|
|M5 Pro 带宽上限|**264 GB/s**|
|**带宽利用率：99\.1%**|**\~273 GB/s**（lm\_head 实测）——已达硬件极限|

两框架共用相同的 MLX GPU kernel，decode 时间差异仅来自 KV cache 管理和 Python 层调度。

## **6\.4 对抗性验证**

修复 5 个代码缺陷后，8 个对抗性测试全部通过：

|测试|结果|关键数据|
|---|---|---|
|确定性|✅|10 轮 byte 级完全一致|
|长序列衰减|✅|256→2048 衰减仅 0\.8%|
|长 prefill|✅|4K prompt 近线性增长|
|KV 边界|✅|所有层 offset 一致|
|KV 满载|✅|2105 tokens 正常|
|延迟抖动|✅|P99/P50 = 7\.3%|
|边界用例|✅|空 prompt / max\_tokens=1 / stream|
|Multi\-turn|✅|两轮 hash 一致|

## **6\.5 跨平台适配的核心发现**

（1）**知识图谱的硬件无关性得到验证**：同一套蓝图（\`inference\_blueprint\.json\`）和 SOP（\`AGENT\_SKILL\.md\`）在完全不同的硬件平台（NVIDIA A800 CUDA vs Apple M5 Pro MLX）上均能指导 Agent 生成功能正确的推理引擎，且性能达到参考框架的 99%。

（2）**平台专属 kernel 的自主替换**：Agent 在 MLX 平台上自动将 vLLM 黑盒 kernel（\`fused\_add\_rms\_norm\`、\`silu\_and\_mul\`、\`rotary\_embedding\` 等）替换为 MLX 等价实现，将 CustomAR P2P 替换为 MLX 的单进程内通信，证明知识图谱中的 kernel 契约足够抽象——描述“做什么”而非“怎么做”。

（3）**单平台限制是当前短板**：当前 MLX 后端不支持张量并行（MLX 无 NCCL/CustomAR 等效的跨进程通信原语），mac\-engine 仅支持单 GPU 推理。这使得 MLX 平台的绝对吞吐（\~17 tok/s）与 NVIDIA A800 TP=4（\~56 tok/s）无法直接对比，但单 GPU vs 单 GPU 的公平比较中两者在同等数量级。

（4）**跨平台迭代效率**：从 Agent 首次生成 mac\-engine 到吞吐追平 mlx\_lm，Human 仅参与提供 8 个对抗性测试用例和性能反馈，未手写任何 MLX 专属代码。验证了 MetaInfer“LLM\-as\-Compiler”范式在跨平台场景下的可行性。

## **6\.6 后续跨平台方向**

|方向|收益|优先级|说明|
|---|---|---|---|
|KV cache 预分配|\+2–3ms（长 prompt）|P0|消除与 mlx\_lm 的最后差距|
|Batched decode|\+50–200% aggregate|P1|Server 场景核心优化|
|投机解码（n\-gram）|\+10–20%|P2|零额外模型，MVP 可行|
|MLX TP 通信抽象|多 GPU 加速|P3|需探索 MLX distributed 或自定义通信方案|

附录D附有mac平台测试环境与复现方法

# 7 海光DCU 平台差异性适配



# **7 当前进展与未来规划**

|里程碑|状态|说明|
|---|---|---|
|原型引擎 meta0|✅ 完成|Qwen3\-8B TP=4 55\.7 tok/s，数值对齐|
|领域知识图谱|✅ 完成|inference\_blueprint\.json，含 11 大知识类别，FM 覆盖率 100%|
|多智能体 SOP|✅ 完成|AGENT\_SKILL\.md \+ CLAUDE\.md，串行审查 \+ L0 防假 PASS \+ PID 隔离|
|先验知识质量保障|✅ 完成|17 轮重构审计 \+ 28 scripts 对抗压测|
|Phase 1\-11 自动生成|✅ 全部验证|白板 Agent 自主生成完整推理框架，单卡与多卡并行推理字字对齐|
|生成过程自愈|✅ 完成|8 个 bug 发现 → 根因定位 → 蓝图回灌修复|

8个bug:

## 7\.1 ** 短期目标：生成框架吞吐对齐**

Phase 1\-11 已全部通过多智能体验收，Agent 生成的推理框架在 TP=4 下正确推理。当前生成框架的吞吐约 18\.8tok/s @ TP=4（vs meta0 55\.7 tok/s @ TP=4），瓶颈在于 Agent 生成的 Python 调度层效率。后续方向：

（1）**性能知识图谱强化**：将meta0的性能规则对应的更细粒度实现模式回灌到蓝图 \`性能优化\` 节点

（2）张量并行**通信效率**：Agent 生成的 CustomAR 索引 bug 已在 Phase 9 构建时修复，但整体 TP 通信调度仍有优化空间

## **7\.2 中期目标：CUDA Graph 生成管线**

第五章 Profiling 分析表明，CPU dispatch 是当前唯最大瓶颈。将 CUDA Graph 捕获和 torch\.compile 集成到知识图谱中：

|瓶颈|当前耗时|目标耗时|方案|
|---|---|---|---|
|GEMM dispatch|180ms|\~5ms|torch\.compile inductor 融合|
|通信 dispatch|69ms|\~5ms|CustomAR 入图 \+ sglang 切图方案|
|Kernel launch|44ms|\~1ms|CUDA Graph 单次 replay|
|Tensor 管理|72ms|\~5ms|inductor 内存规划|

预期完成后：wall time 80\-100ms，吞吐 **120\-150 tok/s**（接近 vLLM CUDA Graph 的 167 tok/s）。

## **7\.3 长期愿景：走向泛化：知识分层、可配置性与 Harness 约束**

当前 AutoLLM 的显著局限性在于：知识图谱高度耦合于 Qwen3\-8B TP=4 特定配置。这是第一轮 Human\-in\-the\-loop 迭代的自然产物——在缺乏多模型经验时，Agent 将已适配的单一配置反复强化。要实现”不同模型、不同并行策略、不同部署方式”的泛化，需要在以下三个维度进行结构性的知识改造。

### 7\.3\.1 知识分层架构

将单一扁平的知识图谱拆分为四层，使agent能根据模型描述文件自动组装对应层次知识。

> ```YAML
> Layer 0: 通用推理基础（所有 Transformer 模型共用）
>   ├── Scheduler、BlockManager、KV Cache 概念、Sampler、LLMEngine 调度循环
>   └── 与模型架构无关，不随模型变化
> 
> Layer 1: 模型家族层（Dense / MLA+MoE）
>   ├── Dense:  QKV 合并投影、GQA、RoPE Neox-style、SiLU Gate
>   ├── MLA:   q_a/kv_a replicated, q_b/kv_b/o sharded, RoPE GPT-J-style
>   └── MoE:   EP 路由、共享专家 vs 路由专家
> 
> Layer 2: 具体模型层（Qwen3-8B / DeepSeek-V2-Lite）
>   ├── 模型维度（hidden_size, intermediate_size, num_heads, head_dim）
>   ├── 特殊配置（Qwen3 的 q_norm/k_norm, DeepSeek 的 kv_lora_rank）
>   └── 当前蓝图的 `qwen3_8b_model_dims` 和 `qwen_hf_key_mapping` 属于此层
> 
> Layer 3: 部署配置层（TP=N, PP, EP）
>   ├── 所有 per-rank 计算参数化: per_rank = full_dim // tp_size
>   ├── 通信 collective 选择: CustomAR vs NCCL（取决于是否支持 P2P）
>   └── 当前蓝图大量硬编码 tp_size=4，需改为参数化模板
> ```
> 
> 

实现路径：Agent 在新模型接入时先读 config\.json → 确定模型家族 → 加载 Layer 0\+1 通用知识 → 提取 Layer 2 模型特定参数 → 用户指定 Layer 3 部署参数 → 组装生成。

### 7\.3\.2 标准化模型描述文件

为降低新模型接入的Human成本，定义类似config\.json的文件，包含架构路由、并行策略偏好、已验证的维度映射。新模型接入适配只需要提供此文件和权重。

```YAML
# model_spec.yaml
architecture: dense_gqa          # Dense with GQA
rope_style: neox                 # Qwen3-style, half-half rotate
num_attention_heads: 32
num_key_value_heads: 8
head_dim: 128
special_features:
  - qk_norm                     # Qwen3特有的 Q/K Norm
parallel_strategy:
  tp_size: 4
  communication: custom_ar_p2p  # 如果硬件支持 P2P
verified_dimensions:
  per_rank_qkv: [1536, 4096]
  per_rank_gate_up: [6144, 4096]
```

Human的介入从“精细化审查代码”降级为“提供标准化描述文件\+审查benchmark结果”。

### 7\.4\.3 Harness约束机制

Harness是固不变的代码约束层，约束Agent的生成行为，使其在“可自由发挥的生成代码”和“必须严格遵守的规则之间”保持平衡。现有的 srcipts / 28 个脚本是最基础的harness\-\- 确保功能性正确。未来需引入两类新的harness:

（1）编译器的 Shape Checker：端到端推理前，遍历所有 module 的forward签名，验证上下游的 shape / dtype/device等的一致性。不依赖GPU即可运行，拦截当前最高频发生的维度错误。

（2）Module Interface Checker：对关键类强制要求实现标准接口方法（如get\_input\_shape\(\),

get\_output\_shape\(\)），在测试脚本中自动串行验证。

Harness 不随模型变化——它是通用的验证基础设施，是所有生成引擎的”安检层”。

### 7\.4\.4 长期愿景：全自动闭环

**（1）多模型泛化**：将 DeepSeek\-V2（MLA \+ MoE）的知识图谱补充完整，验证 MetaInfer 跨模型架构的生成能力。

**（2）TTFT 优化**：prefill 路径引入 torch\.compile 融合，目标将 TTFT 从 43ms 降至 20ms 以下。

**（3）自愈闭环**：当生成引擎在 profiler 中出现性能回退时，Agent 自动定位瓶颈组件并重新生成。

**（4）按需扩展**：用户声明新场景（如“Qwen3\-MoE, TP=8, FP8 量化, H100”）→ MetaInfer 从知识图谱检索对应契约 → 生成新引擎，无需人工阅读框架源码。

**（5）社区知识图谱**：不同用户的定制场景自动反馈补充知识图谱，形成公共的推理框架知识基础设施。

# 附录

## A Human\-in\-loop 迭代**过程文件清单**

以下为 §3\.3 所述 Human\-in\-the\-loop 迭代过程中产生的核心文档。这些文档是知识提取的原材料，记录了从零到 55\.7 tok/s 的完整路径。

### A\.1 **核心过程文件**

原型引擎 meta0 构建的每个子阶段产生了结构化的过程文档，按阶段分布如下：

（1）**阶段 1 — 最小可用框架** \(\`notebooks\-cn/01\_framework\_design/\`\)：

|文件|内容|
|---|---|
|01\_architecture\.md|推理框架架构总览，组件职责与交互关系|
|02\_scheduler\.md|调度器设计：prefill/decode 调度策略、连续批处理|
|03\_kv\_cache\.md|KV Cache 概念与 paged attention 基础|
|04\_model\_runner\.md|模型运行器：HF 模型的加载与执行|
|05\_sampler\.md|采样器设计：temperature、top\_p、top\_k|
|06\_memory\_pool\.md|显存池管理：block 分配/回收、内存估算|
|07\_request\_lifecycle\.md|请求完整生命周期：创建→调度→执行→完成|

（2）**阶段 2 — 张量并行支持** \(\`notebooks\-cn/04\_parallel\_strategies/\`\)：

|文件|内容|
|---|---|
|01\_tensor\_parallel\.md|Tensor Parallel 通用原理：Column/Row Parallel、通信原语|
|02\_qwen\_dense\_tp\_implementation\_guide\.md|Qwen3 Dense 模型 TP=4 实现指南：QKV 切分、权重加载、属性命名|
|03\_tp4\_moe\_implementation\_guide\_deepseekv2\.md|DeepSeek\-V2 MoE 模型的 TP=4 实现（后续多模型扩展）|

**（3）子阶段 3 — 性能优化** \(\`notebooks\-cn/07\_improvementPlan/\`\)：

|文件|内容|
|---|---|
|tasks\.md|初始需求文档，定义构建目标：分析五个参考工程的共性，搭建最小可用推理框架|
|improvement\_plan\.md|全程迭代记录：P0（增量 KV Cache 8\.49 tok/s）→ P2（torch\.compile 12\.75 tok/s）→ P3（Flash Attention）→ P5（TP 通信优化 55\.7 tok/s），每阶段的方案、结果、踩坑|
|kernel\_replacement\_plan\.md|7 个 vLLM 黑盒 kernel 替换的完整契约提取：Snippet A\-F 代码模板，dtype/shape/import 约束|
|qwen3\_effective\_changes\.md|10 个关键改动点的实现方案、陷阱、验证方法|
|stage0\_2\_vs\_vllm\.md|meta\-infer vs vLLM \(eager/graph\) 三模式 GPU/CPU kernel 级 Profiling 对比|

### A\.2 **知识提取原则**

从过程文档中提取知识时遵循以下规则：

（1）只提取验证通过的经验（Benchmark 有提升或 Bug 已修复）

（2）已过时的方案（如 HF past\_key\_values 旧方案）标注为历史参考，不纳入正式契约

（3）硬编码的数值替换为物理 config\.json 的值，并标注"示例值，禁止硬编码"

## B **测试环境与复现方法**

```Bash
# 环境：meta conda (PyTorch 2.9.1, flash_attn 2.8.3, vLLM 0.15.1)
export PATH=/home/honglin/miniconda3/envs/meta/bin:$PATH

# MetaInfer Agent 生成引擎正确性测试（单卡）
cd /home/honglin/inference-agent-system
export PATH=/home/honglin/miniconda3/envs/meta/bin:$PATH
export PYTHONPATH="$(pwd):$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 python -c "
import os; os.environ['META_INFER_LOG_RANK0_ONLY']='1'; os.environ['META_INFER_CUDA_GRAPH']='0'
from llm_engine import LLMEngine; from pathlib import Path
engine = LLMEngine(model_dir=Path('/home/honglin/models/qwen/Qwen3-8B'), inference_backend='qwen_tp', max_num_seqs=4)
out = engine.generate('苏州园林的特点是', max_new_tokens=24, temperature=0.0)
print(f'OUTPUT: {out!r}')
# 预期: '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'
"

# Agent 生成引擎 TP=4 正确性测试
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$((29500+RANDOM%1000)) python -c "
import os; os.environ['META_INFER_LOG_RANK0_ONLY']='1'; os.environ['META_INFER_CUDA_GRAPH']='0'
from llm_engine import LLMEngine; from pathlib import Path
engine = LLMEngine(model_dir=Path('/home/honglin/models/qwen/Qwen3-8B'), inference_backend='qwen_tp', max_num_seqs=4)
out = engine.generate('苏州园林的特点是', max_new_tokens=24, temperature=0.0)
if int(os.environ.get('RANK','0')) == 0:
    expected = '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'
    print(f'TP=4 OUTPUT: {out!r}')
    print(f'MATCH: {out == expected}')
"

# 性能 benchmark
cd /home/honglin/MetaInfer
SKIP_VLLM=1 CUDA_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 ROUNDS=10 output_len=8 REQUEST_RATE=4 MAX_CONCURRENCY=1 \
  MODEL_DIR=/home/honglin/models/qwen/Qwen3-8B \
  bash run_compare_metainfer_vllm.sh qwen
```

## C **相关文档索引**

|文档|路径|
|---|---|
|项目任务定义|m/notebooks\-cn/tasks\.md|
|知识图谱|inference\-agent\-system/inference\_blueprint\.json|
|多智能体 SOP|inference\-agent\-system/AGENT\_SKILL\.md|
|子代理协作协议|inference\-agent\-system/CLAUDE\.md|
|Qwen Dense TP 实现指南|MetaInfer/notebooks\-cn/04\_parallel\_strategies/02\_qwen\_dense\_tp\_implementation\_guide\.md|
|Qwen3 全部有效改动追溯|MetaInfer/notebooks\-cn/07\_improvementPlan/qwen3\_effective\_changes\.md|
|Stage 0\-2 vs vLLM 详细对比|MetaInfer/notebooks\-cn/07\_improvementPlan/stage0\_2\_vs\_vllm\.md|
|完整优化计划|MetaInfer/notebooks\-cn/07\_improvementPlan/improvement\_plan\.md|
|Kernel 替换计划|MetaInfer/notebooks\-cn/07\_improvementPlan/kernel\_replacement\_plan\.md|
|Profiling Traces|MetaInfer/notebooks\-cn/07\_improvementPlan/traces/|
|28 个门禁脚本|inference\-agent\-system/scripts/|

## D Mac平台**测试环境与复现方法**

```Bash
# 芯片: Apple M5 Pro, 20 GPU cores, 48 GB
# OS: macOS 26.4.1, MLX 0.31.2, mlx-lm 0.31.3
# 模型: Qwen3-8B, bf16, safetensors
cd subprojects/mac-engine
python scripts/bench_one.py mlx_lm      # mlx_lm 全场景
python scripts/bench_one.py mac_engine   # mac-engine 全场景
python scripts/bench_one.py summary     # 打印对比表
```

数据文件：scripts/bench\_compare\_results\.json（原始 JSON）、docs/experiment\-optimization/deep\_profile\_report\.md（剖析数据）、scripts/adversarial\_test\.py（对抗性测试）。


## Citation

If you find this project useful, please consider citing:

```bibtex
@software{metainfer_2026,
  author  = {Honglin Wang and Xiaoning Huang and Mingheng Mi},
  title   = {MetaInfer: LLM-as-Compiler — A Generative Inference Framework Built End-to-End by an Autonomous Multi-Agent Loop},
  year    = {2026},
  url     = {https://github.com/MetaInfer/MetaInfer}
}
```





