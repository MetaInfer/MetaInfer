## 1. 当前工作思路：
### 从 ref_projects 参考工程下抽取知识存入到notebooks-cn中，根据需要和问题让cursor agent读取知识并编码，实现自研框架
### 当自研框架完成某一阶段功能，将当前状态下框架已使用的知识整理更新到 curosr/skills/inference_blueprint.json 中，里面包含每个组件功能定义、参考知识路径和接口约束；更新总的agent工作约束 curosr/skills/AGENT_SKILL.md。 利用这两个文档和一句需求prompt，让agent一次生成满足需求的框架。

## 1.5. Skill 实验方法论

通过 4 个 skill 形成推理引擎开发的标准流程，每个 skill 独立可触发：

| 步骤 | Skill | 触发词 | 产出 |
|------|-------|--------|------|
| ① | `infer-baseline` | "建立基线" / "硬件环境" | 硬件检测 + experiment_baseline.md |
| ② | `infer-mlx-ref` | "MLX 基准" / "参考效率" | MLX-lm 标准推理指标 (100% 参考线) |
| ③ | `infer-engine-build` | "写引擎" / "70% 效率" | 自研引擎 Phase0→3 (目标 ≥70% baseline) |
| ④ | `infer-optimize-plan` | "优化方向" / "差距分析" | Tier 分级优化路线图 |

流程链: `infer-baseline` → `infer-mlx-ref` → `infer-engine-build` → `infer-optimize-plan`
（另含 `infer-bench` 多框架对抗对比，跨平台适用）

## 2.文件介绍
### engine 文件夹下为自研引擎的各组件，llm_engine.py为引擎入口，tests 和 .sh为测试脚本，openai_tp_server.py 用来挂起服务方便vllm bench统计自研框架指标
### notebooks-cn为不断更新的知识，.cursor为cursor subagents约束， commit-changes 是每次commit修改的文件方便查看

## 3. 当前进度
### 当前可以根据 curosr/skills 下两个文档和一句prompt生成最朴素推理框架（不含TP及各种优化措施，待再次更新）
### 当前 engine 自研框架可以实现qwen3、deepseekv2的多TP并行推理，但是吞吐率相对于vllm极低，原因是还未使用其它参考框架优化过的算子和一些优化措施。