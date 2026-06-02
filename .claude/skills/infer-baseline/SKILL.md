---
name: infer-baseline
description: >
  检测当前硬件环境（CPU/GPU/内存/OS），采集框架版本信息，创建结构化实验基线表文件
  (experiment_baseline.md)，为后续基准测试和引擎开发建立可追踪的实验起点。
  当用户说"建立基线"、"实验表"、"记录硬件环境"、"baseline setup"、"先看看硬件"、
  "记录环境"、"开始实验"时立即触发。
  适用于任何推理引擎开发项目的起步阶段。
---

# Infer-Baseline: 实验基线建立

在推理引擎开发项目中，第一步永远是搞清楚跑在什么硬件上、有什么工具可用。
本 skill 检测环境并生成结构化的实验基线表。

---

## 第一步：硬件环境检测

执行以下检测命令，收集全部硬件信息：

### Apple Silicon

```bash
# CPU
sysctl -n machdep.cpu.brand_string
# 内存
sysctl hw.memsize | awk '{printf "%.0f GB\n", $2/1024/1024/1024}'
# GPU
system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chip|Cores|Metal"
# OS
sw_vers
```

### NVIDIA

```bash
# CPU
lscpu | grep "Model name"
# GPU
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# 内存
free -h | grep Mem
# OS
uname -a
```

---

## 第二步：框架版本检测

```bash
python3 -c "
import mlx; print(f'mlx={mlx.__version__}')
import mlx_lm; print(f'mlx-lm={mlx_lm.__version__}')
" 2>/dev/null

python3 -c "
import torch; print(f'torch={torch.__version__}')
print(f'mps_available={torch.backends.mps.is_available()}')
" 2>/dev/null

python3 -c "
import transformers; print(f'transformers={transformers.__version__}')
import modelscope; print(f'modelscope={modelscope.__version__}')
" 2>/dev/null
```

---

## 第三步：模型盘点

扫描常见模型缓存位置，列出本地可用的模型：名称、大小、架构、路径。

```bash
find ~/.cache/huggingface/hub/ ~/.cache/modelscope/hub/models/ ~/models/ -maxdepth 4 -name "config.json" 2>/dev/null | while read f; do
    size=$(du -sh "$(dirname "$f")" 2>/dev/null | cut -f1)
    arch=$(python3 -c "import json; print(json.load(open('$f')).get('architectures',['?'])[0])" 2>/dev/null)
    echo "$(dirname "$f") | $size | $arch"
done
```

选择 1-2 个适合当前硬件的小模型（≤7B 参数）作为实验模型，标注在基线表中。

---

## 第四步：创建实验基线表

在子项目的 `docs/01_planning/experiment_baseline.md` 创建以下结构化文件：

```markdown
# 实验基线表

> 记录 <project-name> 推理引擎开发的实验过程与性能对比。

## 1. 硬件环境

| 项目 | 值 |
|------|-----|
| 芯片 | <detected> |
| GPU 核心 | <detected> |
| 统一内存 | <detected> |
| 操作系统 | <detected> |
| 深度学习框架 | <detected versions> |

## 2. 基准实验 (Baseline: mlx-lm)

### 2.1 单次推理

| 模型 | 输入长度 | 输出长度 | TTFT (ms) | TPOT (ms/tok) | 总耗时 (s) | 吞吐 (tok/s) | 内存 (GB) |
|------|---------|---------|-----------|---------------|-----------|-------------|-----------|
| <model> | - | - | - | - | - | - | - |

### 2.2 并发压测

| 模型 | 并发数 | 请求数 | 总吞吐 (tok/s) | Mean TTFT (ms) | Mean TPOT (ms/tok) | 内存 (GB) |
|------|--------|--------|---------------|----------------|-------------------|-----------|
| <model> | - | - | - | - | - | - |

## 3. 自研引擎对比 (目标: ≥70% baseline)

| 实验编号 | 日期 | 吞吐 (tok/s) | vs baseline | TTFT (ms) | TPOT (ms/tok) | 内存 (GB) | 正确性 | 备注 |
|---------|------|-------------|-------------|-----------|---------------|-----------|--------|------|
| - | - | - | - | - | - | - | - | - |

## 4. 正确性验证记录

| 验证编号 | 日期 | Golden 来源 | 测试用例数 | 通过数 | 状态 | 备注 |
|---------|------|-----------|----------|--------|------|------|
| V01 | - | mlx-lm v0.31.3 | 7 | - | ⬜ 待验证 | Golden outputs from infer-ref-bench |

## 5. 优化方向记录

| 方向 | 优先级 | 预期收益 | 状态 |
|------|--------|---------|------|
| - | - | - | - |
```

---

## 第五步：输出摘要

在终端输出简洁的硬件摘要，并告知基线表路径：

```
✅ 基线表已创建 → subprojects/<project>/docs/01_planning/experiment_baseline.md

硬件摘要:
- 芯片: Apple M5 Pro, 20-core GPU
- 内存: 48 GB 统一内存
- 系统: macOS 26.4.1
- 框架: mlx=0.31.2, mlx-lm=0.31.3
- 模型: Qwen2.5-0.5B (953MB)

下一步: /infer-ref-bench 跑推理基准测试
```

---

## 关键约束

1. **检测优先于手动填写**：所有硬件信息通过命令自动获取，不依赖用户记忆。
2. **完整记录**：即使某个框架未安装，也要记录（标记为"未安装"）。
3. **目录自适应**：如果子项目路径不存在，先创建目录结构，再写文件。
