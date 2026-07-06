# MetaInfer v3 — 多机 Worktree 部署指南

## 一、架构概述

```
/shared/metainferv3.git/          ← bare repo（共享存储或 NFS）
    │
    ├── 学习机器 1 ── git clone → /data/metainferv3-learn-1/
    ├── 学习机器 2 ── git clone → /data/metainferv3-learn-2/
    ├── 学习机器 3 ── git clone → /data/metainferv3-learn-3/
    │   └── 三机同时学 Qwen3.6 27B，谁先 L0-L3 通过谁赢
    │
    ├── 调优机器 A ── git clone → /data/metainferv3-tune-A/
    └── 调优机器 B ── git clone → /data/metainferv3-tune-B/
        └── git pull 拿到学习机写入的知识 → 无限循环优化
```

**核心原则**：
- **学习机**：写知识 (`notebooks-cn/`)，验证通过即停止。多机并行 = 竞速模式
- **调优机**：只读知识。无限循环优化。用户退出才停
- **Git 是通信通道**：学习机 push 知识 = 向所有机器广播"我成功了"

---

## 二、Bare Repo 初始化

在共享存储上执行（只做一次）：

```bash
cd /data/whl-test/metainferv3

# 若尚未初始化
git init
git add notebooks-cn/ .claude/ roles/ scripts/ AGENT_SKILL.md CLAUDE.md
git add evolution/EVOLUTION.md evolution/scripts/call-evo-agent.sh
git add master/MASTER.md master/scripts/call-sub-agent.sh
git add .claude/skills/ .claude/roles/
git add .gitignore PROJECT_FLOW.md PROJECT_FLOW_MERMAID.md PROJECT_FLOW_README.md DEPLOYMENT.md
git commit -m "v3 baseline: 3-loop architecture + 7 agent roles + parallel learning"

# 克隆为 bare repo
git clone --bare /data/whl-test/metainferv3 /shared/metainferv3.git
```

> **注意**：`evolution/state.json`、`evolution/strategies/`、`evolution/results/`、`master/state.json`、`master/strategies/`、`master/results/` 都在 `.gitignore` 中——它们是每台机器的本地状态，不共享。

---

## 三、并行学习竞速：三机同时学同一模型

### 3.1 场景

```
三台机器，同一个目标：让知识库学会独立生成 Qwen3.6 27B 推理框架

机器 1: Explorer 可能搜到论文 A，Implementer 采用方案 X ──┐
机器 2: Explorer 可能搜到论文 B，Implementer 采用方案 Y ──┤  竞速
机器 3: Explorer 可能搜到论文 C，Implementer 采用方案 Z ──┘

谁先跑到 L0-L3 全部通过 → commit 知识 → push
其余机器检测到 push → 停止
```

**为什么值得并行**：Explorer 的搜索结果和 Implementer 的实现路线是非确定性的。三台机器可能走三条不同的路，成功率 >> 单台。

### 3.2 初始化（每台机器执行相同步骤）

```bash
# ─── 步骤 1: 从 bare repo 克隆 ───
MACHINE_ID=1   # 每台机器改这个：1 / 2 / 3
git clone /shared/metainferv3.git /data/metainferv3-learn-${MACHINE_ID}
cd /data/metainferv3-learn-${MACHINE_ID}

# ─── 步骤 2: 配置本机环境 ───
cat > .env_agent_infer << 'ENVEOF'
export AGENT_INFER_ROOT="/data/metainferv3-learn-MACHINE_ID_PLACEHOLDER"
export PYTHON_PATH="/opt/conda/envs/meta/bin"
export MODEL_DIR="/data/models/Qwen3.6-27B"
export PATH="${PYTHON_PATH}:$PATH"
export PYTHONPATH="${AGENT_INFER_ROOT}:$PYTHONPATH"
ENVEOF
sed -i "s|MACHINE_ID_PLACEHOLDER|${MACHINE_ID}|g" .env_agent_infer

# ─── 步骤 3: 开源代码缓存（symlink） ───
ln -s /path/to/your/ref_projects/vllm knowledge/vllm
ln -s /path/to/your/ref_projects/sglang knowledge/sglang

# ─── 步骤 4: 创建本机进化状态（gitignored，不共享） ───
mkdir -p evolution/strategies evolution/results
cat > evolution/state.json << 'EOF'
{
  "target_model": "Qwen3.6-27B",
  "evo_round": 0,
  "open_source_switch": false,
  "phase": "attempt_without_opensource",
  "entry_reason": "coverage_fail",
  "stage": "evolution",
  "history": [],
  "knowledge_snapshot": null,
  "consecutive_failures_with_opensource": 0,
  "diagnosis_notes": "竞速机器 MACHINE_ID_PLACEHOLDER。与其他机器并行学习 Qwen3.6 27B。先到 L0-L3 者胜。"
}
EOF
sed -i "s|MACHINE_ID_PLACEHOLDER|${MACHINE_ID}|g" evolution/state.json

# ─── 步骤 5: 确认 git 状态 ───
git status
# 应该只看到 untracked: .env_agent_infer, evolution/state.json（都是 gitignored）
# 关键：evolution/strategies/ 和 evolution/results/ 目录为空（gitignored）
```

### 3.3 启动学习

每台机器上执行（三台可以同时启动）：

```bash
cd /data/metainferv3-learn-1   # 机器 2 用 learn-2，机器 3 用 learn-3
source .env_agent_infer

claude -p "
你是 MetaInfer 主 Agent，按 CLAUDE.md 和 PROJECT_FLOW_MERMAID.md 的流程图执行。

当前是学习机器，目标模型 Qwen3.6 27B。
CLAUDE.md Step 6 会判定 KB 未覆盖 → 自动路由到 evolution/EVOLUTION.md。
evolution/state.json 已配置 entry_reason=coverage_fail → 跳过无开源尝试，直接从 SWITCH=ON 开始。

关键：你运行在竞速模式。三台机器同时学同一模型。
你的每一轮进化结束后，必须执行胜利检测（见下方）。
"
```

### 3.4 竞速机制：胜利检测

每台机器在**每轮进化结束后**（Step 6 更新进化状态之后、Step 7 回到循环之前），执行：

```bash
# 胜利检测脚本
cd /data/metainferv3-learn-${MACHINE_ID}

# 检查远程是否已有其他机器的胜利提交
git fetch origin 2>/dev/null

# 判断：远程的 notebooks-cn/ 是否有本模型的新知识？
if git diff --name-only HEAD..origin/main -- notebooks-cn/02_model_specifics/ 2>/dev/null | grep -q .; then
    echo ""
    echo "═══════════════════════════════════════════════"
    echo "  🏁 竞速结束：另一台机器已率先完成进化"
    echo "  $(git log origin/main --oneline -1)"
    echo "  本机停止。"
    echo "═══════════════════════════════════════════════"
    exit 0
fi
```

**如果本机本轮通过了 L0-L3（进化成功）**：

```bash
cd /data/metainferv3-learn-${MACHINE_ID}

# 先 pull 避免冲突
git pull --rebase origin main 2>/dev/null || true

# 提交本机写入的知识
git add notebooks-cn/
git commit -m "evolution: Qwen3.6 27B — 进化成功，知识库已更新 (machine ${MACHINE_ID})"

# Push 到 bare repo（这步是"宣告胜利"）
git push origin main

echo ""
echo "═══════════════════════════════════════════════"
echo "  🎉 本机获胜！知识已推送到 bare repo。"
echo "  其他机器下一轮检测时会自动停止。"
echo "═══════════════════════════════════════════════"
```

### 3.5 胜负时序

```
时间 ─────────────────────────────────────────────────►

机器 1:  git clone → evo-001(FAIL) → evo-002(FAIL) → evo-003(PASS!)
          │                              │
          │                每次循环后 git fetch 检查
          │                发现远程无更新 → 继续
          │                                            │
          │                          Consolidator 写知识
          │                          git commit + push ──┐
          │                          🛑 获胜，停止        │ 远程有新 commit
          │                                             │
机器 2:  git clone → evo-001(PASS!) → Consolidator      │
          │                              │               │
          │              git fetch → 发现机器 1 已 push!  │
          │              🏁 竞速结束，本机停止 ←──────────┘
          │
机器 3:  git clone → evo-001(FAIL) → git fetch 检查
                                      发现远程有新 commit
                                      🏁 竞速结束，本机停止
```

### 3.6 竞速模式下的特殊规则

| 场景 | 处置 |
|------|------|
| 本机 open-source 辅助通过 → Consolidator | **立即** git push knowledge（给其他机器同步学习成果），然后继续 verify_without_opensource |
| 本机 verify_without_opensource 通过（L0-L3 ✅） | **最终胜利**。git push 知识。停止。 |
| 检测到远程已有本模型知识 | 竞速失败。停止。释放机器。 |
| 两台机器同时 push 冲突 | `git pull --rebase` 解决。知识追加不冲突。先 push 成功的算赢。 |
| 本机连续 3 次失败 | 本机退出竞速（不阻塞其他机器），issue-analyzer 记录日志 |

> **Consolidator 写知识后立即 push 很重要**——即使本机还在 verify_without_opensource 阶段，先 push 的知识能让其他机器少走弯路。

---

## 四、文件归属

| 文件/目录 | git tracked? | 为什么 |
|-----------|-------------|--------|
| `notebooks-cn/` | ✅ | **唯一共享资产**。任何机器写入后 push，其他机器 pull 同步 |
| `.claude/`、`scripts/`、`AGENT_SKILL.md`、`CLAUDE.md` | ✅ | Agent 角色定义和固定测试合约 |
| `evolution/EVOLUTION.md` | ✅ | 进化编排器定义（行为逻辑共享） |
| `evolution/scripts/call-evo-agent.sh` | ✅ | 进化子 agent 启动脚本 |
| `master/MASTER.md` | ✅ | 调优编排器定义（行为逻辑共享） |
| `master/scripts/call-sub-agent.sh` | ✅ | 调优子 agent 启动脚本 |
| `PROJECT_FLOW*.md`、`DEPLOYMENT.md` | ✅ | 流程文档 |
| `evolution/state.json` | ❌ gitignored | 每台机器独立（竞速机器不能共享同一个状态） |
| `evolution/decision-log.jsonl` | ❌ gitignored | 每台机器独立的裁决日志 |
| `evolution/strategies/` | ❌ gitignored | 每台机器独立（各写各的策略） |
| `evolution/results/` | ❌ gitignored | 每台机器独立（各存各的结果） |
| `master/state.json` | ❌ gitignored | 每台机器独立 |
| `master/decision-log.jsonl` | ❌ gitignored | 每台机器独立 |
| `master/strategies/` | ❌ gitignored | 每台机器独立 |
| `master/results/` | ❌ gitignored | 每台机器独立 |
| `engine/`、`llm_engine.py`、`openai_tp_server.py` | ❌ gitignored | 每轮从零重建 |
| `.env_agent_infer` | ❌ gitignored | 机器特定 |
| `phase_report/` | ❌ gitignored | 本地构建报告 |
| `iterations/` | ❌ gitignored | 临时工作区 |
| `knowledge/` | ❌ gitignored | 开源代码缓存（symlink） |

**关键设计**：git 只追踪"行为定义"（`.md` 编排器 + `.claude/` 角色 + `scripts/` 测试 + `notebooks-cn/` 知识）。所有运行时状态和生成产物都是每台机器私有的。

---

## 五、从学习到调优的切换

学习机竞速结束后（已有一台 commit 了知识），在调优机上继续：

```bash
# ─── 调优机器初始化 ───
MACHINE_ID=A
git clone /shared/metainferv3.git /data/metainferv3-tune-${MACHINE_ID}
cd /data/metainferv3-tune-${MACHINE_ID}

# 环境配置
cat > .env_agent_infer << 'ENVEOF'
export AGENT_INFER_ROOT="/data/metainferv3-tune-MACHINE_ID_PLACEHOLDER"
export PYTHON_PATH="/opt/conda/envs/meta/bin"
export MODEL_DIR="/data/models/Qwen3.6-27B"
export PATH="${PYTHON_PATH}:$PATH"
export PYTHONPATH="${AGENT_INFER_ROOT}:$PYTHONPATH"
ENVEOF
sed -i "s|MACHINE_ID_PLACEHOLDER|${MACHINE_ID}|g" .env_agent_infer

# 配置开源代码缓存
ln -s /path/to/your/ref_projects/vllm knowledge/vllm
ln -s /path/to/your/ref_projects/sglang knowledge/sglang

# 创建本机调优状态（gitignored）
mkdir -p master/strategies master/results
cat > master/state.json << 'EOF'
{
  "baseline_iter_id": null,
  "iteration_count": 0,
  "baseline_kpi": null,
  "history": [],
  "failed_directions": [],
  "consecutive_rollbacks": 0,
  "diagnosis_notes": "调优机器。学习阶段已完成（notebooks-cn/ 已有 Qwen3.6 27B 知识）。"
}
EOF

# 启动调优
source .env_agent_infer
claude -p "
你是 MetaInfer 主 Agent。
目标模型 Qwen3.6 27B。CLAUDE.md Step 6 判定 KB 已覆盖 → 进入 master/MASTER.md 调优循环。
无限迭代，用户退出才停。
"
```

---

## 六、Git 协作 SOP

### 写入权限矩阵

| 操作 | 谁做 | 时机 | git 命令 |
|------|------|------|---------|
| 写入模型知识 | 学习机（Consolidator） | 开源辅助通过后 | `git add notebooks-cn/ && git commit && git push` |
| 写入失败记录 | 学习机（issue-analyzer） | 每次失败后 | `git add notebooks-cn/08_issues/ && git commit && git push` |
| 写入调优经验 | 调优机（experiment-summarizer） | ADVANCE 且信号≥2 | `git add notebooks-cn/06_experience/ notebooks-cn/07_improvementPlan/ && git commit && git push` |
| 写入调优失败 | 调优机（issue-analyzer） | 显著 ROLLBACK | `git add notebooks-cn/08_issues/ && git commit && git push` |
| 读取最新知识 | 任何机器 | 每轮开始前 | `git pull --rebase` |
| 竞速检测 | 学习机 | 每轮结束后 | `git fetch && git diff --name-only HEAD..origin/main` |

### 冲突预防

- **不同模型**：写入 `02_model_specifics/` 的不同子目录或不同的 md 文件，不冲突
- **同一模型并行学习**：`notebooks-cn/` 追加写不冲突；先 push 者胜
- **同一模型并行调优**：`06_experience/` 和 `07_improvementPlan/` 追加 `## Δ from iter-XXX` 段落，不冲突
- **`00_contracts/`**：Agent 不直接写（需人类确认），因此不存在冲突

---

## 七、部署检查清单

### 学习机器（并行竞速）

- [ ] bare repo 在 `/shared/metainferv3.git` 可被所有机器访问
- [ ] 每台机器 `git clone` 成功
- [ ] `.env_agent_infer` 中 MACHINE_ID 正确（1/2/3 不重复）
- [ ] `MODEL_DIR` 指向本机模型路径，`PYTHON_PATH` 指向本机 conda 环境
- [ ] `knowledge/vllm` 和 `knowledge/sglang` symlink 有效
- [ ] `evolution/state.json` 已创建（`target_model`="Qwen3.6-27B", `entry_reason`="coverage_fail"）
- [ ] `evolution/state.json` 确实被 gitignored（`git status` 不显示它）
- [ ] 三台机器约同时启动（或任意时间——竞速支持任意加入/退出）

### 调优机器

- [ ] 学习机竞速已结束（`git pull` 能看到 Qwen3.6 27B 的知识）
- [ ] `master/state.json` 已创建
- [ ] 启动后 `CLAUDE.md` Step 6 判定 KB 已覆盖 → 走回路 A+C

---

## 八、完整操作命令速查

```bash
# ═══════ 初始化 bare repo（一台共享机，只做一次） ═══════
cd /data/whl-test/metainferv3
git init
git add notebooks-cn/ .claude/ scripts/ AGENT_SKILL.md CLAUDE.md
git add evolution/EVOLUTION.md evolution/scripts/call-evo-agent.sh
git add master/MASTER.md master/scripts/call-sub-agent.sh
git add .claude/skills/ .claude/roles/
git add .gitignore PROJECT_FLOW*.md DEPLOYMENT.md
git commit -m "v3 baseline"
git clone --bare . /shared/metainferv3.git

# ═══════ 每台学习机器 ═══════
MACHINE_ID=1  # 修改为 1/2/3
git clone /shared/metainferv3.git /data/metainferv3-learn-${MACHINE_ID}
cd /data/metainferv3-learn-${MACHINE_ID}
# 编辑 .env_agent_infer（MODEL_DIR, PYTHON_PATH）
ln -s /path/to/your/ref_projects/vllm knowledge/vllm
ln -s /path/to/your/ref_projects/sglang knowledge/sglang
mkdir -p evolution/strategies evolution/results
# 写入 evolution/state.json（target_model=Qwen3.6-27B, entry_reason=coverage_fail）
source .env_agent_infer
claude -p "你是 MetaInfer 主 Agent。目标 Qwen3.6 27B。按 CLAUDE.md 流程图执行。"

# ═══════ 胜利检测（每轮进化后执行） ═══════
git fetch origin
git diff --name-only HEAD..origin/main -- notebooks-cn/02_model_specifics/ | grep -q . && echo "🏁 竞速结束" && exit 0

# ═══════ 获胜后宣告 ═══════
git add notebooks-cn/
git commit -m "evolution: Qwen3.6 27B 进化成功 (machine ${MACHINE_ID})"
git push origin main

# ═══════ 调优机器 ═══════
cd /data/metainferv3-tune
git pull  # 拿到学习机写入的知识
source .env_agent_infer
claude -p "你是 MetaInfer 主 Agent。目标 Qwen3.6 27B。进入 master 循环。"
```
