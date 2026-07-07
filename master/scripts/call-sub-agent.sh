#!/usr/bin/env bash
# call-sub-agent.sh — 子 agent 调用脚本（metainferv3 增量版）
# 用法: 在 metainferv3 根目录下执行
#   bash master/scripts/call-sub-agent.sh <ITER_ID> master/strategies/strategy-<ITER_ID>.json [MODE]
#
# MODE（可选，自动检测）:
#   - "full" (iter-001 或显式指定): 清空引擎代码 + /phase-all 全量构建
#   - "incremental" (iter-002+ 默认): 保留引擎代码 + /phase-modify 增量修改
#   - "rollback" (显式指定): 从上一轮快照恢复代码后，再执行增量修改
#
# 流程:
#   === full 模式（首轮）===
#   1. 清空引擎代码层（保留知识文档 + master/ + iterations/）
#   2. 加载 .env_agent_infer + 拷贝策略文件到 iterations/
#   3. 通过 ${CLAUDE_CLI} -p 启动子 agent（进程隔离）
#      → 子 agent 读知识文档 + 策略，通过 /phase-all 从零生成完整引擎
#      → 每 Phase 的 scripts/test_phaseN_* 全部 PASS 后才进入下一 Phase
#      → Phase 10/11 完成 benchmark + greedy align 实验
#      → 写入 KPI 到 iterations/<ITER_ID>/phase_report/benchmarks.jsonl
#   4. 检查退出码（0 = 全部 Phase 测试通过）
#   5. 从 iterations/<ITER_ID>/ 收集结果到 master/results/<ITER_ID>/
#   6. 成功时保存代码快照到 master/results/<ITER_ID>/code/
#   7. 返回子 agent 退出码
#
#   === incremental 模式（后续轮次）===
#   1. 检查引擎代码存在（不清理）
#   2. 加载 .env_agent_infer + 拷贝策略文件到 iterations/
#   3. 通过 ${CLAUDE_CLI} -p 启动子 agent（进程隔离）
#      → 子 agent 读已有代码 + 策略 changes[]
#      → 通过 /phase-modify 增量应用变更
#      → 只运行受影响 Phase 的 scripts/ 门禁
#   4. 检查退出码（0 = 修改 + 验证通过）
#   5. 从 iterations/<ITER_ID>/ 收集结果到 master/results/<ITER_ID>/
#   6. 成功时保存代码快照，失败时从上一轮快照恢复
#   7. 返回子 agent 退出码

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[FAIL]${NC} $*" >&2; exit 1; }

# ---- 参数 ----
ITER_ID="${1:?用法: $0 <ITER_ID> <STRATEGY_FILE> [MODE]}"
STRATEGY_FILE="${2:?}"
MODE="${3:-auto}"

# ---- 路径（全部相对于 metainferv3 根目录） ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"       # master/scripts/
MASTER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"       # master/
PROJECT_ROOT="$(cd "${MASTER_DIR}/.." && pwd)"     # metainferv3 根目录
RESULTS_DIR="${MASTER_DIR}/results/${ITER_ID}"     # master/results/<ITER_ID>/
STRATEGY_DEST="${PROJECT_ROOT}/iterations/strategy-${ITER_ID}.json"
ENV_FILE="${PROJECT_ROOT}/.env_agent_infer"

# ---- 校验 ----
[ -f "${STRATEGY_FILE}" ] || die "策略文件不存在: ${STRATEGY_FILE}"
[ -f "${PROJECT_ROOT}/CLAUDE.md" ] || die "CLAUDE.md 不存在，请确认在 metainferv3 根目录下执行"
[ -f "${ENV_FILE}" ] || die ".env_agent_infer 不存在，请先配置环境"

# ---- 自动检测 MODE ----
if [ "${MODE}" = "auto" ]; then
    if [ "${ITER_ID}" = "iter-001" ]; then
        MODE="full"
    else
        MODE="incremental"
    fi
fi

log "ITER_ID=${ITER_ID}"
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "STRATEGY_FILE=${STRATEGY_FILE}"
log "MODE=${MODE}"
log "RESULTS_DIR=${RESULTS_DIR}"

# ---- 1. 模式分支 ----
cd "${PROJECT_ROOT}"

if [ "${MODE}" = "full" ]; then
    # ====== FULL MODE: 首轮全量构建 ======
    log "全量构建模式——清空引擎代码..."
    rm -rf engine/
    rm -f llm_engine.py
    rm -f openai_tp_server.py
    rm -rf phase_report/
    mkdir -p iterations
    log "引擎代码已清空——保留知识文档和 master/"

elif [ "${MODE}" = "rollback" ]; then
    # ====== ROLLBACK MODE: 恢复上一轮代码再增量修改 ======
    warn "ROLLBACK 模式——从上一轮快照恢复代码..."

    # 找到上一个成功的 iter_id
    PREV_STATE="${MASTER_DIR}/state.json"
    if [ -f "${PREV_STATE}" ]; then
        PREV_SUCCESS=$(python3 -c "
import json,sys
try:
    s=json.load(open('${PREV_STATE}'))
    hist=s.get('history',[])
    # 找到最后一个 success=true 的 iter
    for h in reversed(hist):
        if h.get('success'):
            print(h['iter_id'])
            sys.exit(0)
except: pass
" 2>/dev/null || echo "")
    else
        PREV_SUCCESS=""
    fi

    if [ -n "${PREV_SUCCESS}" ] && [ -d "${MASTER_DIR}/results/${PREV_SUCCESS}/code" ]; then
        log "从 master/results/${PREV_SUCCESS}/code/ 恢复代码..."
        rm -rf engine/
        rm -f llm_engine.py openai_tp_server.py
        cp -r "${MASTER_DIR}/results/${PREV_SUCCESS}/code/engine" engine/ 2>/dev/null || true
        cp "${MASTER_DIR}/results/${PREV_SUCCESS}/code/llm_engine.py" llm_engine.py 2>/dev/null || true
        cp "${MASTER_DIR}/results/${PREV_SUCCESS}/code/openai_tp_server.py" openai_tp_server.py 2>/dev/null || true
        log "代码已从 ${PREV_SUCCESS} 快照恢复"
        # 恢复后走 incremental 模式
        MODE="incremental"
    else
        die "无法找到上一轮成功快照，无法恢复。请检查 master/results/"
    fi
fi

if [ "${MODE}" = "incremental" ]; then
    # ====== INCREMENTAL MODE: 后续轮次增量修改 ======
    log "增量修改模式——保留引擎代码..."

    # 检查引擎代码是否存在
    if [ ! -d "engine" ]; then
        die "引擎代码不存在，无法增量修改。请先完成首轮全量构建（iter-001）"
    fi
    if [ ! -f "llm_engine.py" ]; then
        warn "llm_engine.py 不存在，将在增量修改中创建"
    fi
    log "引擎代码检查通过，将在现有代码基础上增量修改"
fi

# ---- 2. 加载环境 + 拷贝策略 ----
log "配置环境与策略..."
source "${ENV_FILE}" 2>/dev/null || die "无法加载 .env_agent_infer"
cp "${STRATEGY_FILE}" "${STRATEGY_DEST}"
log "环境已加载，策略已拷贝到 iterations/"

# ---- 3. 生成子 agent prompt ----
if [ "${MODE}" = "full" ]; then
    # 全量构建 prompt
    SUBAGENT_PROMPT="第 ${ITER_ID} 轮迭代（全量构建）。

你的行为由 CLAUDE.md 定义。先读它，然后严格遵循其中描述的 SOP。

引擎代码层已被清空。你正在从头构建 MetaInfer。

## 本轮策略

读 iterations/strategy-${ITER_ID}.json。这个文件告诉你本轮与默认的不同之处：
- strategy_type = \"baseline\"，changes 为空 → 按契约文件构建默认引擎
- strategy_type = \"change_param\" → 使用策略中指定的参数值（如 block_size=128）
- strategy_type = \"algorithmic\" → 使用策略中指定的算法/算子方案
- strategy_type = \"architectural\" → 使用策略中指定的架构决策
- strategy_type = \"expose_param\" → 暴露新的可调参数到引擎代码中

先理解本轮策略，然后通过 /phase-all 构建完整引擎。

## 规则

- 严格按 AGENT_SKILL.md 门禁规则：每 Phase 的 scripts/test_phaseN_* 全部 PASS 后才能进入下一 Phase
- scripts/ 不可变——测试不过就改实现代码，绝不改测试
- 全部 Phase 通过（含 Phase 10/11 的 benchmark 和 greedy align）后：
  1. 写摘要到 iterations/${ITER_ID}/code_changes.txt
  2. 将最终 KPI 写入 iterations/${ITER_ID}/phase_report/benchmarks.jsonl，格式：
     {\"iter_id\": \"${ITER_ID}\", \"metrics\": {\"throughput_tok_s\": <float>, \"ttft_ms\": <float>, \"tpot_ms\": <float>}, \"correctness\": {\"greedy_match\": <bool>}, \"output\": \"<truncated>\"}
  3. 写 AGGREGATE_REPORT.md 到 iterations/${ITER_ID}/phase_report/
  4. 写 FAILURE_REPORT.md 到 iterations/${ITER_ID}/phase_report/（如有任何失败）
  然后退出"
else
    # 增量修改 prompt
    SUBAGENT_PROMPT="第 ${ITER_ID} 轮迭代（增量修改）。

你的行为由 CLAUDE.md 定义。先读它，然后严格遵循其中描述的 SOP。

引擎代码已经存在（从上一轮成功迭代继承）。你不需要从头构建——只需要增量修改。

## 增量修改模式

引擎代码已经存在于以下位置：
- engine/ — 推理框架代码
- llm_engine.py — 引擎主循环

不要清理这些文件。读取它们，理解当前架构。

## 本轮策略

读 iterations/strategy-${ITER_ID}.json。这个文件的 changes[] 数组是你要应用的精确修改。

然后按照 .claude/skills/phase-modify/SKILL.md 中定义的增量修改流程执行：
1. 读策略 changes[]
2. 对每个 change 定位文件 → 修改代码（使用 Edit 工具，不要重写整个文件）
3. 只运行受影响 Phase 的 scripts/ 门禁测试
4. 如果任一 FAIL → 回退修改 → 报告 FAIL
5. 全部 PASS → 写结果

## 规则

- 不清理引擎代码，不运行 /phase-all
- scripts/ 不可变——测试不过就回退修改，绝不改测试
- 修改全部通过后：
  1. 写摘要到 iterations/${ITER_ID}/code_changes.txt（列出所有修改的文件和内容）
  2. 运行 benchmark 测量 KPI，写入 iterations/${ITER_ID}/phase_report/benchmarks.jsonl，格式：
     {\"iter_id\": \"${ITER_ID}\", \"metrics\": {\"throughput_tok_s\": <float>, \"ttft_ms\": <float>, \"tpot_ms\": <float>}, \"correctness\": {\"greedy_match\": <bool>}, \"output\": \"<truncated>\"}
  3. 写 AGGREGATE_REPORT.md 到 iterations/${ITER_ID}/phase_report/
  4. 写 FAILURE_REPORT.md 到 iterations/${ITER_ID}/phase_report/（如有任何失败）
  然后退出"
fi

# ---- 4. 启动子 agent ----
echo "============================================"
echo " 启动子 agent: ${ITER_ID}"
echo " 模式:       ${MODE}"
echo " 项目根目录:  ${PROJECT_ROOT}"
echo "============================================"

START_TS=$(date +%s)

# 用临时文件传递 prompt，避免 shell 转义问题
PROMPT_FILE="/tmp/subagent_prompt_${ITER_ID}.txt"
echo "${SUBAGENT_PROMPT}" > "${PROMPT_FILE}"

set +e
cd "${PROJECT_ROOT}"
source "${ENV_FILE}" 2>/dev/null || true
${CLAUDE_CLI:-claude} -p "$(cat ${PROMPT_FILE})" \
    --output-format text \
    --allowedTools "Bash(*:*) Read(*:*) Write(*:*) Edit(*:*) Glob(*:*) Grep(*:*) Skill(*:*) Agent(*:*) Task(*:*)"
SUBAGENT_EXIT=$?
set -e

rm -f "${PROMPT_FILE}"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "============================================"
echo " 子 agent 已退出"
echo "   退出码:  ${SUBAGENT_EXIT}"
echo "   耗时:    ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "============================================"

# ---- 5. 子 agent 自检结果 ----
if [ ${SUBAGENT_EXIT} -ne 0 ]; then
  warn "子 agent 退出码非零（${SUBAGENT_EXIT}）——scripts/ 测试未全部通过"
fi

# ---- 6. 收集结果 ----
log "收集结果..."

ITER_RESULTS="${PROJECT_ROOT}/iterations/${ITER_ID}/phase_report"
RESULTS_CODE="${RESULTS_DIR}/code"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_CODE}"

# 拉取 KPI 数据
if [ -f "${ITER_RESULTS}/benchmarks.jsonl" ]; then
    cp "${ITER_RESULTS}/benchmarks.jsonl" "${RESULTS_DIR}/benchmarks.jsonl"
    log "已收集 benchmarks.jsonl"
else
    warn "benchmarks.jsonl 未找到"
fi

if [ -f "${ITER_RESULTS}/AGGREGATE_REPORT.md" ]; then
    cp "${ITER_RESULTS}/AGGREGATE_REPORT.md" "${RESULTS_DIR}/AGGREGATE_REPORT.md"
    log "已收集 AGGREGATE_REPORT.md"
else
    warn "AGGREGATE_REPORT.md 未找到"
fi

# 拉取额外文件
for f in FAILURE_REPORT.md diagnostics_summary.json; do
    src="${ITER_RESULTS}/${f}"
    if [ -f "${src}" ]; then
        cp "${src}" "${RESULTS_DIR}/${f}"
    fi
done

# 拉取子 agent 的代码变更摘要
CODE_CHANGES="${PROJECT_ROOT}/iterations/${ITER_ID}/code_changes.txt"
if [ -f "${CODE_CHANGES}" ]; then
    cp "${CODE_CHANGES}" "${RESULTS_DIR}/code_changes.txt"
fi

# 保存发送给子 agent 的 prompt（用于审计）
echo "${SUBAGENT_PROMPT}" > "${RESULTS_DIR}/subagent_prompt.txt"

# ---- 7. 代码快照（仅在成功时） ----
if [ ${SUBAGENT_EXIT} -eq 0 ]; then
    log "保存代码快照..."
    if [ -d "${PROJECT_ROOT}/engine" ]; then
        cp -r "${PROJECT_ROOT}/engine" "${RESULTS_CODE}/"
    fi
    if [ -f "${PROJECT_ROOT}/llm_engine.py" ]; then
        cp "${PROJECT_ROOT}/llm_engine.py" "${RESULTS_CODE}/"
    fi
    if [ -f "${PROJECT_ROOT}/openai_tp_server.py" ]; then
        cp "${PROJECT_ROOT}/openai_tp_server.py" "${RESULTS_CODE}/"
    fi
    log "代码快照已保存到 ${RESULTS_CODE}/"
fi

log "结果已收集到 ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"

exit ${SUBAGENT_EXIT}
