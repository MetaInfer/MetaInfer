#!/usr/bin/env bash
# ============================================================
# call-evo-agent.sh — 进化子 Agent 启动脚本
#
# 用法: bash evolution/scripts/call-evo-agent.sh <EVO_ID> <STRATEGY_FILE>
#
# 此脚本做的事：
#   1. 清空引擎代码层（保留知识文档）
#   2. 加载环境
#   3. 将策略文件拷入 iterations/
#   4. 按策略的阶段（phase）决定执行哪些子 agent
#   5. 将结果收集到 evolution/results/<EVO_ID>/
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[FAIL]${NC} $*" >&2; exit 1; }

# --- 参数解析 ---
EVO_ID="${1:-}"
STRATEGY_FILE="${2:-}"

if [ -z "$EVO_ID" ] || [ -z "$STRATEGY_FILE" ]; then
    die "Usage: bash $0 <EVO_ID> <STRATEGY_FILE>"
fi

if [ ! -f "$STRATEGY_FILE" ]; then
    die "Strategy file not found: $STRATEGY_FILE"
fi

# --- 确定项目根目录 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
[ -f "CLAUDE.md" ] || die "CLAUDE.md not found — not in project root"

# --- 加载环境 ---
if [ -f .env_agent_infer ]; then
    source .env_agent_infer
else
    die ".env_agent_infer not found. Run setup first."
fi

# --- 清空引擎代码层 ---
log "清空引擎代码层..."
rm -rf engine/
rm -f llm_engine.py
rm -f openai_tp_server.py
rm -rf phase_report/
mkdir -p iterations
log "引擎代码已清空——保留知识文档"

# --- 创建结果目录 ---
RESULTS_DIR="evolution/results/${EVO_ID}"
mkdir -p "$RESULTS_DIR"

# --- 拷贝策略文件 ---
cp "$STRATEGY_FILE" "iterations/"
log "策略已拷贝到 iterations/"

echo ""
echo "=== Evolution Sub-Agent Pipeline ==="
echo "EVO_ID:         $EVO_ID"
echo "Strategy:       $STRATEGY_FILE"
echo "Results dir:    $RESULTS_DIR"
echo ""

# --- 读取开关状态 ---
OPEN_SOURCE_ENABLED=$("${PYTHON_PATH}/python" -c "
import json
with open('${STRATEGY_FILE}') as f:
    s = json.load(f)
print('true' if s.get('open_source_enabled', False) else 'false')
")

PHASE=$("${PYTHON_PATH}/python" -c "
import json
with open('${STRATEGY_FILE}') as f:
    s = json.load(f)
print(s.get('phase', 'attempt_without_opensource'))
")

EXPLORER_MODE=$("${PYTHON_PATH}/python" -c "
import json
with open('${STRATEGY_FILE}') as f:
    s = json.load(f)
print(s.get('explorer_mode', 'full'))
")

TARGET_MODEL=$("${PYTHON_PATH}/python" -c "
import json
with open('${STRATEGY_FILE}') as f:
    s = json.load(f)
print(s.get('target_model', 'unknown'))
")

echo "Phase:              $PHASE"
echo "Open Source Switch: $OPEN_SOURCE_ENABLED"
echo "Explorer Mode:      $EXPLORER_MODE"
echo "Target Model:       $TARGET_MODEL"
echo ""

# --- 退出码累计 ---
FINAL_EXIT=0

# ============================================================
# Phase 1: Explorer（若非 skip 模式）
# ============================================================
if [ "$EXPLORER_MODE" != "skip" ]; then
    echo "═══════════════════════════════════════════"
    echo "  Phase 1: Explorer Agent"
    echo "═══════════════════════════════════════════"

    EXPLORER_SWITCH_FLAG="OFF"
    if [ "$OPEN_SOURCE_ENABLED" = "true" ]; then
        EXPLORER_SWITCH_FLAG="ON"
    fi

    set +e
    claude -p "
读取 .claude/roles/explorer.md 了解你的角色边界。

探索目标模型: ${TARGET_MODEL}
EVO_ID: ${EVO_ID}
开源开关: ${EXPLORER_SWITCH_FLAG}
探索模式: ${EXPLORER_MODE}

当前知识库位于 notebooks-cn/。若开关为 ON，可读取 knowledge/vllm/ 和 knowledge/sglang/ 下的源码。

输出探索报告到 evolution/results/${EVO_ID}/exploration_report.md。
同时输出机器可读差异文件到 evolution/results/${EVO_ID}/model_diff.json。

不写代码。不跑测试。只搜索和分析。
" 2>&1 | tee "$RESULTS_DIR/explorer_stdout.log"
    EXPLORER_EXIT=$?
    set -e

    if [ $EXPLORER_EXIT -ne 0 ]; then
        warn "Explorer exited with code $EXPLORER_EXIT"
    fi

    if [ -f "$RESULTS_DIR/exploration_report.md" ]; then
        log "exploration_report.md generated"
    else
        warn "Explorer did not produce exploration_report.md"
        # 不 fatal——实现者仍然可以尝试用纯知识库
    fi
else
    echo "[SKIP] Explorer phase skipped (mode=skip, using existing KB)"
fi

# ============================================================
# Phase 2: Implementer（生成全部引擎代码）
# ============================================================
echo ""
echo "═══════════════════════════════════════════"
echo "  Phase 2: Implementer Agent (/phase-all)"
echo "═══════════════════════════════════════════"

IMPL_EXTRA_CONTEXT=""
if [ -f "$RESULTS_DIR/exploration_report.md" ]; then
    IMPL_EXTRA_CONTEXT="
额外上下文：请先读取 evolution/results/${EVO_ID}/exploration_report.md 和 evolution/results/${EVO_ID}/model_diff.json。
这些文件包含目标模型 ${TARGET_MODEL} 的架构分析和实现建议。
"
fi

if [ "$OPEN_SOURCE_ENABLED" = "true" ]; then
    IMPL_EXTRA_CONTEXT="${IMPL_EXTRA_CONTEXT}
开源参考开关已开启：探索报告中已包含从 vLLM/SGLang 提取的关键实现信息。
你应在理解开源实现思路的基础上，用自己的代码实现（不逐行复制）。
"
fi

set +e
claude -p "
读取 .claude/roles/implementer-inference.md 了解你的角色边界。

你的 Task：为模型 ${TARGET_MODEL} 实现完整的推理框架。

执行 /phase-all：按 Phase 1-11 顺序构建全部组件。
${IMPL_EXTRA_CONTEXT}

启动前强制读取：
1. notebooks-cn/00_contracts/ 全部契约文件
2. AGENT_SKILL.md §1 执行铁律
3. CLAUDE.md 导航索引表

要求：
- 只写代码，不跑 scripts/ 测试
- 自读 diff，确认没有修改 scripts/ 下的文件
- 报告状态为 SUBMITTED，不是 PASS
- 输出文件清单、改动的关键代码段、自检结果

代码直接写入本目录下（./engine/、./llm_engine.py、./openai_tp_server.py）。
" 2>&1 | tee "$RESULTS_DIR/implementer_stdout.log"
IMPL_EXIT=$?
set -e

if [ $IMPL_EXIT -ne 0 ]; then
    warn "Implementer exited with code $IMPL_EXIT"
    FINAL_EXIT=1
else
    log "Implementer completed"
fi

# ============================================================
# Phase 3: Verification（测试验收）
# ============================================================
echo ""
echo "═══════════════════════════════════════════"
echo "  Phase 3: Verification Agent"
echo "═══════════════════════════════════════════"

set +e
claude -p "
读取 .claude/roles/verification-inference.md 了解你的角色边界。

验收对象：./engine/、./llm_engine.py、./openai_tp_server.py
目标模型: ${TARGET_MODEL}

验收内容（按 verification-inference.md 的双重验证标准）：
- L0（强制）：防假 PASS 路径验证——确认 import 的代码来自本目录
- L0.5（强制）：self_check 反作弊预检——运行时验证代码非 no-op
- L0.6（强制）：Agent 自检 5 条——静态分析测试覆盖盲区
- L1：运行全部 Phase 的 scripts/ 脚本，记录每个的 PASS/FAIL
- L2：跨 Phase 回归——重跑所有前序 Phase 的 scripts/

将最终验收结果直接写入 evolution/results/${EVO_ID}/：
- benchmarks.jsonl: {\"evo_id\":\"${EVO_ID}\",\"metrics\":{\"throughput_tok_s\":<float>,\"ttft_ms\":<float>,\"tpot_ms\":<float>},\"correctness\":{\"greedy_match\":<bool>}}
- AGGREGATE_REPORT.md: Phase 状态表（每个 Phase 的 PASS/FAIL + one_pass_rate + regression_count + spec_review_all_pass）
- diagnostics_summary.json（若 profiler 可用）: category_breakdown/headroom_gb/recommendations

不要读 implementer 或 explorer 的输出。只看测试结果。
全部 PASS 才算通过，任一 FAIL 则列出失败脚本 + 错误码。
" 2>&1 | tee "$RESULTS_DIR/verification_stdout.log"
VERIF_EXIT=$?
set -e

if [ $VERIF_EXIT -ne 0 ]; then
    warn "Verification reported issues (exit code $VERIF_EXIT)"
fi

# ============================================================
# 收尾：收集所有结果
# ============================================================
echo ""
echo "=== 收尾：收集结果 ==="

# 如果 Phase 2 的 /phase-all 内部写入了 phase_report/，补充收集到 results/
if [ -d "phase_report" ]; then
    for f in benchmarks.jsonl AGGREGATE_REPORT.md diagnostics_summary.json FAILURE_REPORT.md; do
        if [ -f "phase_report/${f}" ] && [ ! -f "${RESULTS_DIR}/${f}" ]; then
            cp "phase_report/${f}" "${RESULTS_DIR}/${f}"
            log "从 phase_report/ 补充: ${f}"
        fi
    done
fi

# 验证关键产出是否存在
echo ""
for required in AGGREGATE_REPORT.md benchmarks.jsonl; do
    if [ -f "${RESULTS_DIR}/${required}" ]; then
        log "${required} — present"
    else
        warn "${required} — MISSING"
    fi
done

echo ""
echo "=== Evolution Pipeline Complete ==="
echo "Results: $RESULTS_DIR"
echo "Files:"
ls -la "$RESULTS_DIR/"
echo "Exit code: $FINAL_EXIT"

exit $FINAL_EXIT
