# Phase 8 Spec Review Report（再審查）

- **PID**: 976651
- **Role**: spec-reviewer
- **Timestamp**: 2026-05-30
- **Phase**: 8
- **Scope**: `engine/scheduler.py` only（上次 2 個 FAIL 的修復確認）

---

## Spec Compliance: ✅ PASS

---

## Re-Review Focus: Previous 2 FAILs

### ❌→✅ Old Issue 1: RUNNING_DECODE→RUNNING_DECODE crash（已修復）

- **JSON Path**: `framework_layer.data_flow_contracts.scheduler_to_runner.schedule_algorithm.state_transition_safety`
- **Old Location**: `engine/scheduler.py:96`（舊程式碼：decode branch 呼叫 `seq.transition_to(SeqStatus.RUNNING_DECODE)`）
- **Current Code**: `engine/scheduler.py:94-99`

**核驗結果**：decode branch 已完全移除 `seq.transition_to()` 呼叫。

```python
# engine/scheduler.py:87-99 — Phase 2: Decode
        for seq in list(self.running):
            if num_free - reserved >= 1:
                batch.append(seq)
                reserved += 1

        if batch:
            # No status transition — decode sequences are already RUNNING_DECODE
            # from previous postprocess() call. transition_to(RUNNING_DECODE) again
            # would AssertionError (not in valid transition table).
            self._reserved_blocks += reserved
            return batch, False
```

- 第 95-97 行註釋明確說明：decode sequences 已在 `postprocess()` 中設為 `RUNNING_DECODE`，再次 `transition_to(RUNNING_DECODE)` 會觸發 AssertionError。
- `structs.py:80` 的 transition table：`RUNNING_DECODE → (FINISHED, REJECTED)` — 不包含 `RUNNING_DECODE` 自身。
- **修復確認**：✅ 不再有無效狀態轉換，框架可以完成任意步數的 decode step。

### ❌→🟡 Old Issue 2: prefill branch 狀態轉換仍在 schedule() 中（功能安全）

- **JSON Path**: `framework_layer.data_flow_contracts.scheduler_to_runner.schedule_algorithm.state_transition_safety`
- **Location**: `engine/scheduler.py:79-85`

**核驗結果**：prefill branch 仍執行 queue 操作與狀態轉換：

```python
# engine/scheduler.py:79-85
        if batch:
            for seq in batch:
                self.waiting.remove(seq)
                seq.transition_to(SeqStatus.RUNNING_PREFILL)
                self.running.append(seq)
            self._reserved_blocks += reserved
            return batch, True
```

但此行為**功能安全**，原因：
1. `WAITING → RUNNING_PREFILL` 是有效轉換（`structs.py:78`）
2. `postprocess()` prefill branch（`scheduler.py:121`）接續執行 `RUNNING_PREFILL → RUNNING_DECODE`
3. 從 waiting 移除序列是必要的——否則下一次 `schedule()` 會重複排程同一序列
4. `postprocess()` 中 `seq not in self.running` 的 guard 已不再需要（因為已在 schedule 中 append）

**與藍圖的差異**：藍圖 pseudocode（`schedule_complete_method`）未在 `schedule()` 中顯示 queue mutation，但 pseudocode 本身存在資訊缺口——它未說明如何防止已排程序列被重複選取。當前實作在 prefill branch 中處理 queue 遷移是合理的工程選擇。舊 Issue 2 的核心風險（crash）已隨 Issue 1 的修復而消除。

**結論**：從 ❌ FAIL 降級為 🟡 觀察項。不阻擋 verification。

---

## Full Evidence Chain（逐條核驗）

### Scheduler（scheduler.py）

| # | JSON Path / Contract | Status | Location | Detail |
|---|---------------------|--------|----------|--------|
| 1 | `_nano_vllm_override` — preempt() deleted | ✅ | `:148-150` | preempt() 完全不存在；註釋確認故意刪除 |
| 2 | `schedule_algorithm.schedule_complete_method` — `self._block_size` 可注入 | ✅ | `:27` | `self._block_size = 16`（DEFAULT），LLMEngine 覆寫 |
| 3 | `schedule_algorithm.max_blocks_for_model` — `_max_blocks` 可注入 | ✅ | `:28` | `self._max_blocks = 128`（DEFAULT），LLMEngine 覆寫 |
| 4 | `schedule(num_free)` 簽名 | ✅ | `:46` | `def schedule(self, num_free):` |
| 5 | Phase 1 prefill-first from waiting | ✅ | `:63-85` | 優先從 waiting 隊列取序列 |
| 6 | Phase 2 decode from running（waiting empty 後） | ✅ | `:88-102` | waiting 為空後才從 running 選取 |
| 7 | prefill/decode 不混批 | ✅ | `:79-85,94-99` | 兩個獨立 phase，prefill 有結果立即返回 |
| 8 | `can_allocate` 公式 `num_free >= required_blocks()` | ✅ | `:71` | `if reserved + req > num_free: break` |
| 9 | `can_append_one_more` 公式 `num_free >= 1` | ✅ | `:90` | `if num_free - reserved >= 1:` |
| 10 | overlength rejection in `add()` | ✅ | `:40-43` | `req > self._max_blocks → REJECTED` |
| 11 | overlength rejection guard in `schedule()` | ✅ | `:66-69` | 雙重防護 |
| 12 | **decode branch 不呼叫 transition_to**（舊 Issue 1） | ✅ | `:94-99` | **已修復**：無狀態轉換，只有 reserve + return |
| 13 | postprocess prefill 分支：RUNNING_PREFILL→RUNNING_DECODE | ✅ | `:121` | `seq.transition_to(SeqStatus.RUNNING_DECODE)` |
| 14 | postprocess decode 分支：EOS/max_tokens→FINISHED | ✅ | `:128-136` | 正確檢測終止條件 |
| 15 | `_release()` 清空 block_table + block_table_tensor | ✅ | `:142-144` | `seq.block_table = []`; `seq._block_table_tensor = None` |
| 16 | `is_finished()` 檢查 waiting + running 均為空 | ✅ | `:146-147` | `return not self.waiting and not self.running` |

### Sequence（structs.py）— 關聯驗證

| # | Contract | Status | Location | Detail |
|---|----------|--------|----------|--------|
| 17 | `WAITING→RUNNING_PREFILL` valid | ✅ | `structs.py:78` | transition table 包含此轉換 |
| 18 | `RUNNING_PREFILL→RUNNING_DECODE` valid | ✅ | `structs.py:79` | postprocess 使用此轉換 |
| 19 | `RUNNING_DECODE→FINISHED` valid | ✅ | `structs.py:80` | postprocess decode 使用此轉換 |
| 20 | `RUNNING_DECODE→RUNNING_DECODE` **invalid**（正確） | ✅ | `structs.py:80` | transition table 正確阻止此循環轉換 |

---

## ⚠️ 3 High-Alert Items — Results

| Item | Status | Detail |
|------|--------|--------|
| preempt() 完全刪除 | ✅ PASS | `engine/scheduler.py:148-150` — method does not exist anywhere in the file |
| block_size 可注入 | ✅ PASS | `engine/scheduler.py:27` — `self._block_size = 16` class attribute; LLMEngine overrides via `scheduler._block_size = 256` |
| TP 採樣僅 rank0+broadcast | ✅ PASS | `engine/sampler.py:85-93` — unchanged from previous review |

---

## Remaining Observations（非阻擋）

### 🟡 Obs-1: prefill branch 中 queue mutation 仍在 schedule() 而非 postprocess()

- **JSON Path**: `framework_layer.data_flow_contracts.scheduler_to_runner.schedule_algorithm.state_transition_safety`
- **Location**: `engine/scheduler.py:80-84`
- **Detail**: 藍圖 pseudocode 的 `schedule()` 未顯示 queue mutation，但 pseudocode 本身未提供防止重複排程的機制。當前實作的 `WAITING→RUNNING_PREFILL` 轉換是有效且功能正確的。
- **Risk**: 低。若 `schedule()` 返回後、`postprocess()` 執行前發生異常，序列會留在 `self.running` 中且狀態為 `RUNNING_PREFILL`。但此場景在 TP 推裡路徑中極罕見（無搶佔、無並發）。
- **Recommendation**: 可接受。若未來要嚴格符合藍圖，可將 queue mutation 移至 `postprocess()` prefill 分支並增加 `seq not in self.running` guard。非 Phase 8 範圍。

### 🟡 Obs-2: `num_cached_tokens` 字段仍缺失（繼承自上次審查）

- **JSON Path**: `framework_layer.data_flow_contracts.request_level.sequence_fields.num_cached_tokens`
- **Location**: `engine/structs.py:32-51`（Sequence.__init__）
- **Detail**: TP 路徑不使用 prefix caching，不影響功能。此字段為 HF 路徑的 BlockManager prefix caching 所需。
- **Impact**: 不影響 Phase 8 交付。

---

## Blueprint Information Gaps（繼承自上次審查）

- `data_flow_contracts.request_level.sequence_fields.status`: 🟡 藍圖列出 4 個狀態但 `REJECTED` 為第 5 個。status 文檔應包含 REJECTED。
- `schedule_algorithm.schedule_complete_method`: 🟡 Pseudocode 有語法錯誤（lines 484-485 orphaned `if req > self._max_blocks`）。`max_blocks_for_model` 變量名與注入的 `_max_blocks` 屬性名不一致。

---

## Verdict

**✅ PASS** — 上次審查的 2 個 FAIL 已處理：

1. **舊 Issue 1（Critical）**：`schedule()` decode branch 的 `seq.transition_to(RUNNING_DECODE)` 已完全移除。框架現在可以安全執行任意步數的 decode step，不再觸發 AssertionError。
2. **舊 Issue 2（Architectural）**：prefill branch 的 queue mutation 仍存在於 `schedule()` 中，但僅涉及有效轉換 `WAITING→RUNNING_PREFILL`，後續由 `postprocess()` 正確接續 `RUNNING_PREFILL→RUNNING_DECODE`。功能安全，降級為觀察項。

無新增阻擋問題。Phase 8 Scheduler 可移交 verification。
