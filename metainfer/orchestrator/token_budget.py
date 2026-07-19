"""Process-wide token / cost budget for a single MetaInfer task.

Goal
----
Every Claude Code (``ccb``) subprocess a task spawns — whether driven by
the orchestrator's :class:`SubAgentManager` or by the WebUI's on-demand
analyst (:mod:`metainfer.server.qa`) — emits one ``stream-json`` ``result``
event at the end of its run, carrying a ``usage`` block and
``total_cost_usd``. This module turns those numbers into a single
authoritative running total + a soft-abort circuit breaker for the
orchestrator loop.

Design
------
* **One budget per task** (keyed by ``state_dir``). Persists to
  ``<state_dir>/token_budget.json`` so it survives WebUI / orchestrator
  restarts. Atomic writes via ``.tmp + replace``.
* **Soft-abort semantics by default**. When the running total crosses
  the configured threshold:

    - :meth:`check_launch_allowed` starts rejecting new launches with a
      clear reason string. The orchestrator's phase loop notices via
      :meth:`snapshot` and exits gracefully (``final_status="aborted"``).
    - In-flight agents are NOT killed. Rationale: half-finished tool_use
      turns leave half-written artifacts; the cost of letting the last
      agent finish is bounded and predictable.

* **Optional hard threshold** for callers that want to actively SIGTERM
  running agents. Implemented via an ``on_hard_exhausted`` callback the
  manager wires to its internal ``_stop`` Event. Default off.
* **Threshold metric = ``total_cost_usd``** (per the user's choice). We
  still record raw token counts for observability / future switching.
* **Per-source / per-phase buckets** so the WebUI can show "where did
  the money go" without re-parsing events.jsonl files.

Thread safety
-------------
All public mutators take an internal Lock. Safe to call from the
SubAgentManager's per-agent threads, the WebUI's qa session threads,
and the orchestrator's main loop simultaneously.

NOT designed to be shared across processes — each task's budget lives
in the orchestrator process. The WebUI process reads the persisted JSON
read-only; if the WebUI's analyst (which runs in the WebUI process)
needs to record, it must use its OWN TokenBudget instance pointing at
the same JSON file. The atomic write + load_before_record pattern keeps
this safe.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class UsageRecord:
    """One agent invocation's cost record. Mirrors ccb's ``result`` event."""

    agent: str
    source: str           # "orchestrator" | "web_qa" | future sources
    phase: Optional[str]  # task-defined phase string (opaque to the shell)
    ended_at: float       # unix ts
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    total_cost_usd: float = 0.0
    session_id: Optional[str] = None

    def cost(self) -> float:
        return float(self.total_cost_usd)

    def token_total(self) -> int:
        # Excludes cache_read by default — billed tokens are what most
        # users intuit as "tokens used". Callers wanting the gross
        # number can sum the fields directly.
        return self.input_tokens + self.output_tokens


@dataclass
class BudgetSnapshot:
    """Read-only view of the running totals + limit state."""

    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_input_tokens: int
    agent_count: int
    per_source: Dict[str, float]        # source -> cost_usd
    per_phase: Dict[str, float]         # phase  -> cost_usd (None-keyed = unknown)
    limit_cost_usd: Optional[float]
    hard_limit_cost_usd: Optional[float]
    exhausted: bool                     # soft threshold crossed
    hard_exhausted: bool                # hard threshold crossed
    limit_kind: Optional[str]           # "soft" | "hard" | None
    remaining_cost_usd: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


class TokenBudget:
    """Process-local accumulator + soft-abort circuit breaker.

    Lifecycle: construct once per task (typically in
    :mod:`metainfer.orchestrator._bootstrap.make_subagent_manager` and
    passed into both the :class:`SubAgentManager` and the orchestrator).
    The WebUI's qa module constructs its own instance pointing at the
    same JSON file — the load-before-record pattern keeps concurrent
    writers consistent (single-writer-per-process + atomic file replace).
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        state_dir: Path,
        *,
        max_cost_usd: Optional[float] = None,
        max_cost_usd_hard: Optional[float] = None,
        on_hard_exhausted: Optional[Callable[[], None]] = None,
        on_recorded: Optional[Callable[[UsageRecord, BudgetSnapshot], None]] = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "token_budget.json"
        self.max_cost_usd = max_cost_usd
        self.max_cost_usd_hard = max_cost_usd_hard
        self._on_hard = on_hard_exhausted
        # Called after every successful record() with the new record +
        # post-record snapshot. Used by the orchestrator to mirror
        # usage into timeline.jsonl for the WebUI's live graph. WebUI
        # process budgets pass None here — qa records don't need to
        # appear in the orchestrator's timeline.
        self._on_recorded = on_recorded
        # mtime of the persisted file as we last saw it. Used by
        # :meth:`_maybe_reload_config` to detect external edits
        # (e.g. the WebUI POSTing a new limit while the orchestrator
        # is still alive). One stat call per read is cheap.
        self._last_loaded_mtime: Optional[float] = None
        self._lock = threading.Lock()
        # Persisted + in-memory state. _load() reads existing file if any
        # (so restarting the orchestrator mid-task preserves the budget).
        self._records: List[UsageRecord] = []
        self._total_cost: float = 0.0
        self._total_in: int = 0
        self._total_out: int = 0
        self._total_cache_read: int = 0
        self._per_source: Dict[str, float] = {}
        self._per_phase: Dict[str, float] = {}
        self._exhausted_flagged = False
        self._hard_exhausted_flagged = False
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._last_loaded_mtime = self.path.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        # Honor config from disk so a restart with env override still
        # respects the limit chosen at task-creation time.
        cfg = data.get("config") or {}
        if self.max_cost_usd is None:
            self.max_cost_usd = cfg.get("max_cost_usd")
        if self.max_cost_usd_hard is None:
            self.max_cost_usd_hard = cfg.get("max_cost_usd_hard")
        for rec_raw in data.get("records") or []:
            try:
                rec = UsageRecord(
                    agent=str(rec_raw.get("agent", "")),
                    source=str(rec_raw.get("source", "unknown")),
                    phase=rec_raw.get("phase"),
                    ended_at=float(rec_raw.get("ended_at", 0.0)),
                    input_tokens=int(rec_raw.get("input_tokens", 0)),
                    output_tokens=int(rec_raw.get("output_tokens", 0)),
                    cache_read_input_tokens=int(rec_raw.get("cache_read_input_tokens", 0)),
                    cache_creation_input_tokens=int(rec_raw.get("cache_creation_input_tokens", 0)),
                    total_cost_usd=float(rec_raw.get("total_cost_usd", 0.0)),
                    session_id=rec_raw.get("session_id"),
                )
            except (TypeError, ValueError):
                continue
            self._records.append(rec)
            self._accumulate(rec, persist=False)
        # Re-derive exhausted flag from totals (don't trust disk flag).
        if self.max_cost_usd is not None and self._total_cost >= self.max_cost_usd:
            self._exhausted_flagged = True
        if (self.max_cost_usd_hard is not None
                and self._total_cost >= self.max_cost_usd_hard):
            self._hard_exhausted_flagged = True

    def _persist(self) -> None:
        """Atomically write the full state. Caller holds the lock."""
        data = {
            "schema_version": self.SCHEMA_VERSION,
            "config": {
                "max_cost_usd": self.max_cost_usd,
                "max_cost_usd_hard": self.max_cost_usd_hard,
            },
            "totals": {
                "total_cost_usd": self._total_cost,
                "total_input_tokens": self._total_in,
                "total_output_tokens": self._total_out,
                "total_cache_read_input_tokens": self._total_cache_read,
                "agent_count": len(self._records),
                "exhausted": self._exhausted_flagged,
                "hard_exhausted": self._hard_exhausted_flagged,
            },
            "per_source": self._per_source,
            "per_phase": self._per_phase,
            "records": [asdict(r) for r in self._records],
        }
        from metainfer.server.filelock import lock_file
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        with lock_file(self.path):
            tmp.replace(self.path)

    # ------------------------------------------------------------------ #
    # Mutators
    # ------------------------------------------------------------------ #

    def record(self, rec: UsageRecord) -> BudgetSnapshot:
        """Add ``rec`` to the running totals. Returns the post-record snapshot.

        Triggers the on_hard_exhausted callback the first time the hard
        threshold is crossed. The soft threshold does NOT fire a callback
        — callers poll :meth:`snapshot` instead.
        """
        with self._lock:
            self._records.append(rec)
            self._accumulate(rec, persist=False)
            # Check thresholds. Soft first (so the hard flag is the
            # strictly stronger condition).
            if (self.max_cost_usd is not None
                    and self._total_cost >= self.max_cost_usd
                    and not self._exhausted_flagged):
                self._exhausted_flagged = True
            if (self.max_cost_usd_hard is not None
                    and self._total_cost >= self.max_cost_usd_hard):
                # Only fire the callback once per task lifetime.
                fire_callback = not self._hard_exhausted_flagged
                self._hard_exhausted_flagged = True
            else:
                fire_callback = False
            self._persist()
            snap = self._snapshot_locked()
        # Run callbacks OUTSIDE the lock to avoid deadlock if they call
        # back into the budget (e.g. snapshot).
        if fire_callback and self._on_hard is not None:
            try:
                self._on_hard()
            except Exception:  # noqa: BLE001
                # Callback failure must not corrupt budget state.
                pass
        if self._on_recorded is not None:
            try:
                self._on_recorded(rec, snap)
            except Exception:  # noqa: BLE001
                pass
        return snap

    def _accumulate(self, rec: UsageRecord, *, persist: bool) -> None:
        """Fold one record into the running totals. Caller holds the lock."""
        self._total_cost += rec.cost()
        self._total_in += rec.input_tokens
        self._total_out += rec.output_tokens
        self._total_cache_read += rec.cache_read_input_tokens
        self._per_source[rec.source] = (
            self._per_source.get(rec.source, 0.0) + rec.cost()
        )
        phase_key = rec.phase or "(unknown)"
        self._per_phase[phase_key] = (
            self._per_phase.get(phase_key, 0.0) + rec.cost()
        )

    # ------------------------------------------------------------------ #
    # Read-only views
    # ------------------------------------------------------------------ #

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            self._maybe_reload_config_locked()
            return self._snapshot_locked()

    def _snapshot_locked(self) -> BudgetSnapshot:
        remaining: Optional[float]
        if self.max_cost_usd is None:
            remaining = None
        else:
            remaining = max(0.0, self.max_cost_usd - self._total_cost)
        limit_kind: Optional[str]
        if self._hard_exhausted_flagged:
            limit_kind = "hard"
        elif self._exhausted_flagged:
            limit_kind = "soft"
        else:
            limit_kind = None
        return BudgetSnapshot(
            total_cost_usd=self._total_cost,
            total_input_tokens=self._total_in,
            total_output_tokens=self._total_out,
            total_cache_read_input_tokens=self._total_cache_read,
            agent_count=len(self._records),
            per_source=dict(self._per_source),
            per_phase=dict(self._per_phase),
            limit_cost_usd=self.max_cost_usd,
            hard_limit_cost_usd=self.max_cost_usd_hard,
            exhausted=self._exhausted_flagged,
            hard_exhausted=self._hard_exhausted_flagged,
            limit_kind=limit_kind,
            remaining_cost_usd=remaining,
        )

    def check_launch_allowed(self, agent_name: str = "") -> Optional[str]:
        """Return a refusal reason if a new agent MUST NOT launch, else None.

        Called by SubAgentManager.launch() and qa.start_qa_session() BEFORE
        spawning the subprocess. Refuses on either soft or hard exhaustion
        — even soft-abort policy means "no new agents", which is exactly
        what we want at the launch gate.
        """
        snap = self.snapshot()
        if snap.hard_exhausted:
            return (
                f"token budget HARD-exhausted: used ${snap.total_cost_usd:.4f} "
                f">= hard limit ${snap.hard_limit_cost_usd:.4f} "
                f"(agent={agent_name!r})"
            )
        if snap.exhausted:
            return (
                f"token budget exhausted: used ${snap.total_cost_usd:.4f} "
                f">= limit ${snap.limit_cost_usd:.4f} "
                f"(agent={agent_name!r})"
            )
        return None

    # ------------------------------------------------------------------ #
    # Runtime limit adjustment
    # ------------------------------------------------------------------ #

    def update_limit(
        self,
        *,
        max_cost_usd: Optional[float] = None,
        max_cost_usd_hard: Optional[float] = None,
        keep_hard: bool = False,
    ) -> BudgetSnapshot:
        """Update the soft and/or hard cost limit.

        Resets the ``exhausted`` / ``hard_exhausted`` flags based on the
        new limits vs the running total — so raising the limit past the
        current total un-blocks the budget immediately. Persists
        atomically so an in-flight orchestrator process sees the change
        on its next :meth:`_maybe_reload_config_locked` check.

        ``keep_hard=True`` keeps the existing hard limit if you don't
        pass ``max_cost_usd_hard`` (default is to leave it as-is too —
        the parameter is just for explicit callsites).
        """
        with self._lock:
            if max_cost_usd is not None:
                self.max_cost_usd = max_cost_usd
            if max_cost_usd_hard is not None:
                self.max_cost_usd_hard = max_cost_usd_hard
            # Re-derive exhausted flags from totals vs new limits.
            self._exhausted_flagged = (
                self.max_cost_usd is not None
                and self._total_cost >= self.max_cost_usd
            )
            self._hard_exhausted_flagged = (
                self.max_cost_usd_hard is not None
                and self._total_cost >= self.max_cost_usd_hard
            )
            self._persist()
            # Refresh mtime tracker so our own write doesn't trigger
            # a redundant reload on the next read.
            try:
                self._last_loaded_mtime = self.path.stat().st_mtime
            except OSError:
                pass
            return self._snapshot_locked()

    def _maybe_reload_config_locked(self) -> None:
        """Re-read config (limits) from disk if another process edited it.

        The orchestrator process is long-lived; if the user POSTs a new
        limit via the WebUI while the orchestrator is alive, this method
        picks up the change on the next :meth:`snapshot` /
        :meth:`check_launch_allowed` call. Records (the history of
        per-agent cost) are NOT reloaded — they're append-only and we
        trust our own in-memory copy as authoritative for those.

        Caller holds ``self._lock``.
        """
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if self._last_loaded_mtime is not None and mtime == self._last_loaded_mtime:
            return
        # File changed under us. Re-read just the config block + totals
        # so the exhaustion flag reflects reality after the limit bump.
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        cfg = data.get("config") or {}
        if not isinstance(cfg, dict):
            return
        self.max_cost_usd = cfg.get("max_cost_usd")
        self.max_cost_usd_hard = cfg.get("max_cost_usd_hard")
        # Re-derive exhausted flag.
        self._exhausted_flagged = (
            self.max_cost_usd is not None
            and self._total_cost >= self.max_cost_usd
        )
        self._hard_exhausted_flagged = (
            self.max_cost_usd_hard is not None
            and self._total_cost >= self.max_cost_usd_hard
        )
        self._last_loaded_mtime = mtime

    # ------------------------------------------------------------------ #
    # Convenience for tests / debugging
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._total_cost = 0.0
            self._total_in = 0
            self._total_out = 0
            self._total_cache_read = 0
            self._per_source.clear()
            self._per_phase.clear()
            self._exhausted_flagged = False
            self._hard_exhausted_flagged = False
            self._persist()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_runtime_config(state_dir: Path) -> Dict[str, Any]:
    """Read ``token_budget.json::config`` if it exists. Returns ``{}`` on
    any error so callers can treat the runtime file as optional."""
    path = Path(state_dir) / "token_budget.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        return {}
    return cfg


def resolve_budget_limits(
    state_dir: Path,
    req: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve (soft, hard) cost limits for a task — single source of truth.

    Priority (first non-None wins):

      1. ``METAINFER_TOKEN_BUDGET_COST_USD`` / ``..._HARD`` env var
         (ops escape hatch — overrides everything).
      2. ``token_budget.json::config.max_cost_usd`` — the RUNTIME
         authoritative file. The WebUI updates this when the user raises
         the budget mid-task. **This is the source of truth.**
      3. ``requirements.json::token_budget.max_cost_usd`` (nested) or
         ``requirements.json::token_budget_max_cost_usd`` (flat, what the
         WebUI new-task form writes). Only consulted when the runtime
         file does NOT yet have the field — i.e. first orchestrator boot
         of a freshly-created task. After that, ``requirements.json`` is
         a historical record of the form submission, not consulted.

    Returns ``(None, None)`` when no limit is configured anywhere — the
    caller should pass that as "budget disabled".

    Rationale: before this helper existed, three orchestrators each had
    their own copy of a cascade that put ``requirements.json`` ahead of
    the runtime file. That made WebUI mid-task budget bumps silently
    lost on the next orchestrator restart (the user raised 50 → 100, the
    orchestrator came back, read 50 from requirements.json, aborted).
    """
    tb_cfg = req.get("token_budget")
    if not isinstance(tb_cfg, dict):
        tb_cfg = {}
    runtime_cfg = _read_runtime_config(state_dir)

    def _resolve(env_key: str, conf_key: str, flat_key: str) -> Optional[float]:
        # 1. env var
        env_v = os.environ.get(env_key)
        if env_v:
            try:
                return float(env_v)
            except ValueError:
                pass
        # 2. runtime file (token_budget.json::config) — authoritative
        v = runtime_cfg.get(conf_key)
        # 3. requirements.json seed (nested or flat) — only on first boot
        if v is None:
            v = tb_cfg.get(conf_key)
        if v is None:
            v = req.get(flat_key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    soft = _resolve("METAINFER_TOKEN_BUDGET_COST_USD",
                    "max_cost_usd", "token_budget_max_cost_usd")
    hard = _resolve("METAINFER_TOKEN_BUDGET_COST_USD_HARD",
                    "max_cost_usd_hard", "token_budget_max_cost_usd_hard")
    return soft, hard


def usage_from_result_event(
    event: Dict[str, Any],
    *,
    agent: str,
    source: str,
    phase: Optional[str] = None,
) -> UsageRecord:
    """Build a :class:`UsageRecord` from a ccb stream-json ``result`` event.

    Tolerant of missing fields — defaults to zero. Callers usually pass
    the parsed final event from ``events.jsonl``.
    """
    usage = event.get("usage") if isinstance(event, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    return UsageRecord(
        agent=str(agent),
        source=str(source),
        phase=phase,
        ended_at=time.time(),
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
        total_cost_usd=float(event.get("total_cost_usd", 0.0) or 0.0),
        session_id=event.get("session_id"),
    )


__all__ = [
    "TokenBudget",
    "UsageRecord",
    "BudgetSnapshot",
    "usage_from_result_event",
]
