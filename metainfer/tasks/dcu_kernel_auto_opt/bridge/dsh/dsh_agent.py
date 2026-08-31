#!/usr/bin/env python3
"""MetaInfer -> DSH agent driver (ccb-compatible CLI).

SubAgentManager spawns sub-agents as a CLI process (``claude_bin``, default
``ccb``) with this contract:

    <bin> -p --output-format stream-json --input-format text --verbose \\
          --permission-mode <mode> --add-dir <workdir> [--add-dir <extra>...] \\
          [--model <m>] [--effort <e>] [--resume <sid> | --session-id <sid>] \\
          [extra_args...]

with the agent prompt piped on **stdin**.  The process must emit a
line-delimited stream-json event stream on **stdout** and exit 0 on success:

    {"type":"system",    "session_id": "<id>", ...}          (first)
    {"type":"assistant", "message": {"content": [{"type":"text","text": "..."}]}}
    {"type":"result",    "session_id": "<id>", "result": "<final text>",
     "usage": {...}}                                         (last)

This wrapper speaks that exact protocol but runs a **DeepSeek Harness agent**
through the Python SDK (``deepseek_harness``) instead of Claude Code.  The
orchestrator therefore needs zero code changes: point ``claude_bin`` at this
script (``METAINFER_CLAUDE_BIN`` or ``--claude-bin``) and both the coordinator
(main agent) and the kernel workers (sub-agents) run on DSH.

Environment:
    DSH_AGENT_PROVIDER      provider name for the DSH runtime
                            (default: deepseek-official — the only adapter
                            shipped by the dev-checkout runtime)
    DSH_AGENT_MODEL         model override (default deepseek/deepseek-v4-flash-0731)
    DSH_AGENT_BASE_URL      model endpoint; falls back to DEEPSEEK_BASE_URL
                            (default: https://tokenhub.tencentmaas.com/plan/v3)
    TENCENT_API_KEY         preferred API key (matches ~/.dsh/settings.yaml)
    DEEPSEEK_API_KEY        fallback API key
    DSH_AGENT_CORDIS        custom cordis.yml for the SDK runtime
                            (default: dsh/cordis.yml next to this file)
    DSH_AGENT_SESSION_ROOT  session JSONL persistence root; stable across
                            resume chains (default: <cwd>/.dsh-sessions)
    DSH_AGENT_MAX_TOKENS    per-request output cap (default 65536)
    DSH_AGENT_DEBUG         set to 1 to keep the SDK runtime log lines
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Dev-checkout carrier: this host has no compiled ``deepseek-harness-runtime-bin``
# executable, so the SDK runtime is launched through the dev-only node carrier
# (see resolve_bundled_launch_args). Callers with the exe carrier installed can
# override with DSH_RUNTIME_MODE=exe.
os.environ.setdefault("DSH_RUNTIME_MODE", "node")

# --------------------------------------------------------------------------- #
# Host wiring (worker29: TokenHub DSV4-Flash via the local dsh CLI profile)
# --------------------------------------------------------------------------- #

# Provider name understood by the DSH runtime. The dev-checkout runtime only
# ships the llm-deepseek adapter (provider "deepseek-official"); the TokenHub
# endpoint is selected via DEEPSEEK_BASE_URL below, so the official adapter
# talks to the same gateway the local dsh CLI uses.
def default_provider() -> str:
    return (
        os.environ.get("DSH_AGENT_PROVIDER", "deepseek-official").strip()
        or "deepseek-official"
    )


def default_model() -> str:
    return (
        os.environ.get("DSH_AGENT_MODEL", "deepseek/deepseek-v4-flash-0731")
        .strip() or "deepseek/deepseek-v4-flash-0731"
    )


def default_base_url() -> str:
    env = os.environ.get("DEEPSEEK_BASE_URL")
    if env and env.strip():
        return env.strip()
    return (
        os.environ.get(
            "DSH_AGENT_BASE_URL",
            "https://tokenhub.tencentmaas.com/plan/v3",
        ).strip()
        or "https://tokenhub.tencentmaas.com/plan/v3"
    )


def _credentials_api_key() -> str:
    """Read the API key from the local dsh credentials file when the env
    variables are not set (e.g. when this wrapper runs standalone)."""
    try:
        import yaml

        path = Path(
            os.environ.get("DSH_CREDENTIALS", "~/.dsh/.credentials.yaml")
        ).expanduser()
        if not path.is_file():
            return ""
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key in ("TENCENT_API_KEY", "DEEPSEEK_API_KEY"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:  # noqa: BLE001 - credentials are best-effort
        pass
    return ""


def default_api_key() -> str:
    """Resolve the API key for the configured gateway.

    The TokenHub (tencent) gateway is authenticated by the TENCENT key. The
    DEEPSEEK key is also present in ~/.dsh/.credentials.yaml and is exported
    into the metainfer server env by start_webui.sh, but TokenHub rejects it;
    for that gateway the tencent credential therefore takes priority over the
    DEEPSEEK env fallback.
    """
    tencent_gateway = (
        "tencentmaas" in default_base_url().lower()
        or default_provider().strip().lower() == "tencent"
    )
    if tencent_gateway:
        value = os.environ.get("TENCENT_API_KEY")
        if value and value.strip():
            return value.strip()
        cred = _credentials_api_key()
        if cred:
            return cred
        value = os.environ.get("DEEPSEEK_API_KEY")
        if value and value.strip():
            return value.strip()
    else:
        for env in ("DEEPSEEK_API_KEY", "TENCENT_API_KEY"):
            value = os.environ.get(env)
            if value and value.strip():
                return value.strip()
        return _credentials_api_key()
    return ""


# --------------------------------------------------------------------------- #
# Model mapping (legacy Claude labels -> this host's DSH model ids)
# --------------------------------------------------------------------------- #

_MODEL_MAP = {
    "opus": "deepseek-v4-pro",
    "sonnet": "deepseek-v4-flash",
    "haiku": "deepseek-v4-flash",
}


def map_model(requested: Optional[str]) -> str:
    if not requested:
        return default_model()
    key = requested.strip().lower()
    if key in _MODEL_MAP:
        return _MODEL_MAP[key]
    if key == "deepseek-v4-flash":
        # Bare user-facing label -> the pinned host model id.
        return default_model()
    if key.startswith("deepseek"):
        # Full model id (e.g. deepseek/deepseek-v4-flash-0731): pass through.
        return key
    # Unknown label: keep the caller's intent but stay on a known model id.
    return default_model()


# --------------------------------------------------------------------------- #
# stream-json event emission (SubAgentManager wire protocol)
# --------------------------------------------------------------------------- #

def emit(event: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def emit_system(session_id: str, model: str) -> None:
    emit({
        "type": "system",
        "session_id": session_id,
        "model": model,
        "subagent_id": session_id,
        "cwd": os.getcwd(),
    })


def extract_text_blocks(content: Any) -> List[str]:
    """Pull text blocks from an assistant message content array."""
    if not isinstance(content, list):
        return []
    out: List[str] = []
    for blk in content:
        if isinstance(blk, dict) and blk.get("type") == "text":
            text = blk.get("text")
            if isinstance(text, str):
                out.append(text)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dsh_agent", add_help=False)
    p.add_argument("-p", action="store_true")
    p.add_argument("--output-format")
    p.add_argument("--input-format")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--permission-mode")
    p.add_argument("--add-dir", action="append", default=[])
    p.add_argument("--model")
    p.add_argument("--effort")
    p.add_argument("--resume")
    p.add_argument("--session-id")
    p.add_argument("--max-turns")
    # Anything else (claude-specific) is ignored.
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args, _unknown = build_parser().parse_known_args(argv)

    prompt = sys.stdin.buffer.read().decode("utf-8", errors="replace").strip()
    if not prompt:
        sys.stderr.write("dsh_agent: empty prompt on stdin\n")
        return 1

    try:
        from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig
    except ImportError as exc:  # pragma: no cover - environment check
        sys.stderr.write(
            "dsh_agent: deepseek_harness SDK not installed "
            f"({exc}); run: pip install deepseek-harness-sdk\n"
        )
        return 1

    model = map_model(args.model)
    # Session continuity: --resume continues an existing DSH conversation;
    # --session-id pins the id on the first turn so the orchestrator can
    # resume by it later. Both map to the SDK's per-session id.
    requested_session = args.resume or args.session_id
    session_id = requested_session or f"session-{os.urandom(8).hex()}"

    add_dirs = [str(Path(d).resolve()) for d in args.add_dir if d]
    # Stable session persistence: prefer env, else the last --add-dir (the
    # orchestrator passes workspace_dir after the per-agent workdir), else cwd.
    session_root = os.environ.get("DSH_AGENT_SESSION_ROOT")
    if not session_root and add_dirs:
        session_root = str(Path(add_dirs[-1]) / ".dsh-sessions")
    if not session_root:
        session_root = str(Path.cwd() / ".dsh-sessions")

    cordis = os.environ.get("DSH_AGENT_CORDIS")
    if not cordis:
        cordis = str(Path(__file__).resolve().parent / "cordis.yml")

    max_tokens = int(os.environ.get("DSH_AGENT_MAX_TOKENS", "65536"))
    api_key = default_api_key()
    if not api_key:
        sys.stderr.write(
            "dsh_agent: no API key found (set TENCENT_API_KEY or "
            "DEEPSEEK_API_KEY)\n"
        )
        return 1

    def run_agent(sid: str):
        config = DeepSeekHarnessConfig(
            provider=default_provider(),
            model=model,
            max_tokens=max_tokens,
            cwd=os.getcwd(),
            session_root=session_root,
            cordis=cordis,
            # API key / base url are injected explicitly; the runtime also
            # inherits them from the process environment by default.
            base_url=default_base_url(),
            api_key=api_key,
            env={
                "DSH_SESSION_ROOT": session_root,
                "DSH_CWD": os.getcwd(),
            },
            request_timeout_seconds=3600.0,
            shutdown_timeout_seconds=15.0,
        )

        def on_notification(notification: Any) -> None:
            # Stream assistant text live so the SubAgentManager stuck-watchdog
            # sees fresh stdout and the WebUI log stays readable.
            try:
                if getattr(notification, "method", None) != "session.event":
                    return
                payload = notification.payload or {}
                if payload.get("sessionId") != sid:
                    return
                event = payload.get("event")
                if not isinstance(event, dict):
                    return
                if event.get("type") != "assistant/message":
                    return
                data = event.get("data") or {}
                message = data.get("message")
                content = message.get("content") if isinstance(message, dict) else data.get("content")
                blocks = extract_text_blocks(content)
                if blocks:
                    emit({
                        "type": "assistant",
                        "session_id": sid,
                        "message": {"content": [{"type": "text", "text": b} for b in blocks]},
                    })
            except Exception:  # pragma: no cover - observability must not kill the run
                pass

        try:
            result = DeepSeekHarness(config).run(
                prompt,
                session_id=sid,
                on_notification=on_notification,
            )
            return result, None
        except Exception as exc:  # pragma: no cover - surfaced to the orchestrator
            return None, exc

    # Resume is best-effort: large prior sessions can fail to reload in the
    # runtime (turn/end reason "error" with an empty response). Fall back to a
    # fresh session so every iteration still produces real agent work; the
    # prompt MetaInfer passes is self-contained and includes continuation
    # context, so the fresh run remains effective.
    result, exc = run_agent(session_id)
    if result is not None and result.finish_reason not in (None, "completed", "max-tokens"):
        sys.stderr.write(
            f"dsh_agent: session {session_id} resume failed "
            f"(finish_reason={result.finish_reason!r}); retrying as a fresh session\n"
        )
        session_id = f"session-{os.urandom(8).hex()}"
        result, exc = run_agent(session_id)
    if exc is not None:
        sys.stderr.write(f"dsh_agent: DSH run failed: {exc!r}\n")
        return 1
    if result.finish_reason not in (None, "completed", "max-tokens"):
        sys.stderr.write(
            f"dsh_agent: DSH run finished with reason {result.finish_reason!r}\n"
        )
        return 1

    # Emit the system event only after the successful run so the stream's
    # first session_id is the id the orchestrator should resume from later.
    emit_system(result.session_id, model)
    final_text = result.final_response or ""
    emit({
        "type": "result",
        "session_id": result.session_id,
        "result": final_text,
        "finish_reason": result.finish_reason,
        "usage": {},  # optional; token-budget accounting skips when absent
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
