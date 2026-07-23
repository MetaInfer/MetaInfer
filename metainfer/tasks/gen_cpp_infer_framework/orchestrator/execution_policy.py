"""Task-local command policy for generated-framework implementation turns.

Prompts explain the safe build and process lifecycle, but prompt text is not
an enforcement boundary.  This module inspects the structured Claude event
stream after B and turns dangerous or non-reproducible shell actions into a
deterministic phase failure with exact file/line evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator, Mapping, Tuple


_SHELL_TOOLS = {"bash", "shell", "exec_command"}
_COMMAND_START = r"(?:^|[;&|()\n]\s*)"
_PREFIX = r"(?:(?:sudo|command)\s+|(?:env\s+)?(?:[A-Za-z_]\w*=\S+\s+)*)"
_DIRECT_BUILD_RE = re.compile(
    _COMMAND_START
    + _PREFIX
    + r"(?:cmake|hipcc|clang\+\+|g\+\+|c\+\+|make|ninja)(?=\s|$)",
    re.IGNORECASE,
)
_KILL_RE = re.compile(
    r"(?:^|[;&|()'\"\n]\s*)" + _PREFIX + r"kill(?=\s|$)", re.IGNORECASE
)
_OWNED_PID_RE = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*\$!")
_VARIABLE_REF_RE = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")
_BACKGROUND_TEST_RE = re.compile(
    r"(?:^|[;&|()\n]\s*)(?:bash\s+|\./)?test\.sh\b"
    r"[^\n;]*?(?<![>&])&(?![&0-9])(?:\s|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CommandPolicyViolation:
    events_file: Path
    line: int
    rule: str
    command: str

    def render(self) -> str:
        compact = " ".join(self.command.split())
        if len(compact) > 240:
            compact = compact[:237] + "..."
        return f"{self.events_file.name}:{self.line} [{self.rule}] {compact}"


def validate_code_writer_commands(
    logs_dir: Path,
    agent_name: str,
    *,
    max_errors: int = 20,
    final_attempt_only: bool = True,
) -> Tuple[str, ...]:
    """Return unblocked policy errors from the accepted code-writer attempt.

    PreToolUse denials are corrective feedback, not iteration failures. A
    failed SubAgentManager attempt also cannot poison a later clean retry.
    Callers may set ``final_attempt_only=False`` for forensic audits.
    """
    violations = []
    blocked_hashes = _blocked_command_hashes(logs_dir, agent_name)
    events_files = _attempt_event_files(
        logs_dir, agent_name, final_attempt_only=final_attempt_only
    )
    for events_file in events_files:
        violations.extend(_scan_events_file(events_file, blocked_hashes))
        if len(violations) >= max_errors:
            break
    return tuple(item.render() for item in violations[:max_errors])


# Compatibility for existing task-local callers and forensic scripts.
validate_implementer_commands = validate_code_writer_commands


def _scan_events_file(
    path: Path, blocked_hashes: frozenset[str] = frozenset()
) -> Tuple[CommandPolicyViolation, ...]:
    violations = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    for line_number, raw in enumerate(lines, start=1):
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for command in _iter_shell_commands(event):
            if _command_sha256(command) in blocked_hashes:
                continue
            for rule in command_policy_rules(command):
                violations.append(CommandPolicyViolation(
                    events_file=path,
                    line=line_number,
                    rule=rule,
                    command=command,
                ))
        for tool_name in _iter_disallowed_tools(event):
            violations.append(CommandPolicyViolation(
                events_file=path,
                line=line_number,
                rule="subagent-delegation",
                command=f"tool={tool_name}",
            ))
    return tuple(violations)


def _iter_shell_commands(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        if value.get("type") == "tool_use":
            name = str(value.get("name") or "").casefold()
            tool_input = value.get("input")
            if name in _SHELL_TOOLS and isinstance(tool_input, Mapping):
                command = tool_input.get("command") or tool_input.get("cmd")
                if isinstance(command, str) and command.strip():
                    yield command
        for child in value.values():
            yield from _iter_shell_commands(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_shell_commands(child)


def _iter_disallowed_tools(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        if value.get("type") == "tool_use":
            name = str(value.get("name") or "")
            if name.casefold() in {"agent", "task"}:
                yield name
        for child in value.values():
            yield from _iter_disallowed_tools(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_disallowed_tools(child)


def command_policy_rules(command: str) -> Tuple[str, ...]:
    """Return rules that must block one proposed B-stage shell command."""
    rules = []
    if re.search(r"(?:^|[^\w-])(?:pkill|killall)(?=\s|$)", command):
        rules.append("global-process-kill")
    if re.search(r"\bpgrep\s+(?:-[^\s]*f\b|--full\b)", command):
        rules.append("global-process-selection")
    if _DIRECT_BUILD_RE.search(command):
        rules.append("bypass-system-build-sh")
    if _BACKGROUND_TEST_RE.search(command):
        rules.append("background-test-sh")
    if _KILL_RE.search(command) and not _all_kills_are_owned(command):
        rules.append("unowned-process-kill")
    return tuple(dict.fromkeys(rules))


def evaluate_pre_tool_use(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate a Claude Code PreToolUse payload without executing anything."""
    tool_name = str(payload.get("tool_name") or payload.get("name") or "")
    tool_input = payload.get("tool_input") or payload.get("input") or {}
    command = ""
    if tool_name.casefold() in _SHELL_TOOLS and isinstance(tool_input, Mapping):
        raw = tool_input.get("command") or tool_input.get("cmd")
        command = raw if isinstance(raw, str) else ""
    rules = command_policy_rules(command) if command.strip() else ()
    return {
        "allowed": not rules,
        "tool_name": tool_name,
        "command": command,
        "command_sha256": _command_sha256(command),
        "rules": list(rules),
        "message": _preflight_message(rules),
    }


def _preflight_message(rules: Tuple[str, ...]) -> str:
    guidance = {
        "bypass-system-build-sh": "Use `bash build.sh` for every build.",
        "global-process-kill": "Do not use pkill/killall; terminate only a child PID captured from `$!`.",
        "global-process-selection": "Do not select global processes with `pgrep -f`.",
        "background-test-sh": "Run `bash test.sh` in the foreground.",
        "unowned-process-kill": (
            "Start the server and capture `server_pid=$!`, then kill and wait for "
            "that variable in the same shell command."
        ),
    }
    return " ".join(guidance.get(rule, rule) for rule in rules)


def _attempt_event_files(
    logs_dir: Path, agent_name: str, *, final_attempt_only: bool
) -> Tuple[Path, ...]:
    files = list(logs_dir.glob(f"{agent_name}.attempt*.events.jsonl"))
    files.sort(key=_attempt_number)
    if not files or not final_attempt_only:
        return tuple(files)
    status_path = logs_dir / f"{agent_name}.status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = {}
    # SubAgentManager writes ``attempt`` today. Accept the older/plural
    # spelling as well so task state remains readable across upgrades.
    final_attempt = status.get("attempt")
    if not isinstance(final_attempt, int):
        final_attempt = status.get("attempts")
    if isinstance(final_attempt, int):
        selected = [path for path in files if _attempt_number(path) == final_attempt]
        if selected:
            return tuple(selected)
    return (files[-1],)


def _attempt_number(path: Path) -> int:
    match = re.search(r"\.attempt(\d+)\.events\.jsonl$", path.name)
    return int(match.group(1)) if match else -1


def _blocked_command_hashes(logs_dir: Path, agent_name: str) -> frozenset[str]:
    path = logs_dir / f"{agent_name}.policy-denials.jsonl"
    hashes = set()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return frozenset()
    for raw in lines:
        try:
            item = json.loads(raw)
        except ValueError:
            continue
        digest = item.get("command_sha256") if isinstance(item, Mapping) else None
        if isinstance(digest, str) and digest:
            hashes.add(digest)
    return frozenset(hashes)


def _command_sha256(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def _all_kills_are_owned(command: str) -> bool:
    """Allow every kill only when all targets came from this shell's ``$!``.

    This is deliberately narrower than trying to reconstruct PID ownership
    across independent Bash tool calls.  A safe smoke test can start the
    server, capture ``server_pid=$!``, install a trap, probe it, terminate it,
    and wait for it in one command.
    """
    owned = set(_OWNED_PID_RE.findall(command))
    if not owned:
        return False
    matches = tuple(_KILL_RE.finditer(command))
    if not matches:
        return True
    for match in matches:
        target_segment = re.split(
            r"[;&|)'\n]", command[match.end():], maxsplit=1
        )[0]
        references = {
            first or second
            for first, second in _VARIABLE_REF_RE.findall(target_segment)
        }
        if not references or not references.issubset(owned):
            return False

        remainder = re.sub(
            r"^\s*(?:(?:-[A-Za-z0-9]+|--signal\s+\S+)\s+)?",
            "",
            target_segment,
        )
        for variable in references:
            remainder = re.sub(
                rf"[\"']?\$(?:{re.escape(variable)}|\{{{re.escape(variable)}\}})[\"']?",
                "",
                remainder,
            )
        remainder = re.sub(r"\s*\d*(?:>>?|<)\S+", "", remainder)
        if remainder.strip():
            return False
    return True


__all__ = [
    "CommandPolicyViolation",
    "command_policy_rules",
    "evaluate_pre_tool_use",
    "validate_code_writer_commands",
    "validate_implementer_commands",
]
