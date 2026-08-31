#!/usr/bin/env python3
"""Host-side, credential-preserving Claude CLI bridge over a Unix socket."""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import struct
import subprocess
import threading
from pathlib import Path


_DEFAULT_META_ROOT = Path(__file__).resolve().parents[4]
HOST_ROOT = Path(os.environ.get(
    "METAINFER_AGENT_BRIDGE_ROOT",
    str(_DEFAULT_META_ROOT),
)).resolve()
HOST_WORKSPACE_ROOT = Path(os.environ.get(
    "METAINFER_AGENT_BRIDGE_WORKSPACE_ROOT",
    str(HOST_ROOT.parent),
)).resolve()
SOCKET_PATH = Path(os.environ.get(
    "METAINFER_AGENT_BRIDGE_SOCKET",
    str(HOST_ROOT / ".metainfer-agent-bridge.sock"),
))
ALLOWED_ROOTS = (
    HOST_ROOT,
    (HOST_WORKSPACE_ROOT / "kernel-repos").resolve(),
    (HOST_WORKSPACE_ROOT / "API").resolve(),
)
CLAUDE_BIN = os.environ.get(
    "METAINFER_HOST_CLAUDE_BIN",
    "/usr/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe",
)
ALLOWED_ENV = {
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "TORCH_EXTENSIONS_DIR",
    "TRITON_CACHE_DIR",
    "XDG_CACHE_HOME",
    "TMPDIR",
    "DISABLE_INTERACTIVITY",
}
PATH_ENV = {
    "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR",
}
VALUE_FLAGS = {
    "--output-format", "--input-format", "--permission-mode", "--add-dir",
    "--model", "--effort", "--resume", "--session-id",
    "--disallowedTools", "--disallowed-tools",
}
BARE_FLAGS = {"-p", "--verbose"}
SOURCE_ONLY_TOOLS = "Bash,Skill,WebFetch,WebSearch"
CLI_SLOT = threading.Semaphore(int(os.environ.get(
    # worker29 has four physical GPUs and the control plane assigns at most
    # one child per GPU.  A single slot serializes independent workers and
    # makes their absolute bootstrap timeout include queueing behind peers.
    "METAINFER_AGENT_BRIDGE_CONCURRENCY", "4"
)))
TASK_LOCK = threading.Lock()
TASK_CONNECTIONS: dict[str, set[socket.socket]] = {}
TASK_PROCESSES: dict[str, set[subprocess.Popen[bytes]]] = {}
TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise EOFError("bridge request ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _frame(conn: socket.socket, kind: bytes, payload: bytes) -> None:
    conn.sendall(kind + struct.pack("!I", len(payload)) + payload)


def _validated_task_id(value: object) -> str | None:
    if value is None:
        return None
    task_id = str(value)
    if not TASK_ID_RE.fullmatch(task_id):
        raise ValueError(f"invalid task id: {task_id!r}")
    return task_id


def _track_connection(task_id: str, conn: socket.socket) -> None:
    with TASK_LOCK:
        TASK_CONNECTIONS.setdefault(task_id, set()).add(conn)


def _track_process(
    task_id: str, process: subprocess.Popen[bytes],
) -> None:
    with TASK_LOCK:
        TASK_PROCESSES.setdefault(task_id, set()).add(process)


def _untrack_task_resources(
    task_id: str | None,
    conn: socket.socket,
    process: subprocess.Popen[bytes] | None,
) -> None:
    if task_id is None:
        return
    with TASK_LOCK:
        connections = TASK_CONNECTIONS.get(task_id)
        if connections is not None:
            connections.discard(conn)
            if not connections:
                TASK_CONNECTIONS.pop(task_id, None)
        if process is not None:
            processes = TASK_PROCESSES.get(task_id)
            if processes is not None:
                processes.discard(process)
                if not processes:
                    TASK_PROCESSES.pop(task_id, None)


def _terminate_process(
    process: subprocess.Popen[bytes], *, force: bool = False,
) -> bool:
    if process.poll() is not None:
        return False
    try:
        os.killpg(
            os.getpgid(process.pid),
            signal.SIGKILL if force else signal.SIGTERM,
        )
        process.wait(timeout=1 if force else 5)
    except ProcessLookupError:
        return False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    return True


def _task_process_groups(
    task_id: str,
    tracked_pids: set[int],
) -> set[int]:
    """Resolve tracked descendants plus exact task-marker process groups."""
    rows: dict[int, tuple[int, int]] = {}
    marker = task_id.encode()
    marker_pids: set[int] = set()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            raw_stat = Path(
                f"/proc/{pid}/stat"
            ).read_text(encoding="utf-8", errors="replace")
            rparen = raw_stat.rfind(")")
            fields = raw_stat[rparen + 2:].split()
            if rparen == -1 or len(fields) < 3:
                continue
            ppid = int(fields[1])
            pgid = os.getpgid(pid)
            rows[pid] = (ppid, pgid)
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
            if marker in cmdline:
                marker_pids.add(pid)
        except (OSError, ValueError, ProcessLookupError):
            continue

    owned = set(tracked_pids) | marker_pids
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in rows.items():
            if ppid in owned and pid not in owned:
                owned.add(pid)
                changed = True
    server_pgid = os.getpgrp()
    return {
        rows[pid][1]
        for pid in owned
        if pid in rows and rows[pid][1] != server_pgid
    }


def _cancel_task(task_id: str, *, force: bool = False) -> dict[str, int]:
    """Stop host agents and disconnect queued clients owned by one task."""
    with TASK_LOCK:
        processes = list(TASK_PROCESSES.get(task_id, ()))
        connections = list(TASK_CONNECTIONS.get(task_id, ()))
    process_groups = _task_process_groups(
        task_id, {process.pid for process in processes}
    )
    group_signal = signal.SIGKILL if force else signal.SIGTERM
    killed_groups = 0
    for pgid in process_groups:
        try:
            os.killpg(pgid, group_signal)
            killed_groups += 1
        except ProcessLookupError:
            pass
    killed = sum(
        _terminate_process(process, force=force)
        for process in processes
    )
    disconnected = 0
    for client in connections:
        try:
            client.shutdown(socket.SHUT_RDWR)
            disconnected += 1
        except OSError:
            pass
    return {
        "killed_processes": killed,
        "killed_process_groups": killed_groups,
        "disconnected_clients": disconnected,
    }


def _translate_prompt_paths(prompt: bytes) -> bytes:
    """Translate only the /workspace path segment, not /workspaces names."""
    return re.sub(
        rb"/workspace(?=/|$)",
        lambda _: str(HOST_WORKSPACE_ROOT).encode(),
        prompt,
    )


def _host_path(value: str) -> Path:
    if value == "/workspace" or value.startswith("/workspace/"):
        value = str(HOST_WORKSPACE_ROOT) + value[len("/workspace"):]
    path = Path(value).resolve()
    if not any(path == root or root in path.parents for root in ALLOWED_ROOTS):
        raise ValueError(
            f"path outside allowed MetaInfer workspace roots: {path}"
        )
    return path


def _host_agent_cwd(path: Path) -> Path:
    """Use a neutral non-Git cwd for a container-created worktree."""
    git_marker = path / ".git"
    if not git_marker.is_file():
        return path
    # The common repo contains worktree metadata with container-only
    # /workspace paths, while the worker path sits below the large MetaInfer
    # checkout. Both make Claude Code repository discovery stall. The
    # kernel-repos parent is allowed, writable, and is not itself a Git repo.
    return (HOST_WORKSPACE_ROOT / "kernel-repos").resolve()


def _validated_args(args: list[str]) -> list[str]:
    out = []
    index = 0
    while index < len(args):
        flag = args[index]
        if flag in BARE_FLAGS:
            out.append(flag)
            index += 1
            continue
        if flag not in VALUE_FLAGS or index + 1 >= len(args):
            raise ValueError(f"unsupported Claude argument: {flag}")
        value = args[index + 1]
        # Claude Code 2.1.161 on worker29 fails to consume stdin when its
        # default text input mode is repeated explicitly. Omit only this
        # redundant pair; stream-json input remains forwarded unchanged.
        if flag == "--input-format" and value == "text":
            index += 2
            continue
        if flag == "--add-dir":
            add_dir = _host_path(value)
            # Container-created Git worktrees have a .git indirection file
            # containing a /workspace/... gitdir that is invalid on the host.
            # Grant the worker root instead; the source directory remains
            # inside that exact allowed tree and is named in the prompt.
            if (add_dir / ".git").is_file():
                add_dir = add_dir.parent
            value = str(add_dir)
        out.extend([flag, value])
        index += 2
    if "-p" not in out:
        raise ValueError("print mode is required")
    if not any(
        flag in out
        for flag in ("--disallowedTools", "--disallowed-tools")
    ):
        out.extend(["--disallowedTools", SOURCE_ONLY_TOOLS])
    return out


def _handle(conn: socket.socket) -> None:
    slot_acquired = False
    process: subprocess.Popen[bytes] | None = None
    task_id: str | None = None
    try:
        header_size = struct.unpack("!I", _recv_exact(conn, 4))[0]
        if header_size > 128 * 1024:
            raise ValueError("bridge header too large")
        request = json.loads(_recv_exact(conn, header_size))
        prompt_size = struct.unpack("!I", _recv_exact(conn, 4))[0]
        if prompt_size > 128 * 1024:
            raise ValueError("bridge prompt too large")
        prompt = _translate_prompt_paths(_recv_exact(conn, prompt_size))
        if request.get("action") == "cancel_task":
            cancel_id = _validated_task_id(request.get("task_id"))
            if cancel_id is None:
                raise ValueError("cancel_task requires task_id")
            result = json.dumps(_cancel_task(
                cancel_id,
                force=bool(request.get("force", False)),
            )).encode() + b"\n"
            _frame(conn, b"O", result)
            _frame(conn, b"E", struct.pack("!i", 0))
            return
        task_id = _validated_task_id(request.get("task_id"))
        if task_id is not None:
            _track_connection(task_id, conn)
        cwd = _host_path(str(request["cwd"]))
        # Container-created git worktrees store /workspace/... in their
        # .git indirection file. That path does not exist on the host where
        # Claude runs, and Claude Code stalls during repository discovery.
        # Use the worker root as cwd while retaining source via --add-dir.
        cwd = _host_agent_cwd(cwd)
        args = _validated_args([str(v) for v in request.get("args", [])])
        prompt_text = prompt.decode("utf-8", errors="strict")
        print_index = args.index("-p")
        args.insert(print_index + 1, prompt_text)
        overrides = {
            str(key): str(value)
            for key, value in (request.get("env") or {}).items()
            if key in ALLOWED_ENV
        }
        for key in PATH_ENV & overrides.keys():
            overrides[key] = str(_host_path(overrides[key]))
        env = dict(os.environ)
        env.update(overrides)
        env["PWD"] = str(cwd)
        while not CLI_SLOT.acquire(timeout=5):
            heartbeat = json.dumps({
                "type": "system",
                "subtype": "bridge_queued",
                "message": "Waiting for the host Claude CLI slot",
            }).encode() + b"\n"
            _frame(conn, b"O", heartbeat)
        slot_acquired = True
        process = subprocess.Popen(
            [CLAUDE_BIN, *args],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if task_id is not None:
            _track_process(task_id, process)
        assert process.stdout is not None
        for chunk in iter(process.stdout.readline, b""):
            _frame(conn, b"O", chunk)
        returncode = process.wait()
        _frame(conn, b"E", struct.pack("!i", returncode))
    except Exception as exc:
        if process is not None and process.poll() is None:
            try:
                _terminate_process(process)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        try:
            _frame(conn, b"X", str(exc).encode("utf-8", errors="replace"))
        except OSError:
            pass
    finally:
        _untrack_task_resources(task_id, conn, process)
        if slot_acquired:
            CLI_SLOT.release()
        conn.close()


def main() -> int:
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        SOCKET_PATH.unlink()
    except FileNotFoundError:
        pass
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(SOCKET_PATH))
        os.chmod(SOCKET_PATH, 0o660)
        server.listen(16)
        while True:
            conn, _ = server.accept()
            threading.Thread(target=_handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
