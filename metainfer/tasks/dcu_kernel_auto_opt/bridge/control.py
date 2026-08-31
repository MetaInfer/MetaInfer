"""Task-scoped control client for the host Claude bridge."""

from __future__ import annotations

import json
import os
import socket
import struct
from typing import Any, Dict


SOCKET_PATH = os.environ.get(
    "METAINFER_AGENT_BRIDGE_SOCKET",
    "/workspace/MetaInfer/.metainfer-agent-bridge.sock",
)


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        chunk = conn.recv(size)
        if not chunk:
            raise EOFError("agent bridge closed unexpectedly")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def cancel_task(
    task_id: str,
    timeout_s: float = 10.0,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Terminate host agents and queued bridge clients for ``task_id``."""
    request = json.dumps({
        "action": "cancel_task",
        "task_id": task_id,
        "force": force,
    }).encode()
    payload: Dict[str, Any] = {}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout_s)
        conn.connect(SOCKET_PATH)
        conn.sendall(struct.pack("!I", len(request)) + request)
        conn.sendall(struct.pack("!I", 0))
        while True:
            kind = _recv_exact(conn, 1)
            size = struct.unpack("!I", _recv_exact(conn, 4))[0]
            body = _recv_exact(conn, size)
            if kind == b"O":
                payload = json.loads(body)
            elif kind == b"E":
                return {
                    "ok": struct.unpack("!i", body)[0] == 0,
                    **payload,
                }
            else:
                raise RuntimeError(body.decode("utf-8", errors="replace"))
