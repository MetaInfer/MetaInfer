#!/usr/bin/env python3
"""Container-side executable compatible with SubAgentManager's Claude CLI."""

from __future__ import annotations

import json
import os
import socket
import struct
import sys


SOCKET_PATH = os.environ.get(
    "METAINFER_AGENT_BRIDGE_SOCKET",
    "/workspace/MetaInfer/.metainfer-agent-bridge.sock",
)
FORWARDED_ENV = (
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "TORCH_EXTENSIONS_DIR",
    "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR",
    "DISABLE_INTERACTIVITY",
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


def main() -> int:
    prompt = sys.stdin.buffer.read()
    request = json.dumps({
        "args": sys.argv[1:],
        "cwd": os.getcwd(),
        "task_id": os.environ.get("METAINFER_TASK_ID"),
        "env": {
            key: os.environ[key] for key in FORWARDED_ENV
            if key in os.environ
        },
    }).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(SOCKET_PATH)
        conn.sendall(struct.pack("!I", len(request)) + request)
        conn.sendall(struct.pack("!I", len(prompt)) + prompt)
        while True:
            kind = conn.recv(1)
            if not kind:
                return 1
            size = struct.unpack("!I", _recv_exact(conn, 4))[0]
            payload = _recv_exact(conn, size)
            if kind == b"O":
                sys.stdout.buffer.write(payload)
                sys.stdout.buffer.flush()
            elif kind == b"E":
                return struct.unpack("!i", payload)[0]
            else:
                sys.stderr.buffer.write(payload + b"\n")
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
