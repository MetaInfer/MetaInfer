"""
Phase 10 — OpenAI TP Server: HTTP API for LLMEngine with TP multi-rank sync.

Architecture:
  - ThreadingHTTPServer + BaseHTTPRequestHandler
  - Endpoints: GET /health, POST /v1/completions
  - TP sync: broadcast_obj distributes commands to all ranks
  - threading.Lock serializes all requests (NCCL collective order)
  - Streaming (SSE) + non-streaming (JSON)

Blueprint: inference_blueprint.json > OpenAITPServer
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from llm_engine import LLMEngine


# ===========================================================================
# Signal handler — prevent GPU VRAM leak on shutdown (blueprint past_bug_20260606)
# ===========================================================================

def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT: os._exit(0) to avoid NCCL destructor hangs."""
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    if rank != 0:
        os._exit(0)
    else:
        # rank0: let finally block run to signal shutdown to workers
        raise KeyboardInterrupt


# ===========================================================================
# TP Command Protocol
# ===========================================================================

def _broadcast_command(rank: int, payload: dict) -> dict:
    """Broadcast a command dict from rank0 to all ranks, return the same dict.

    Uses dist.broadcast_object_list with a single-element list.
    All ranks receive the same payload.
    """
    buf = [payload if rank == 0 else None]
    dist.broadcast_object_list(buf, src=0)
    return buf[0]


# ===========================================================================
# HTTP Request Handler
# ===========================================================================

class TPInferRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler with TP sync and SSE streaming support.

    Critical constraints (from blueprint streaming.critical_constraints):
      - Connection: close on every response (NOT keep-alive)
      - self.close_connection = True set on ALL code paths
      - SSE final chunk: data: [DONE]\\n\\n
    """

    # Class-level engine and config — set by run_tp_server()
    engine: LLMEngine = None
    engine_lock: threading.Lock = None
    rank: int = 0

    def _set_close(self) -> None:
        """Ensure connection closes — set on every path per blueprint."""
        self.close_connection = True

    def log_message(self, fmt, *args):
        """Suppress default stderr logging."""
        pass

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    def do_GET(self):
        if self.path == "/health":
            self._set_close()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            resp = json.dumps({"status": "ok", "rank": self.rank})
            self.wfile.write(resp.encode())
        else:
            self._set_close()
            self.send_response(404)
            self.send_header("Connection", "close")
            self.end_headers()

    # ------------------------------------------------------------------
    # POST /v1/completions
    # ------------------------------------------------------------------

    def do_POST(self):
        if self.path != "/v1/completions":
            self._set_close()
            self.send_response(404)
            self.send_header("Connection", "close")
            self.end_headers()
            return

        # Parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._set_close()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            return

        prompt = req.get("prompt", "")
        max_tokens = req.get("max_tokens", 256)
        temperature = req.get("temperature", 0.0)
        top_p = req.get("top_p", None)
        stream = req.get("stream", False)

        # Broadcast command to all TP ranks (all ranks execute the same generation)
        cmd = {
            "action": "generate",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        _broadcast_command(self.rank, cmd)

        # Serialize all requests through a lock (NCCL collective ordering)
        with self.engine_lock:
            if stream:
                self._handle_stream(prompt, max_tokens, temperature, top_p)
            else:
                self._handle_non_stream(prompt, max_tokens, temperature, top_p)

    # ------------------------------------------------------------------
    # Non-streaming response (JSON)
    # ------------------------------------------------------------------

    def _handle_non_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
    ) -> None:
        try:
            generated = self.engine.generate(
                prompts=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception:
            self._set_close()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Generation failed"}).encode())
            return

        resp = {
            "model": "qwen_tp",
            "object": "text_completion",
            "choices": [{"text": generated, "index": 0, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(generated),
                "total_tokens": 0,
            },
        }
        self._set_close()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(json.dumps(resp, ensure_ascii=False).encode())

    # ------------------------------------------------------------------
    # Streaming response (SSE)
    # ------------------------------------------------------------------

    def _handle_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
    ) -> None:
        try:
            self._set_close()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # Use step-by-step API for streaming
            seqs = self.engine._enqueue(
                [prompt], max_tokens, temperature, top_p
            )
            self.engine.begin_generation(seqs)

            prev_len = 0
            while self.engine.has_unfinished_requests():
                self.engine.step(temperature, top_p)

                # Emit new tokens for each active sequence
                for seq in seqs:
                    new_tokens = seq.output_ids[prev_len:]
                    for token_id in new_tokens:
                        token_text = self.engine.runner.tokenizer.decode(
                            [token_id], skip_special_tokens=True
                        )
                        chunk = json.dumps(
                            {"choices": [{"text": token_text, "index": 0}]},
                            ensure_ascii=False,
                        )
                        self.wfile.write(f"data: {chunk}\n\n".encode())
                        self.wfile.flush()
                    prev_len = len(seq.output_ids)

            # Final SSE chunk with finish_reason
            finish_chunk = json.dumps(
                {"choices": [{"text": "", "index": 0, "finish_reason": "stop"}]},
                ensure_ascii=False,
            )
            self.wfile.write(f"data: {finish_chunk}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        except Exception:
            # On error, still close connection properly
            self._set_close()
            try:
                self.send_response(500)
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass


# ===========================================================================
# Server entry points
# ===========================================================================

def _get_rank() -> int:
    """Get distributed rank, default 0."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def run_tp_generation_loop(
    model_dir: str | Path,
    backend: str = "qwen_tp",
    host: str = "0.0.0.0",
    port: int = 9000,
    tp_size: int = 1,
    max_num_seqs: int = 4,
    max_new_tokens_cap: int = 2048,
) -> None:
    """Start the HTTP server on rank0, TP worker loop on non-rank0.

    Rank0: creates LLMEngine → starts ThreadingHTTPServer → handles requests.
           Each request is serialized by engine_lock and broadcast to all ranks.
    Non-rank0: runs worker loop waiting for broadcast commands.

    Args:
        model_dir: Path to model weights directory.
        backend: Inference backend ('qwen_tp' or 'deepseek_tp').
        host: Bind address for HTTP server.
        port: Port for HTTP server.
        tp_size: Tensor parallelism size.
        max_num_seqs: Maximum concurrent sequences.
        max_new_tokens_cap: Cap on max_new_tokens per request.
    """
    # Initialize TP distributed (idempotent — guarded internally)
    from engine.tp_layers.distributed import init_tp_distributed
    init_tp_distributed()

    # Register signal handlers (blueprint past_bug_20260606: prevent GPU VRAM leak)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    rank = _get_rank()

    # Create engine (all ranks)
    engine = LLMEngine(
        model_dir=Path(model_dir),
        tp_size=tp_size,
        inference_backend="qwen_tp",
        max_num_seqs=4,
    )
    lock = threading.Lock()

    if rank == 0:
        # Configure handler
        TPInferRequestHandler.engine = engine
        TPInferRequestHandler.engine_lock = lock
        TPInferRequestHandler.rank = rank

        server = ThreadingHTTPServer(("0.0.0.0", port), TPInferRequestHandler)
        print(f"[TP Server] Listening on port {port} (TP={tp_size})", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            # Signal shutdown to non-rank0 workers
            shutdown_cmd = {"action": "shutdown"}
            _broadcast_command(rank, shutdown_cmd)
            server.server_close()
    else:
        # Non-rank0 worker loop
        _tp_worker_loop(engine, lock)


def _tp_worker_loop(engine: LLMEngine, lock: threading.Lock) -> None:
    """Non-rank0 loop: wait for commands, execute the same generation."""
    rank = _get_rank()
    while True:
        cmd = _broadcast_command(rank, {})
        action = cmd.get("action", "")

        if action == "shutdown":
            break

        if action == "generate":
            prompt = cmd["prompt"]
            max_tokens = cmd["max_tokens"]
            temperature = cmd["temperature"]
            top_p = cmd.get("top_p")

            with lock:
                if cmd.get("stream"):
                    # Streaming: run step-by-step loop
                    seqs = engine._enqueue(
                        [prompt], max_tokens, temperature, top_p
                    )
                    engine.begin_generation(seqs)
                    while engine.has_unfinished_requests():
                        engine.step(temperature, top_p)
                else:
                    # Non-streaming: single generate call
                    engine.generate(
                        prompts=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI-compatible TP inference server")
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", ""),
                        help="Model weights directory (or set MODEL_DIR env var)")
    parser.add_argument("--backend", default="qwen_tp",
                        choices=["qwen_tp", "deepseek_tp"],
                        help="Inference backend (default: qwen_tp)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "9000")),
                        help="Port to listen on (default: 9000)")
    parser.add_argument("--tp-size", type=int,
                        default=int(os.environ.get("TP_SIZE", os.environ.get("WORLD_SIZE", "1"))),
                        help="Tensor parallelism size (default: 1)")
    parser.add_argument("--max-num-seqs", type=int, default=4,
                        help="Maximum concurrent sequences (default: 4)")
    parser.add_argument("--max-new-tokens-cap", type=int, default=2048,
                        help="Cap on max_new_tokens per request (default: 2048)")
    args = parser.parse_args()

    if not args.model_dir:
        parser.error("--model-dir is required (or set MODEL_DIR env var)")

    run_tp_generation_loop(
        model_dir=args.model_dir,
        backend=args.backend,
        host=args.host,
        port=args.port,
        tp_size=args.tp_size,
        max_num_seqs=args.max_num_seqs,
        max_new_tokens_cap=args.max_new_tokens_cap,
    )
