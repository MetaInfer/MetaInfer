from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch.distributed as dist

from llm_engine import LLMEngine


@dataclass
class PendingRequest:
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float | None
    result_event: threading.Event = field(default_factory=threading.Event)
    result_text: str = ""


def is_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def init_dist_if_needed() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and (not dist.is_initialized()):
        dist.init_process_group(backend="nccl")


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def broadcast_obj(obj: Any) -> Any:
    if not dist_ready():
        return obj
    payload = [obj if is_rank0() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def run_tp_generation_loop(
    engine: LLMEngine, host: str, port: int, max_new_tokens_cap: int
) -> None:
    request_queue: queue.Queue[PendingRequest] = queue.Queue()

    # --- Non-rank-0: wait for batch commands from rank 0 ---
    if dist_ready() and not is_rank0():
        while True:
            cmd = broadcast_obj(None)
            action = cmd.get("action")
            if action == "shutdown":
                break
            if action == "idle":
                continue
            if action != "batch":
                raise RuntimeError(f"Unknown action: {action}")
            prompts = cmd["prompts"]
            if prompts:
                engine._enqueue(
                    prompts,
                    max_new_tokens=cmd["max_tokens"],
                    temperature=cmd["temperature"],
                    top_p=cmd.get("top_p"),
                )
            num_steps = cmd["num_steps"]
            temperature = cmd["temperature"]
            top_p = cmd.get("top_p")
            for _ in range(num_steps):
                engine.step(temperature=temperature, top_p=top_p)
        return

    # --- Rank-0: HTTP server + batch-based continuous batching ---
    def worker_loop() -> None:
        """Continuous batching worker: collect requests, process batch, repeat."""
        while True:
            # Wait for at least one request with short timeout
            try:
                first = request_queue.get(timeout=0.1)
            except queue.Empty:
                if dist_ready():
                    broadcast_obj({"action": "idle"})
                continue

            # Drain queue to collect more requests (non-blocking)
            batch_reqs = [first]
            max_batch = engine.scheduler.max_num_seqs
            while len(batch_reqs) < max_batch:
                try:
                    batch_reqs.append(request_queue.get_nowait())
                except queue.Empty:
                    break

            # Enqueue into engine
            prompts = [r.prompt for r in batch_reqs]
            max_tokens = batch_reqs[0].max_tokens
            temperature = batch_reqs[0].temperature
            top_p = batch_reqs[0].top_p
            seqs = engine._enqueue(
                prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            req_map = {seq.request_id: req for seq, req in zip(seqs, batch_reqs)}

            # Steps: prefill all seqs (one per step) + decode (max_tokens shared)
            num_steps = len(seqs) + max_tokens

            # Broadcast batch to non-rank-0
            if dist_ready():
                cmd = {
                    "action": "batch",
                    "prompts": prompts,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_steps": num_steps,
                }
                broadcast_obj(cmd)

            # Step exactly num_steps times (must match non-rank-0 for TP sync)
            for _ in range(num_steps):
                finished_seqs = engine.step(temperature=temperature, top_p=top_p)
                for seq in finished_seqs:
                    req = req_map.pop(seq.request_id, None)
                    if req is not None:
                        text = engine.runner.tokenizer.decode(
                            seq.output_ids, skip_special_tokens=True
                        )
                        req.result_text = text
                        req.result_event.set()

    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    class Handler(BaseHTTPRequestHandler):
        server_version = "MetaInferOpenAI/0.1"

        def _json_response(self, code: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json_response(HTTPStatus.OK, {"status": "ok"})
                return
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/completions":
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                req = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
                return

            prompt = req.get("prompt", "")
            if not isinstance(prompt, str):
                self._json_response(
                    HTTPStatus.BAD_REQUEST, {"error": "only string prompt is supported"}
                )
                return

            max_tokens = int(req.get("max_tokens", 16))
            if max_tokens > max_new_tokens_cap:
                max_tokens = max_new_tokens_cap
            temperature = float(req.get("temperature", 0.0))
            top_p_raw = req.get("top_p")
            top_p = None if top_p_raw is None else float(top_p_raw)
            stream = bool(req.get("stream", False))
            model_name = str(req.get("model", "meta-infer-tp"))
            req_id = str(req.get("request_id", f"cmpl-{uuid.uuid4().hex[:10]}"))

            pending = PendingRequest(
                prompt=prompt, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p,
            )
            request_queue.put(pending)
            pending.result_event.wait()

            text = pending.result_text
            completion_tokens = len(
                engine.runner.tokenizer.encode(text, add_special_tokens=False)
            )

            if stream:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                chunk = {
                    "id": req_id, "object": "text_completion", "model": model_name,
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                }
                usage = {
                    "id": req_id, "object": "text_completion", "model": model_name,
                    "choices": [],
                    "usage": {"completion_tokens": completion_tokens},
                }
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.write(f"data: {json.dumps(usage, ensure_ascii=False)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            body = {
                "id": req_id, "object": "text_completion", "model": model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": {"completion_tokens": completion_tokens},
            }
            self._json_response(HTTPStatus.OK, body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[meta-infer] OpenAI-like TP server listening at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if dist_ready():
            broadcast_obj({"action": "shutdown"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a minimal OpenAI-compatible service for meta-infer TP engine."
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--backend", type=str, default="tp",
                        choices=["tp", "qwen_tp", "deepseek_tp", "hf"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens-cap", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_dist_if_needed()
    engine = LLMEngine(
        model_dir=Path(args.model_dir),
        inference_backend=args.backend,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    run_tp_generation_loop(engine, host=args.host, port=args.port,
                           max_new_tokens_cap=args.max_new_tokens_cap)


if __name__ == "__main__":
    main()
