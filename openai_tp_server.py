from __future__ import annotations

import argparse
import json
import os
import threading
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch.distributed as dist

from llm_engine import LLMEngine


# 终端1： TP_SIZE=4 PORT=9000 bash /data/whl-test/meta-infer/start_tp_infer_service.sh dsv2
# 终端2： export PYTHONPATH=/workspace/vllm-v0.15.1-dev:$PYTHONPATH
# PORT=9000 NUM_PROMPTS=50 REQUEST_RATE=1 MAX_CONCURRENCY=1 bash /data/whl-test/meta-infer/run_myengine_benchmark.sh dsv2

@dataclass
class GenerateTask:
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float | None


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


def broadcast_obj(obj: dict[str, Any]) -> dict[str, Any]:
    if not dist_ready():
        return obj
    payload = [obj if is_rank0() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def run_tp_generation_loop(
    engine: LLMEngine, host: str, port: int, max_new_tokens_cap: int
) -> None:
    # Important: for TP collectives, all ranks must execute in strict same order.
    # If HTTP requests are handled concurrently, collective order can diverge
    # across ranks and cause NCCL/RCCL timeouts. Serialize request execution.
    request_lock = threading.Lock()
    rank_logged_once = False

    def generate_once(task: GenerateTask) -> str:
        output = engine.generate(
            task.prompt,
            max_new_tokens=task.max_tokens,
            temperature=task.temperature,
            top_p=task.top_p,
        )
        if isinstance(output, list):
            return output[0]
        return output

    if dist_ready() and not is_rank0():
        while True:
            cmd = broadcast_obj({})
            action = cmd.get("action")
            if action == "shutdown":
                break
            if action != "generate":
                raise RuntimeError(f"Unknown action: {action}")
            if not rank_logged_once:
                rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
                print(f"[meta-infer][rank={rank}] received first generate broadcast")
                rank_logged_once = True
            task = GenerateTask(
                prompt=str(cmd["prompt"]),
                max_tokens=int(cmd["max_tokens"]),
                temperature=float(cmd["temperature"]),
                top_p=None if cmd.get("top_p") is None else float(cmd["top_p"]),
            )
            _ = generate_once(task)
        return

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
            # Guardrail: avoid unexpectedly long decode from client-side params.
            if max_tokens > max_new_tokens_cap:
                max_tokens = max_new_tokens_cap
            temperature = float(req.get("temperature", 0.0))
            top_p_raw = req.get("top_p")
            top_p = None if top_p_raw is None else float(top_p_raw)
            stream = bool(req.get("stream", False))
            model_name = str(req.get("model", "meta-infer-tp"))
            req_id = str(req.get("request_id", f"cmpl-{uuid.uuid4().hex[:10]}"))

            with request_lock:
                cmd = {
                    "action": "generate",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                if dist_ready():
                    _ = broadcast_obj(cmd)
                text = generate_once(
                    GenerateTask(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                )
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
                    "id": req_id,
                    "object": "text_completion",
                    "model": model_name,
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                }
                usage = {
                    "id": req_id,
                    "object": "text_completion",
                    "model": model_name,
                    "choices": [],
                    "usage": {"completion_tokens": completion_tokens},
                }
                self.wfile.write(
                    f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                )
                self.wfile.write(
                    f"data: {json.dumps(usage, ensure_ascii=False)}\n\n".encode("utf-8")
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            body = {
                "id": req_id,
                "object": "text_completion",
                "model": model_name,
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
            _ = broadcast_obj({"action": "shutdown"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a minimal OpenAI-compatible service for meta-infer TP engine."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Model path, e.g. /data/xinference/cache/Qwen3-8B",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="tp",
        choices=["tp", "qwen_tp", "deepseek_tp", "hf"],
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-new-tokens-cap",
        type=int,
        default=512,
        help="Upper bound for per-request max_tokens.",
    )
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
    run_tp_generation_loop(
        engine,
        host=args.host,
        port=args.port,
        max_new_tokens_cap=args.max_new_tokens_cap,
    )


if __name__ == "__main__":
    main()
