"""OpenAI 兼容 API 服务 (QwenPaw-Flash-2B)。

用法:
    uv run python -m engine.20260507.mac_qwen.openai_server \\
        --model-dir ~/.cache/modelscope/hub/models/AgentScope/QwenPaw-Flash-2B \\
        --port 8000

测试:
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"qwen","messages":[{"role":"user","content":"你好"}]}'
"""

from __future__ import annotations

import argparse
import importlib
import json
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch

from engine.sampler import sample_next_tokens
from engine.structs import Sequence

_model_mod = importlib.import_module("engine.20260507.mac_qwen.model")
Qwen35MoeModelRunner = _model_mod.Qwen35MoeModelRunner
states_to_kwargs = _model_mod.states_to_kwargs
update_states_from_kwargs = _model_mod.update_states_from_kwargs


def _select_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def _json_resp(handler: BaseHTTPRequestHandler, code: int, body: dict[str, Any]) -> None:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _read_json(handler: BaseHTTPRequestHandler) -> dict | None:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        _json_resp(handler, HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
        return None


def _sse_write(handler: BaseHTTPRequestHandler, data: str) -> None:
    handler.wfile.write(f"data: {data}\n\n".encode("utf-8"))
    handler.wfile.flush()


class QwenOpenAIServer:
    def __init__(
        self,
        model_dir: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_new_tokens_cap: int = 2048,
    ):
        self.host = host
        self.port = port
        self.max_new_tokens_cap = max_new_tokens_cap
        self._lock = threading.Lock()

        device, dtype = _select_device()
        print(f"[Server] device={device}, dtype={dtype}")

        self.runner = Qwen35MoeModelRunner(
            model_dir=model_dir,
            device=device,
            dtype=dtype,
            max_seq_len=8192,
        )
        self.tokenizer = self.runner.tokenizer
        self.device = device

    def _generate_tokens(
        self,
        token_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        skip_thinking: bool = False,
    ):
        """Yield (token_text, finish_reason) per step.

        If skip_thinking=True, suppress output until </think\> is seen
        (Qwen3.5 thinking mode: model first generates <think\>...thinking...</think\> then answer).
        """
        seq = Sequence(
            request_id=f"req-{uuid.uuid4().hex[:8]}",
            input_ids=token_ids,
            sampling_params={},
        )
        self.runner.init_sequence(seq.request_id)
        states = self.runner._seq_states[seq.request_id]
        dev = self.device

        thinking_done = not skip_thinking

        # Prefill
        ids_t = torch.tensor([seq.token_ids], dtype=torch.long, device=dev)
        pos_t = torch.arange(seq.total_tokens, device=dev, dtype=torch.long).unsqueeze(0)
        kw_list = states_to_kwargs(states)
        for kw in kw_list:
            kw["position_ids"] = pos_t

        logits = self.runner.model(ids_t, kw_list)
        tok = int(
            sample_next_tokens(
                logits[:, -1, :].cpu().float(), temperature=temperature, top_p=top_p
            ).item()
        )
        seq.append_token(tok)
        update_states_from_kwargs(states, kw_list)

        if skip_thinking and tok == self._THINK_END:
            thinking_done = True
        elif thinking_done:
            yield self.tokenizer.decode([tok], skip_special_tokens=True), None

        for _ in range(1, max_tokens):
            dec_ids = torch.tensor([[seq.output_ids[-1]]], dtype=torch.long, device=dev)
            cur_pos = seq.total_tokens - 1
            kw_list = states_to_kwargs(states)
            for kw in kw_list:
                kw["position_ids"] = torch.tensor([[cur_pos]], device=dev, dtype=torch.long)

            logits = self.runner.model(dec_ids, kw_list)
            tok = int(
                sample_next_tokens(
                    logits[:, -1, :].cpu().float(), temperature=temperature, top_p=top_p
                ).item()
            )
            seq.append_token(tok)
            update_states_from_kwargs(states, kw_list)

            reason = None
            if tok == self.runner.eos_token_id:
                reason = "stop"
            elif seq.total_tokens >= self.runner.max_seq_len:
                reason = "length"

            if not thinking_done:
                if tok == self._THINK_END:
                    thinking_done = True
                # Don't yield anything during thinking phase
            else:
                yield self.tokenizer.decode([tok], skip_special_tokens=True), reason

            if reason is not None:
                break

        self.runner.cleanup_sequence(seq.request_id)

    # ── Chat Completions ─────────────────────────────────────────────────

    # Token IDs injected by chat template for thinking mode
    _THINK_START = 248068  # <think\>
    _THINK_END = 248069  # </think\>

    def _handle_chat(self, req: dict, h: BaseHTTPRequestHandler) -> None:
        messages = req.get("messages", [])
        model_name = str(req.get("model", "qwen"))
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        max_tokens = min(int(req.get("max_tokens", 512)), self.max_new_tokens_cap)
        temperature = float(req.get("temperature", 0.6))
        top_p = float(req.get("top_p", 0.95))
        stream = bool(req.get("stream", False))

        result = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        token_ids = result["input_ids"]
        # Prompt contains <think\></think\> at the end, which signals the model
        # that thinking is done. The model will answer directly without thinking.

        with self._lock:
            gen = self._generate_tokens(token_ids, max_tokens, temperature, top_p)
            if stream:
                self._stream_chat(h, req_id, model_name, gen)
            else:
                self._batch_chat(h, req_id, model_name, gen)

    def _stream_chat(self, h, req_id, model_name, gen):
        h.send_response(HTTPStatus.OK)
        h.send_header("Content-Type", "text/event-stream; charset=utf-8")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Connection", "close")
        h.end_headers()

        n_compl = 0
        for text, reason in gen:
            n_compl += 1
            delta = {"role": "assistant", "content": text} if n_compl == 1 else {"content": text}
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
            }
            _sse_write(h, json.dumps(chunk, ensure_ascii=False))

        final = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        _sse_write(h, json.dumps(final, ensure_ascii=False))
        h.wfile.write(b"data: [DONE]\n\n")
        h.wfile.flush()

    def _batch_chat(self, h, req_id, model_name, gen):
        parts: list[str] = []
        finish = "stop"
        for text, reason in gen:
            parts.append(text)
            if reason is not None:
                finish = reason
        content = "".join(parts)
        compl_tokens = len(self.tokenizer.encode(content, add_special_tokens=False))
        prompt_tokens = 0  # not available from generator

        body = {
            "id": req_id,
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": compl_tokens,
                "total_tokens": prompt_tokens + compl_tokens,
            },
        }
        _json_resp(h, HTTPStatus.OK, body)

    # ── Text Completions ─────────────────────────────────────────────────

    def _handle_completions(self, req: dict, h: BaseHTTPRequestHandler) -> None:
        prompt = str(req.get("prompt", ""))
        model_name = str(req.get("model", "qwen"))
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        max_tokens = min(int(req.get("max_tokens", 16)), self.max_new_tokens_cap)
        temperature = float(req.get("temperature", 0.6))
        top_p = float(req.get("top_p", 0.95))
        stream = bool(req.get("stream", False))

        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        with self._lock:
            if stream:
                self._stream_compl(h, req_id, model_name, token_ids, max_tokens, temperature, top_p)
            else:
                self._batch_compl(h, req_id, model_name, token_ids, max_tokens, temperature, top_p)

    def _stream_compl(self, h, req_id, model_name, token_ids, max_tokens, temp, top_p):
        h.send_response(HTTPStatus.OK)
        h.send_header("Content-Type", "text/event-stream; charset=utf-8")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Connection", "close")
        h.end_headers()

        for text, reason in self._generate_tokens(token_ids, max_tokens, temp, top_p):
            chunk = {
                "id": req_id,
                "object": "text_completion",
                "model": model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": reason}],
            }
            _sse_write(h, json.dumps(chunk, ensure_ascii=False))

        h.wfile.write(b"data: [DONE]\n\n")
        h.wfile.flush()

    def _batch_compl(self, h, req_id, model_name, token_ids, max_tokens, temp, top_p):
        parts: list[str] = []
        finish = "stop"
        for text, reason in self._generate_tokens(token_ids, max_tokens, temp, top_p):
            parts.append(text)
            if reason is not None:
                finish = reason
        content = "".join(parts)
        prompt_tokens = len(token_ids)
        compl_tokens = len(self.tokenizer.encode(content, add_special_tokens=False))

        body = {
            "id": req_id,
            "object": "text_completion",
            "model": model_name,
            "choices": [{"index": 0, "text": content, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": compl_tokens,
                "total_tokens": prompt_tokens + compl_tokens,
            },
        }
        _json_resp(h, HTTPStatus.OK, body)

    # ── HTTP Server ───────────────────────────────────────────────────────

    def run(self) -> None:
        srv = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "QwenOpenAI/0.1"

            def do_GET(self):
                if self.path == "/health":
                    _json_resp(self, HTTPStatus.OK, {"status": "ok"})
                elif self.path == "/v1/models":
                    _json_resp(
                        self,
                        HTTPStatus.OK,
                        {
                            "object": "list",
                            "data": [{"id": "qwen", "object": "model", "owned_by": "local"}],
                        },
                    )
                else:
                    _json_resp(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self):
                body = _read_json(self)
                if body is None:
                    return
                if self.path == "/v1/chat/completions":
                    srv._handle_chat(body, self)
                elif self.path == "/v1/completions":
                    srv._handle_completions(body, self)
                else:
                    _json_resp(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

            def log_message(self, fmt, *args):
                return

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        print(f"[Server] listening at http://{self.host}:{self.port}")
        print("[Server] endpoints: /v1/chat/completions, /v1/completions, /health")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[Server] shutting down")
        finally:
            server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 兼容 API 服务 (QwenPaw-Flash-2B)")
    parser.add_argument("--model-dir", required=True, help="模型目录路径")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens-cap", type=int, default=2048)
    args = parser.parse_args()

    server = QwenOpenAIServer(
        model_dir=args.model_dir,
        host=args.host,
        port=args.port,
        max_new_tokens_cap=args.max_new_tokens_cap,
    )
    server.run()


if __name__ == "__main__":
    main()
