#!/usr/bin/env python3
"""
Phase 10 — OpenAI TP Server: OpenAI-compatible HTTP API for TP inference.

Blueprint contracts:
  - framework_layer.components[13] OpenAITPServer
  - framework_layer.components[13].architecture (tp_sync_mechanism, endpoints,
    streaming, non_streaming, startup_sequence)
  - AGENT_SKILL.md §2.4 OpenAI TP Server Architecture

Architecture:
  - ThreadingHTTPServer + BaseHTTPRequestHandler
  - Endpoints: GET /health, POST /v1/completions
  - TP sync: broadcast_object_list for multi-GPU (rank 0 broadcasts requests;
    non-rank0 workers receive and execute identical generate() calls)
  - request_lock: threading.Lock serializes all requests on rank 0
    (NCCL collective order must match across all ranks — parallel requests
     cause interleaved collectives -> deadlock)
  - Streaming: Server-Sent Events (SSE) with `data: {json}\n\n` chunks
  - Non-streaming: JSON response with `choices/text` structure

TP sync mechanism (blueprint OpenAITPServer.architecture.tp_sync_mechanism):
  Rank 0:   receive HTTP -> broadcast_obj({action,...}) -> engine.generate() -> response
  Rank 1-3: while True: cmd=broadcast_obj({}) -> same engine.generate()

NCCL participation: non-rank0 processes participate automatically in every
model forward pass via all_reduce (RowParallelLinear) and all_gather
(ParallelLMHead). The broadcast ensures they know WHICH request to process
and execute the identical generate()/step() call sequence.

Startup:
  Single GPU:
    MODEL_DIR=.${MODEL_DIR}/qwen/Qwen3-8B python openai_tp_server.py --port 9000
  TP=4:
    torchrun --nproc_per_node=4 openai_tp_server.py --model-dir .${MODEL_DIR}/qwen/Qwen3-8B --port 9000

Benchmark usage (blueprint OpenAITPServer.benchmark_usage):
  # Terminal 1: start server
  TP_SIZE=4 PORT=9000 bash start_tp_infer_service.sh qwen

  # Terminal 2: run benchmark
  PORT=9000 NUM_PROMPTS=50 REQUEST_RATE=1 MAX_CONCURRENCY=1 bash run_myengine_benchmark.sh qwen
"""

import os
import sys
import json
import time
import uuid
import threading
import signal
import traceback
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional

import torch
import torch.distributed as dist


# ================================================================
# Engine singleton
# ================================================================
# The LLMEngine is a heavyweight object that owns model weights on GPU.
# Creating multiple engines would OOM. All request threads share one engine
# via this module-level singleton guarded by _engine_lock.
# ================================================================

_engine = None
_engine_lock = threading.Lock()


def _get_or_create_engine(model_dir, backend='qwen_tp', max_num_seqs=8):
    """Create or return the shared LLMEngine singleton.

    Thread-safe: uses _engine_lock for the lazy-init check. The engine
    object itself is NOT thread-safe for inference — callers must
    serialize generate()/step() calls externally (via _engine_lock).

    Args:
        model_dir:  Path to model directory (e.g., .${MODEL_DIR}/qwen/Qwen3-8B)
        backend:    Inference backend identifier (qwen_tp, deepseek_tp, hf)
        max_num_seqs: Max concurrent sequences for the scheduler

    Returns:
        LLMEngine singleton
    """
    global _engine
    if _engine is None:
        from llm_engine import LLMEngine
        from engine.tp_layers.distributed import init_tp_distributed, is_tp_enabled

        if not is_tp_enabled():
            init_tp_distributed()

        _engine = LLMEngine(
            model_dir=Path(model_dir),
            inference_backend=backend,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=8192,
        )
    return _engine


# ================================================================
# TP Worker Loop (non-rank0 processes)
# ================================================================

def _tp_worker_loop(model_dir, backend='qwen_tp', max_num_seqs=8):
    """Non-rank0 worker: initialize model, wait for broadcast commands.

    Blueprint contract:
      AGENT_SKILL.md §2.4 — non_rank0_flow
      OpenAITPServer.architecture.tp_sync_mechanism.non_rank0_flow

    Non-rank0 processes cannot handle HTTP requests. They participate in
    inference via NCCL collectives (all_reduce inside RowParallelLinear,
    all_gather inside ParallelLMHead) embedded in every model forward pass.

    This function:
      1. Creates the LLMEngine (loads model weights onto this rank's GPU)
      2. Enters an infinite loop waiting for broadcast_object_list from rank 0
      3. On each broadcast: executes the identical generate() or step() calls
         that rank 0 is running (order guaranteed by blocking broadcast)
      4. Exits on {'action': 'shutdown'} broadcast

    NCCL collectives inside the model forward pass execute automatically
    when any rank calls model.forward() — non-rank0 does not need to
    explicitly "send results back". The broadcast merely tells non-rank0
    which generate() call to make.
    """
    engine = _get_or_create_engine(model_dir, backend, max_num_seqs)
    rank = dist.get_rank() if dist.is_initialized() else -1

    print(f"[TP Worker Rank {rank}] Model loaded. Waiting for requests...",
          flush=True)

    while True:
        try:
            # Block until rank 0 broadcasts a request payload
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            cmd = obj_list[0]

            if cmd is None or cmd.get('action') == 'shutdown':
                print(f"[TP Worker Rank {rank}] Shutdown signal received.",
                      flush=True)
                break

            prompt = cmd.get('prompt', '')
            max_tokens = cmd.get('max_tokens', 256)
            temperature = cmd.get('temperature', 0.0)
            top_p = cmd.get('top_p', 1.0)
            stream = cmd.get('stream', False)

            # Execute the EXACT same code path as rank 0's HTTP handler.
            # Order must be identical — begin_generation + step loop for
            # streaming, or generate() for non-streaming.
            if stream:
                engine.begin_generation(
                    [prompt], max_tokens, temperature, top_p)
                while engine.has_unfinished_requests():
                    engine.step(temperature, top_p)
            else:
                engine.generate(
                    prompt, max_new_tokens=max_tokens,
                    temperature=temperature, top_p=top_p)

        except Exception as e:
            print(f"[TP Worker Rank {rank}] Error processing request: {e}",
                  flush=True)
            traceback.print_exc()
            # Continue looping — the next broadcast will resynchronize.
            # If the error was a structural NCCL fault (not just a
            # Python-side logic error), the next broadcast_object_list
            # will hang and all ranks must be restarted.

    print(f"[TP Worker Rank {rank}] Worker loop exited.", flush=True)


# ================================================================
# HTTP Server
# ================================================================

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server with daemon threads.

    Blueprint contract:
      OpenAITPServer.architecture.server_type: ThreadingHTTPServer
      (BaseHTTPRequestHandler)

    daemon_threads=True ensures the server process can exit even if
    there are lingering request handler threads.
    """
    daemon_threads = True


class CompletionHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible /v1/completions handler.

    Supports:
      - Non-streaming: POST JSON body -> generate() -> JSON response
      - Streaming: POST JSON body with stream=true -> SSE chunks

    Blueprint contract:
      OpenAITPServer.architecture.endpoints: GET /health, POST /v1/completions
      OpenAITPServer.architecture.streaming.flow
      OpenAITPServer.architecture.non_streaming.flow
    """

    # Class-level attributes injected by run_server() before serve_forever().
    # These are shared across all request handler instances.
    engine: Optional[object] = None       # LLMEngine singleton
    engine_lock: Optional[threading.Lock] = None  # Serializes all requests
    model_dir: Optional[str] = None
    backend: Optional[str] = None
    max_num_seqs: int = 8

    # Silence default http.server access logs (we log our own)
    # Uncomment the next line to suppress all log_message output:
    # def log_message(self, format, *args): pass

    def log_message(self, format, *args):
        """Override for cleaner timestamp-prefixed access logging."""
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        sys.stderr.write(
            f"[{ts}] {self.client_address[0]} {format % args}\n")

    # ----------------------------------------------------------------
    # Routing
    # ----------------------------------------------------------------

    def do_GET(self):
        """Health check endpoint."""
        if self.path == '/health' or self.path == '/':
            self._send_json({
                'status': 'ok',
                'model': 'qwen-tp',
                'backend': CompletionHandler.backend or 'qwen_tp',
            })
        else:
            self.send_error(404, 'Not Found')

    def do_POST(self):
        """Dispatch /v1/completions requests."""
        if self.path != '/v1/completions':
            self.send_error(404, 'Not Found')
            return

        # Parse JSON body
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length <= 0:
                self.send_error(400, 'Empty body')
                return
            body = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON body')
            return
        except Exception:
            self.send_error(400, 'Bad Request')
            return

        prompt = body.get('prompt', '')
        if not prompt:
            self.send_error(400, 'Missing required field: "prompt"')
            return

        max_tokens = int(body.get('max_tokens', 256))
        temperature = float(body.get('temperature', 0.0))
        stream = bool(body.get('stream', False))
        top_p = float(body.get('top_p', 1.0))

        # Clamp values to reasonable ranges
        max_tokens = max(1, min(max_tokens, 4096))
        temperature = max(0.0, min(temperature, 2.0))
        top_p = max(0.0, min(top_p, 1.0))

        if stream:
            self._handle_stream(prompt, max_tokens, temperature, top_p)
        else:
            self._handle_sync(prompt, max_tokens, temperature, top_p)

    # ----------------------------------------------------------------
    # Non-streaming (JSON response)
    # ----------------------------------------------------------------

    def _handle_sync(self, prompt, max_tokens, temperature, top_p):
        """Non-streaming: generate full response, return JSON.

        Blueprint contract:
          OpenAITPServer.architecture.non_streaming.flow:
            engine.generate() -> JSON response with choices/text/usage
        """
        with CompletionHandler.engine_lock:
            engine = CompletionHandler.engine
            self._tp_broadcast(
                prompt, max_tokens, temperature, top_p, stream=False)
            text = engine.generate(
                prompt, max_new_tokens=max_tokens,
                temperature=temperature, top_p=top_p)

        resp = {
            'id': f'cmpl-{uuid.uuid4().hex[:8]}',
            'object': 'text_completion',
            'created': int(time.time()),
            'model': 'qwen-tp',
            'choices': [{
                'text': text,
                'index': 0,
                'finish_reason': 'stop',
            }],
            'usage': {
                'prompt_tokens': len(engine.runner.tokenizer.encode(prompt)),
                'completion_tokens': len(engine.runner.tokenizer.encode(text)),
            },
        }
        self._send_json(resp)

    # ----------------------------------------------------------------
    # Streaming (Server-Sent Events)
    # ----------------------------------------------------------------

    def _handle_stream(self, prompt, max_tokens, temperature, top_p):
        """Streaming: generate token-by-token, return SSE chunks.

        Uses engine.begin_generation() + step() loop for fine-grained
        per-token control. Yields each new token as a separate SSE
        `data:` chunk, followed by a final chunk with finish_reason='stop'
        and `data: [DONE]`.

        Blueprint contract:
          OpenAITPServer.architecture.streaming.flow:
            engine.generate_stream() -> SSE data chunks ->
            final chunk finish_reason='stop' -> data: [DONE]
        """
        # Set SSE headers
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache, no-transform')
        self.send_header('Connection', 'keep-alive')
        self.send_header('X-Accel-Buffering', 'no')   # disable nginx buffering
        self.end_headers()

        request_id = f'cmpl-{uuid.uuid4().hex[:8]}'
        created = int(time.time())

        try:
            with CompletionHandler.engine_lock:
                engine = CompletionHandler.engine

                # Broadcast the request to all TP ranks first
                self._tp_broadcast(
                    prompt, max_tokens, temperature, top_p, stream=True)

                # Enqueue the prompt for step-by-step generation
                engine.begin_generation(
                    [prompt], max_tokens, temperature, top_p)
                seq = engine._active_gen_seqs[0]
                prev_len = 0
                accumulated_text = ''

                # Step loop: advance scheduler one step at a time,
                # yielding new tokens as SSE chunks
                while engine.has_unfinished_requests():
                    finished_seqs = engine.step(temperature, top_p)

                    # Check for newly generated tokens
                    new_token_ids = seq.output_ids[prev_len:]
                    prev_len = len(seq.output_ids)

                    for token_id in new_token_ids:
                        token_text = engine.runner.tokenizer.decode(
                            [token_id], skip_special_tokens=True)
                        accumulated_text += token_text

                        chunk = {
                            'id': request_id,
                            'object': 'text_completion',
                            'created': created,
                            'model': 'qwen-tp',
                            'choices': [{
                                'text': token_text,
                                'index': 0,
                                'finish_reason': None,
                                'logprobs': None,
                            }],
                        }
                        self.wfile.write(
                            f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'
                            .encode('utf-8'))
                        self.wfile.flush()

                    # Exit if this sequence finished (EOS or max_tokens)
                    if finished_seqs and seq in finished_seqs:
                        break

                # Send final chunk with finish_reason='stop'
                final_chunk = {
                    'id': request_id,
                    'object': 'text_completion',
                    'created': created,
                    'model': 'qwen-tp',
                    'choices': [{
                        'text': '',
                        'index': 0,
                        'finish_reason': 'stop',
                        'logprobs': None,
                    }],
                    'usage': {
                        'prompt_tokens':
                            len(engine.runner.tokenizer.encode(prompt)),
                        'completion_tokens': len(seq.output_ids),
                    },
                }
                self.wfile.write(
                    f'data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n'
                    .encode('utf-8'))
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()

        except Exception as e:
            # Send error via SSE so the client doesn't hang
            error_chunk = {
                'id': request_id,
                'object': 'text_completion',
                'created': created,
                'model': 'qwen-tp',
                'choices': [{
                    'text': '',
                    'index': 0,
                    'finish_reason': 'error',
                    'logprobs': None,
                }],
                'error': {
                    'message': str(e),
                    'type': type(e).__name__,
                },
            }
            try:
                self.wfile.write(
                    f'data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n'
                    .encode('utf-8'))
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except Exception:
                pass  # client may have disconnected

    # ----------------------------------------------------------------
    # TP broadcast
    # ----------------------------------------------------------------

    def _tp_broadcast(self, prompt, max_tokens, temperature, top_p, stream):
        """Broadcast generation request to all TP ranks.

        Blueprint contract:
          OpenAITPServer.architecture.tp_sync_mechanism.broadcast_obj:
            dist.broadcast_object_list([payload if rank0 else None], src=0)
            -> all ranks return payload[0]

        All ranks must participate in broadcast_object_list for
        synchronization. Rank 0 sends the payload; other ranks receive it.
        Called only when dist.world_size > 1.

        This is the bridge between the HTTP server thread (rank 0) and
        the TP worker loop (non-rank0): rank 0 broadcasts before running
        generate(), and non-rank0 workers receive the broadcast and run
        the identical generate() call.
        """
        if not dist.is_initialized():
            return
        if dist.get_world_size() <= 1:
            return

        payload = {
            'action': 'generate',
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stream': stream,
        }
        obj_list = [payload]
        dist.broadcast_object_list(obj_list, src=0)
        # After broadcast, obj_list[0] == payload on all ranks
        # (non-rank0 workers receive this in _tp_worker_loop)

    # ----------------------------------------------------------------
    # JSON response helper
    # ----------------------------------------------------------------

    def _send_json(self, data):
        """Send a JSON response with proper headers."""
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ================================================================
# Server runner (rank 0 only)
# ================================================================

def run_server(model_dir, port=9000, backend='qwen_tp', max_num_seqs=8):
    """Start the HTTP server on rank 0.

    Creates the LLMEngine eagerly (so startup errors surface immediately),
    injects it into the handler class, and begins serving.

    This function does not return until a shutdown signal (SIGINT/SIGTERM)
    is received and server.shutdown() is called.

    Args:
        model_dir:    Path to model directory
        port:         Listening port (default 9000, override via PORT env)
        backend:      Inference backend ('qwen_tp', 'deepseek_tp', 'hf')
        max_num_seqs: Max concurrent sequences for the scheduler
    """
    # Create engine eagerly — failures here should prevent server start
    with _engine_lock:
        engine = _get_or_create_engine(model_dir, backend, max_num_seqs)

    # Inject engine and config into the handler class
    # (shared across all request handler instances)
    CompletionHandler.engine = engine
    CompletionHandler.engine_lock = _engine_lock
    CompletionHandler.model_dir = model_dir
    CompletionHandler.backend = backend
    CompletionHandler.max_num_seqs = max_num_seqs

    # Create server
    server = ThreadingHTTPServer(('0.0.0.0', port), CompletionHandler)
    # Attach model_dir to the server object for handler access
    server.model_dir = model_dir

    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    print(
        f"[Rank {rank}/{world_size}] TP Inference Server starting on "
        f"http://0.0.0.0:{port}",
        flush=True)
    print(
        f"[Rank {rank}/{world_size}] Model dir: {model_dir}",
        flush=True)
    print(
        f"[Rank {rank}/{world_size}] Backend: {backend}",
        flush=True)
    print(
        f"[Rank {rank}/{world_size}] Endpoints: "
        f"GET /health, POST /v1/completions",
        flush=True)

    # ---- Shutdown handling ----
    # We cannot call dist.broadcast_object_list from a signal handler
    # (NCCL operations are not signal-safe). Instead, the signal handler
    # triggers server.shutdown(), which causes serve_forever() to return.
    # We broadcast the shutdown signal to TP workers AFTER serve_forever()
    # returns, in the normal execution flow.
    shutdown_requested = threading.Event()

    def _handle_signal(signum, frame):
        """Signal handler: trigger graceful server shutdown."""
        if not shutdown_requested.is_set():
            shutdown_requested.set()
            print(f"\n[Rank {rank}] Received signal {signum}, "
                  f"shutting down...", flush=True)
            # server.shutdown() is thread-safe and causes serve_forever()
            # to return
            server.shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Block until shutdown
    server.serve_forever()

    # ---- Cleanup (runs after serve_forever() returns) ----
    server.server_close()

    # Broadcast shutdown to non-rank0 TP workers
    # This is outside the signal handler context — safe for NCCL ops
    if dist.is_initialized() and dist.get_world_size() > 1:
        try:
            shutdown_payload = [{'action': 'shutdown'}]
            dist.broadcast_object_list(shutdown_payload, src=0)
            print(f"[Rank {rank}] Shutdown broadcast sent to TP workers.",
                  flush=True)
        except Exception as e:
            print(f"[Rank {rank}] Warning: shutdown broadcast failed: {e}",
                  flush=True)

    print(f"[Rank {rank}] Server stopped.", flush=True)


# ================================================================
# CLI entry point
# ================================================================

def main():
    """Parse args, route to server (rank 0) or worker loop (non-rank0).

    Blueprint contract:
      OpenAITPServer.startup_sequence:
        1. parse_args: --model-dir, --backend, --host, --port, --max-num-seqs
        2. init_dist_if_needed(): WORLD_SIZE>1 -> dist.init_process_group('nccl')
        3. LLMEngine(model_dir, inference_backend, max_num_seqs)
        4. run_tp_generation_loop(engine, host, port)

    Supports:
      - Single GPU:        python openai_tp_server.py --model-dir <path> --port 9000
      - TP=4 via torchrun: torchrun --nproc_per_node=4 openai_tp_server.py --model-dir <path>
      - Environment vars:  MODEL_DIR, PORT, BACKEND, MAX_NUM_SEQS
    """
    parser = argparse.ArgumentParser(
        description='OpenAI-compatible TP Inference Server (Phase 10)')
    parser.add_argument(
        '--model-dir', type=str,
        default=os.environ.get('MODEL_DIR', '${MODEL_DIR}/qwen/Qwen3-8B'),
        help='Path to model directory (default: $MODEL_DIR or '
             '${MODEL_DIR}/qwen/Qwen3-8B)')
    parser.add_argument(
        '--backend', type=str,
        default=os.environ.get('INFERENCE_BACKEND', 'qwen_tp'),
        choices=['qwen_tp', 'deepseek_tp', 'hf'],
        help='Inference backend (default: qwen_tp)')
    parser.add_argument(
        '--host', type=str, default='0.0.0.0',
        help='Bind address (default: 0.0.0.0)')
    parser.add_argument(
        '--port', type=int,
        default=int(os.environ.get('PORT', '9000')),
        help='Listening port (default: 9000, or $PORT env var)')
    parser.add_argument(
        '--max-num-seqs', type=int,
        default=int(os.environ.get('MAX_NUM_SEQS', '8')),
        help='Max concurrent sequences (default: 8)')
    parser.add_argument(
        '--max-num-batched-tokens', type=int,
        default=int(os.environ.get('MAX_NUM_BATCHED_TOKENS', '8192')),
        help='Max tokens per batch (default: 8192)')
    # Hidden args consumed by torchrun's --local-rank injection
    parser.add_argument(
        '--local-rank', type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    model_dir = args.model_dir
    port = args.port
    backend = args.backend

    # ---- Determine rank and world size ----
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    # Set CUDA device for this process
    torch.cuda.set_device(local_rank)

    # ---- Route based on rank ----
    if world_size > 1 and rank > 0:
        # Non-rank0: TP worker loop
        print(
            f"[TP Worker Rank {rank}/{world_size}] Starting worker loop. "
            f"Model: {model_dir}, Backend: {backend}",
            flush=True)
        _tp_worker_loop(model_dir, backend, args.max_num_seqs)
    else:
        # Rank 0 (or single-GPU): HTTP server
        print(
            f"[Rank {rank}/{world_size}] Starting TP Inference Server.",
            flush=True)
        run_server(model_dir, port, backend, args.max_num_seqs)


if __name__ == '__main__':
    main()
