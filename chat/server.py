"""Mac GPU 推理引擎 Web 服务。启动后访问 http://localhost:8765 即可使用。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# 将项目根目录加入 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

engine = None  # 延迟初始化


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._serve_file("index.html", "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/generate":
            self._handle_generate()
        else:
            self.send_error(404)

    def _serve_file(self, filename: str, content_type: str):
        fp = os.path.join(CHAT_DIR, filename)
        with open(fp, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_generate(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 1024)
        temperature = body.get("temperature", 0.0)

        try:
            result = engine.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
            self._json_response({"output": result})
        except Exception as e:
            self._json_response({"error": str(e)}, status=500)

    def _json_response(self, data: dict, status: int = 200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[HTTP] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Mac GPU Chat Server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Model name or path")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    args = parser.parse_args()

    global engine
    from mac_gpu.engine import MacGPUEngine

    print(f"[server] Loading model: {args.model}")
    engine = MacGPUEngine(args.model)
    print(f"[server] Model loaded. Starting http://{args.host}:{args.port}")

    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
