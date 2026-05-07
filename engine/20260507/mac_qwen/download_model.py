"""CLI: 下载 Qwen3.5 MoE 模型。

用法:
    python engine/20260507/mac_qwen/download_model.py --endpoint https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 目录名含数字前缀，无法用常规 import，手动加入 sys.path
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from downloader import MODEL_ID, download_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 QwenPaw-Flash-2B 模型")
    parser.add_argument("--model-id", default=MODEL_ID, help="HuggingFace 模型 ID")
    parser.add_argument("--local-dir", default=None, help="本地存储路径")
    parser.add_argument("--endpoint", default=None, help="镜像源地址 (如 https://hf-mirror.com)")
    args = parser.parse_args()
    download_model(model_id=args.model_id, local_dir=args.local_dir, endpoint=args.endpoint)


if __name__ == "__main__":
    main()
