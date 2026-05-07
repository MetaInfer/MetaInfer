"""模型下载工具：支持断点续传、镜像源切换。"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "agentscope-ai/QwenPaw-Flash-2B"

# 视觉编码器相关文件不需要下载
_SKIP_PATTERNS = (
    "visual*",
    "model.vit*",
    "*.mmproj*",
)


def download_model(
    model_id: str = MODEL_ID,
    local_dir: str | Path | None = None,
    endpoint: str | None = None,
) -> Path:
    """下载模型到本地目录。

    Args:
        model_id: HuggingFace 模型 ID。
        local_dir: 本地存储路径，默认 ``~/.cache/huggingface/hub/<model_id>`` 。
        endpoint: 自定义镜像地址，默认读取 ``HF_ENDPOINT`` 环境变量。

    Returns:
        模型本地目录路径。
    """
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    kwargs: dict = {
        "repo_id": model_id,
        "resume_download": True,
        "ignore_patterns": list(_SKIP_PATTERNS),
    }
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)

    path = snapshot_download(**kwargs)
    print(f"[downloader] 模型已就绪: {path}")
    return Path(path)
