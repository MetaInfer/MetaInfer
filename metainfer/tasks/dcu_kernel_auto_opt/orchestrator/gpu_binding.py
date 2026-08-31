"""GPU visibility policy for worker29.

DTK accepts either HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES. Setting both
to the same non-zero physical index applies filtering twice and can hide the
device, so this task standardizes on HIP_VISIBLE_DEVICES only.
"""

from __future__ import annotations

from typing import Dict, MutableMapping


def bind_worker_gpu(
    env: MutableMapping[str, str], physical_gpu: int
) -> Dict[str, str]:
    if physical_gpu not in range(4):
        raise ValueError("physical_gpu must be one of 0,1,2,3")
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = str(physical_gpu)
    return {"HIP_VISIBLE_DEVICES": str(physical_gpu)}


def hide_gpus_from_control_plane(
    env: MutableMapping[str, str],
) -> Dict[str, str]:
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = ""
    return {"HIP_VISIBLE_DEVICES": ""}
