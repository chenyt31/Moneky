"""Shared defaults for LLaVA-OneVision-2 wrist training."""

from __future__ import annotations

import os

DEFAULT_OV2_CKPT = os.environ.get(
    "LLAVA_OV2_CKPT",
    "/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/yangtao/monkey_asset/LLaVA-OneVision-2-8B-Instruct",
)

DEFAULT_CODEC_MAX_PIXELS = int(os.environ.get("LLAVA_CODEC_MAX_PIXELS", "200704"))
