"""Collate wrist SFT samples into LLaVA-OneVision-2 codec inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from datasets.wrist_video_sft import WristVideoSFTCollator
from llava.wrist.constants import DEFAULT_CODEC_MAX_PIXELS
from llava.wrist.video_inputs import (
    build_wrist_ov2_prompt,
    format_history_wrist_prompt,
    prepare_ov2_codec_batch,
)


@dataclass
class WristLlavaCollator:
    processor: object
    future_k: int = 16
    max_pixels: int = DEFAULT_CODEC_MAX_PIXELS
    base_collator: WristVideoSFTCollator = None

    def __post_init__(self):
        if self.base_collator is None:
            self.base_collator = WristVideoSFTCollator()

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        wrist_batch = self.base_collator(instances)

        video_paths = []
        prompts = []
        for inst in instances:
            hw = inst["history_wrists"].numpy()
            hm = inst["history_wrist_mask"].numpy()
            hist_text = format_history_wrist_prompt(hw, hm)
            prompt = build_wrist_ov2_prompt(self.processor, hist_text, self.future_k)
            video_paths.append(inst["video_path"])
            prompts.append(prompt)

        llava_inputs = prepare_ov2_codec_batch(
            self.processor,
            video_paths,
            prompts,
            max_pixels=self.max_pixels,
        )
        wrist_batch["pixel_values"] = llava_inputs["pixel_values"]
        wrist_batch["image_grid_thw"] = llava_inputs["image_grid_thw"]
        wrist_batch["patch_positions"] = llava_inputs["patch_positions"]
        wrist_batch["input_ids"] = llava_inputs["input_ids"]
        wrist_batch["attention_mask"] = llava_inputs["attention_mask"]
        return wrist_batch
