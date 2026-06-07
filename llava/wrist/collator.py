"""Collate wrist SFT samples into LLaVA-OneVision video-model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from datasets.wrist_video_sft import WristVideoSFTCollator
from llava.wrist.video_inputs import (
    build_wrist_video_prompt,
    format_history_wrist_prompt,
    load_video_frames_pil,
    prepare_llava_video_batch,
)


@dataclass
class WristLlavaCollator:
    processor: object
    frames_upbound: int = 8
    future_k: int = 16
    base_collator: WristVideoSFTCollator = None

    def __post_init__(self):
        if self.base_collator is None:
            self.base_collator = WristVideoSFTCollator()

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        wrist_batch = self.base_collator(instances)

        video_lists: List[List] = []
        prompts: List[str] = []
        for inst in instances:
            ann = np.load(inst["ann_path"], allow_pickle=True).item()
            decode = ann["video_decode_frame"]
            hist_start = int(inst["hist_start"])
            hist_end = int(inst["hist_end"])
            decode_hist = [decode[i] for i in range(hist_start, hist_end + 1)]

            pil_frames = load_video_frames_pil(
                inst["video_path"],
                decode_hist,
                max_frames=self.frames_upbound,
            )
            hw = inst["history_wrists"].numpy()
            hm = inst["history_wrist_mask"].numpy()
            hist_text = format_history_wrist_prompt(hw, hm)
            prompt = build_wrist_video_prompt(self.processor, hist_text, self.future_k)

            video_lists.append(pil_frames)
            prompts.append(prompt)

        llava_inputs = prepare_llava_video_batch(self.processor, video_lists, prompts)
        wrist_batch["pixel_values_videos"] = llava_inputs["pixel_values_videos"]
        wrist_batch["input_ids"] = llava_inputs["input_ids"]
        wrist_batch["attention_mask"] = llava_inputs["attention_mask"]
        return wrist_batch
