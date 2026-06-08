"""Collate VITRA samples into LLaVA-OneVision-2 codec history-video features (no language)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import os
import torch

from vitra.datasets.llava_ov2_video import (
    DEFAULT_CODEC_FEATURE_CACHE,
    DEFAULT_CODEC_HISTORY_FRAMES,
    DEFAULT_CODEC_MAX_PIXELS,
    DEFAULT_HISTORY_CLIP_CACHE,
    default_codec_config,
    prepare_ov2_codec_batch,
)


@dataclass
class LlavaOV2HandCollator:
    processor: object
    max_pixels: int = DEFAULT_CODEC_MAX_PIXELS
    history_cache_dir: Optional[str] = DEFAULT_HISTORY_CLIP_CACHE
    history_frames: int = DEFAULT_CODEC_HISTORY_FRAMES
    codec_config: Optional[Mapping[str, object]] = field(default_factory=default_codec_config)
    feature_cache_dir: Optional[str] = DEFAULT_CODEC_FEATURE_CACHE
    cache_tag: str = "v1"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        video_paths = [inst["video_path"] for inst in instances]
        anchor_frames = [int(inst["frame_id"]) for inst in instances]
        llava_inputs = prepare_ov2_codec_batch(
            self.processor,
            video_paths,
            anchor_frames,
            max_pixels=self.max_pixels,
            history_cache_dir=self.history_cache_dir,
            history_frames=self.history_frames,
            codec_config=self.codec_config,
            feature_cache_dir=self.feature_cache_dir,
            cache_tag=self.cache_tag,
        )

        return {
            "pixel_values": llava_inputs["pixel_values"],
            "input_ids": llava_inputs["input_ids"],
            "attention_mask": llava_inputs["attention_mask"],
            "image_grid_thw": llava_inputs["image_grid_thw"],
            "patch_positions": llava_inputs["patch_positions"],
            "actions": torch.stack([inst["actions"] for inst in instances]),
            "action_masks": torch.stack([inst["action_masks"] for inst in instances]),
            "current_state": torch.stack([inst["current_state"] for inst in instances]),
            "current_state_mask": torch.stack([inst["current_state_mask"] for inst in instances]),
            "fov": torch.stack([inst["fov"] for inst in instances]),
            "intrinsics": torch.stack([inst["intrinsics"] for inst in instances]),
            "frame_id": torch.tensor([inst["frame_id"] for inst in instances], dtype=torch.long),
            "sample_idx": torch.tensor([inst["sample_idx"] for inst in instances], dtype=torch.long),
            "episode_id": [inst["episode_id"] for inst in instances],
            "video_path": [inst["video_path"] for inst in instances],
        }


def build_llava_ov2_collator(cfg: dict, processor, cache_root: str) -> LlavaOV2HandCollator:
    trainer_cfg = cfg.get("trainer", {})
    codec_cfg = trainer_cfg.get("codec", {})
    history_frames = int(codec_cfg.get("history_frames", 32))
    codec_config = {
        "target_canvas": int(codec_cfg.get("target_canvas", 4)),
        "group_size": int(codec_cfg.get("group_size", 32)),
        "images_per_group": int(codec_cfg.get("images_per_group", 4)),
        "min_group_frames": int(codec_cfg.get("min_group_frames", 8)),
    }
    history_cache_dir = os.path.join(cache_root, "history_clips")
    feature_cache_dir = os.path.join(cache_root, "features")
    os.makedirs(history_cache_dir, exist_ok=True)
    os.makedirs(feature_cache_dir, exist_ok=True)
    return LlavaOV2HandCollator(
        processor=processor,
        max_pixels=trainer_cfg.get("max_pixels", 200704),
        history_cache_dir=history_cache_dir,
        history_frames=history_frames,
        codec_config=codec_config,
        feature_cache_dir=feature_cache_dir,
        cache_tag=f"tc{codec_config['target_canvas']}_hist{history_frames}",
    )
