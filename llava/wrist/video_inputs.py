"""Build LLaVA-OneVision-2 codec video inputs for wrist SFT samples."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from llava.wrist.constants import DEFAULT_CODEC_MAX_PIXELS


def format_history_wrist_prompt(
    history_wrists: np.ndarray,
    history_mask: np.ndarray,
    *,
    max_steps: int = 8,
) -> str:
    """Compact text summary of history wrists for the language prompt."""
    t = history_wrists.shape[0]
    lines = []
    step_ids = list(range(t))
    if len(step_ids) > max_steps:
        step_ids = [int(i) for i in np.linspace(0, t - 1, max_steps, dtype=int)]
    for i in step_ids:
        parts = []
        for hand, name in enumerate(("L", "R")):
            if history_mask[i, hand]:
                xyz = history_wrists[i, hand]
                if not np.isnan(xyz).any():
                    parts.append(f"{name}({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})")
        if parts:
            lines.append(f"t{i}: " + " ".join(parts))
    if not lines:
        return "No valid history wrists."
    return "History wrists (camera m): " + "; ".join(lines)


def build_wrist_ov2_messages(history_text: str, future_k: int) -> list:
    """Chat messages with a single video slot for LlavaOnevision2Processor."""
    user_text = (
        "You are given a video of hand motion in the camera frame.\n"
        f"{history_text}\n"
        f"Encode the video and history, then predict the next {future_k} wrist positions "
        f"(left and right, xyz in meters) as internal regression targets."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def build_wrist_ov2_prompt(processor, history_text: str, future_k: int) -> str:
    messages = build_wrist_ov2_messages(history_text, future_k)
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_ov2_codec_batch(
    processor,
    video_paths: Sequence[str],
    prompts: Sequence[str],
    *,
    max_pixels: int = DEFAULT_CODEC_MAX_PIXELS,
) -> Dict:
    """Run LlavaOnevision2Processor with codec video backend on a batch."""
    video_paths = list(video_paths)
    prompts = list(prompts)
    if len(video_paths) != len(prompts):
        raise ValueError(f"video_paths ({len(video_paths)}) and prompts ({len(prompts)}) length mismatch")

    if len(video_paths) == 1:
        return processor(
            text=[prompts[0]],
            videos=[video_paths[0]],
            return_tensors="pt",
            video_backend="codec",
            max_pixels=max_pixels,
        )

    items = [
        processor(
            text=[prompt],
            videos=[video_path],
            return_tensors="pt",
            video_backend="codec",
            max_pixels=max_pixels,
        )
        for prompt, video_path in zip(prompts, video_paths)
    ]
    max_len = max(x["input_ids"].shape[1] for x in items)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    input_ids, attention_mask = [], []
    pixel_values, image_grid_thw, patch_positions = [], [], []
    for item in items:
        ids = item["input_ids"]
        attn = item["attention_mask"]
        seq_pad = max_len - ids.shape[1]
        if seq_pad > 0:
            ids = torch.cat([ids, torch.full((1, seq_pad), pad_id, dtype=ids.dtype)], dim=1)
            attn = torch.cat([attn, torch.zeros((1, seq_pad), dtype=attn.dtype)], dim=1)
        input_ids.append(ids)
        attention_mask.append(attn)
        pixel_values.append(item["pixel_values"])
        image_grid_thw.append(item["image_grid_thw"])
        patch_positions.append(item["patch_positions"])

    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "attention_mask": torch.cat(attention_mask, dim=0),
        "pixel_values": torch.cat(pixel_values, dim=0),
        "image_grid_thw": torch.cat(image_grid_thw, dim=0),
        "patch_positions": torch.cat(patch_positions, dim=0),
    }
