"""Build LLaVA-OneVision video inputs from wrist SFT samples (HF processor path)."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image


def uniform_sample_indices(indices: Sequence[int], max_frames: int) -> List[int]:
    """Uniformly subsample frame indices (LLaVA-style frames_upbound)."""
    idx = list(indices)
    if max_frames <= 0 or len(idx) <= max_frames:
        return idx
    sampled = np.linspace(0, len(idx) - 1, max_frames, dtype=int)
    return [idx[i] for i in sampled]


def load_video_frames_pil(
    video_path: str,
    decode_frame_ids: Sequence[int],
    *,
    max_frames: int = 8,
) -> List[Image.Image]:
    """Load RGB PIL frames from video at annotation decode indices."""
    decode_frame_ids = uniform_sample_indices(decode_frame_ids, max_frames)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frames = vr.get_batch(list(decode_frame_ids)).asnumpy()
    vr.seek(0)
    return [Image.fromarray(f) for f in frames]


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


def build_wrist_video_prompt(processor, history_text: str, future_k: int) -> str:
    """Prompt with a single <video> placeholder (expanded by the processor)."""
    return (
        f"{processor.video_token}\n"
        f"You are given a video of hand motion in the camera frame.\n"
        f"{history_text}\n"
        f"Encode the video and history, then predict the next {future_k} wrist positions "
        f"(left and right, xyz in meters) as internal regression targets."
    )


def prepare_llava_video_batch(
    processor,
    video_frame_lists: List[List[Image.Image]],
    prompts: List[str],
) -> Dict:
    """Run LlavaOnevisionProcessor on a batch (one video per sample)."""
    if len(video_frame_lists) == 1:
        return processor(text=prompts[0], videos=video_frame_lists, return_tensors="pt")

    # Variable-length videos: process per item then pad (frame dim + sequences).
    items = [processor(text=p, videos=[v], return_tensors="pt") for p, v in zip(prompts, video_frame_lists)]
    max_frames = max(x["pixel_values_videos"].shape[1] for x in items)
    max_len = max(x["input_ids"].shape[1] for x in items)

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    batch = {
        "input_ids": [],
        "attention_mask": [],
        "pixel_values_videos": [],
    }
    for item in items:
        pv = item["pixel_values_videos"]
        f_pad = max_frames - pv.shape[1]
        if f_pad > 0:
            pad_shape = (pv.shape[0], f_pad, *pv.shape[2:])
            pv = torch.cat([pv, torch.zeros(pad_shape, dtype=pv.dtype)], dim=1)
        batch["pixel_values_videos"].append(pv)

        ids = item["input_ids"]
        attn = item["attention_mask"]
        seq_pad = max_len - ids.shape[1]
        if seq_pad > 0:
            ids = torch.cat([ids, torch.full((ids.shape[0], seq_pad), pad_id, dtype=ids.dtype)], dim=1)
            attn = torch.cat([attn, torch.zeros((attn.shape[0], seq_pad), dtype=attn.dtype)], dim=1)
        batch["input_ids"].append(ids)
        batch["attention_mask"].append(attn)

    return {
        "input_ids": torch.cat(batch["input_ids"], dim=0),
        "attention_mask": torch.cat(batch["attention_mask"], dim=0),
        "pixel_values_videos": torch.cat(batch["pixel_values_videos"], dim=0),
    }
