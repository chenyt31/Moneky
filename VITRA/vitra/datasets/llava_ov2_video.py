"""LLaVA-OneVision-2 codec video inputs for VITRA (history-only, no language)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message=r"\[codec\].*",
    category=UserWarning,
)

DEFAULT_CODEC_MAX_PIXELS = int(os.environ.get("LLAVA_CODEC_MAX_PIXELS", "200704"))
DEFAULT_CODEC_TARGET_CANVAS = int(os.environ.get("VITRA_CODEC_TARGET_CANVAS", "4"))
DEFAULT_CODEC_GROUP_SIZE = int(os.environ.get("VITRA_CODEC_GROUP_SIZE", "32"))
DEFAULT_CODEC_IMAGES_PER_GROUP = int(os.environ.get("VITRA_CODEC_IMAGES_PER_GROUP", "4"))
DEFAULT_CODEC_MIN_GROUP_FRAMES = int(os.environ.get("VITRA_CODEC_MIN_GROUP_FRAMES", "8"))
DEFAULT_CODEC_HISTORY_FRAMES = int(
    os.environ.get(
        "VITRA_CODEC_HISTORY_FRAMES",
        str(
            (DEFAULT_CODEC_TARGET_CANVAS // DEFAULT_CODEC_IMAGES_PER_GROUP)
            * DEFAULT_CODEC_GROUP_SIZE
        ),
    )
)
DEFAULT_HISTORY_CLIP_CACHE = os.environ.get(
    "VITRA_HISTORY_CLIP_CACHE",
    os.path.join(tempfile.gettempdir(), "vitra_llava_ov2_history_clips"),
)
DEFAULT_CODEC_FEATURE_CACHE = os.environ.get("VITRA_CODEC_FEATURE_CACHE", "")

# (abs_video_path, anchor_frame, history_frames, cache_tag) -> clipped mp4 path
_HISTORY_CLIP_CACHE: dict[Tuple[str, int, int, str], str] = {}


def default_codec_config(
    *,
    target_canvas: int = DEFAULT_CODEC_TARGET_CANVAS,
    group_size: int = DEFAULT_CODEC_GROUP_SIZE,
    images_per_group: int = DEFAULT_CODEC_IMAGES_PER_GROUP,
    min_group_frames: int = DEFAULT_CODEC_MIN_GROUP_FRAMES,
) -> dict:
    return {
        "target_canvas": int(target_canvas),
        "group_size": int(group_size),
        "images_per_group": int(images_per_group),
        "min_group_frames": int(min_group_frames),
    }


def build_vitra_ov2_messages() -> list:
    """Video-only user message: no task / instruction text."""
    return [{"role": "user", "content": [{"type": "video"}]}]


def build_vitra_ov2_prompt(processor) -> str:
    messages = build_vitra_ov2_messages()
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def read_history_frames(video_path: str, anchor_frame: int) -> list[np.ndarray]:
    """Read frames [0, anchor_frame] inclusive from a video (BGR, as OpenCV returns)."""
    anchor_frame = int(anchor_frame)
    if anchor_frame < 0:
        raise ValueError(f"anchor_frame must be >= 0, got {anchor_frame}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    idx = 0
    while idx <= anchor_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path} (anchor={anchor_frame})")
    return frames


def normalize_history_frames_for_codec(
    frames_bgr: list[np.ndarray],
    *,
    history_frames: int = DEFAULT_CODEC_HISTORY_FRAMES,
) -> list[np.ndarray]:
    """Keep the most recent ``history_frames`` frames; pad short clips by repeating last frame."""
    if not frames_bgr:
        raise ValueError("empty history frames")
    target = max(int(history_frames), DEFAULT_CODEC_MIN_GROUP_FRAMES)
    if len(frames_bgr) > target:
        frames_bgr = frames_bgr[-target:]
    out = list(frames_bgr)
    while len(out) < target:
        out.append(out[-1].copy())
    return out


def history_clip_cache_path(
    video_path: str,
    anchor_frame: int,
    cache_dir: str,
    *,
    history_frames: int,
    cache_tag: str,
) -> Path:
    abs_path = os.path.abspath(video_path)
    digest = hashlib.sha1(
        f"{abs_path}|{int(anchor_frame)}|hist{int(history_frames)}|{cache_tag}".encode()
    ).hexdigest()
    return Path(cache_dir) / f"hist_{digest}.mp4"


def write_history_video_clip(
    video_path: str,
    anchor_frame: int,
    *,
    cache_dir: Optional[str] = DEFAULT_HISTORY_CLIP_CACHE,
    history_frames: int = DEFAULT_CODEC_HISTORY_FRAMES,
    cache_tag: str = "",
) -> str:
    """Write (or reuse) an mp4 containing normalized history frames ending at ``anchor_frame``."""
    key = (os.path.abspath(video_path), int(anchor_frame), int(history_frames), cache_tag)
    cached = _HISTORY_CLIP_CACHE.get(key)
    if cached and os.path.isfile(cached):
        return cached

    if cache_dir:
        out_path = history_clip_cache_path(
            video_path,
            anchor_frame,
            cache_dir,
            history_frames=history_frames,
            cache_tag=cache_tag,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.is_file():
            _HISTORY_CLIP_CACHE[key] = str(out_path)
            return str(out_path)
    else:
        fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        out_path = Path(tmp_name)

    frames_bgr = normalize_history_frames_for_codec(
        read_history_frames(video_path, anchor_frame),
        history_frames=history_frames,
    )
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    fps = 30.0
    if len(rgb_frames) > 1:
        imageio.mimsave(str(out_path), rgb_frames, fps=fps, codec="libx264")
    else:
        imageio.mimsave(str(out_path), rgb_frames, fps=fps, codec="libx264", macro_block_size=1)

    out_str = str(out_path)
    _HISTORY_CLIP_CACHE[key] = out_str
    return out_str


def _codec_feature_cache_key(
    video_path: str,
    anchor_frame: int,
    *,
    max_pixels: int,
    codec_config: Mapping[str, object],
    history_frames: int,
    cache_tag: str,
) -> str:
    payload = {
        "video_path": os.path.abspath(video_path),
        "anchor_frame": int(anchor_frame),
        "max_pixels": int(max_pixels),
        "codec_config": dict(codec_config),
        "history_frames": int(history_frames),
        "cache_tag": cache_tag,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest


def _load_cached_codec_features(cache_file: Path) -> Optional[Dict]:
    if not cache_file.is_file():
        return None
    try:
        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
    except Exception:
        return None
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "patch_positions"}
    if not required.issubset(payload):
        return None
    return payload


def _save_cached_codec_features(cache_file: Path, features: Dict) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": features["input_ids"].cpu(),
            "attention_mask": features["attention_mask"].cpu(),
            "pixel_values": features["pixel_values"].cpu(),
            "image_grid_thw": features["image_grid_thw"].cpu(),
            "patch_positions": features["patch_positions"].cpu(),
        },
        cache_file,
    )


def prepare_ov2_codec_batch(
    processor,
    video_paths: Sequence[str],
    anchor_frames: Sequence[int],
    *,
    max_pixels: int = DEFAULT_CODEC_MAX_PIXELS,
    history_cache_dir: Optional[str] = DEFAULT_HISTORY_CLIP_CACHE,
    history_frames: int = DEFAULT_CODEC_HISTORY_FRAMES,
    codec_config: Optional[Mapping[str, object]] = None,
    feature_cache_dir: Optional[str] = DEFAULT_CODEC_FEATURE_CACHE,
    cache_tag: str = "",
) -> Dict:
    """Codec-encode normalized history video clips [..anchor] with a video-only prompt."""
    video_paths = list(video_paths)
    anchor_frames = [int(x) for x in anchor_frames]
    if len(video_paths) != len(anchor_frames):
        raise ValueError("video_paths and anchor_frames length mismatch")

    effective_codec_config = default_codec_config()
    if codec_config:
        effective_codec_config.update(dict(codec_config))

    clipped_paths = [
        write_history_video_clip(
            vp,
            af,
            cache_dir=history_cache_dir,
            history_frames=history_frames,
            cache_tag=cache_tag,
        )
        for vp, af in zip(video_paths, anchor_frames)
    ]
    prompt = build_vitra_ov2_prompt(processor)
    prompts = [prompt] * len(clipped_paths)

    def _encode_one(prompt_text: str, clipped_path: str, src_video_path: str, anchor_frame: int) -> Dict:
        cache_file = None
        if feature_cache_dir:
            cache_key = _codec_feature_cache_key(
                src_video_path,
                anchor_frame,
                max_pixels=max_pixels,
                codec_config=effective_codec_config,
                history_frames=history_frames,
                cache_tag=cache_tag,
            )
            cache_file = Path(feature_cache_dir) / f"{cache_key}.pt"
            cached_item = _load_cached_codec_features(cache_file)
            if cached_item is not None:
                return cached_item

        item = processor(
            text=[prompt_text],
            videos=[clipped_path],
            return_tensors="pt",
            video_backend="codec",
            max_pixels=max_pixels,
            codec_config=effective_codec_config,
        )
        if feature_cache_dir and cache_file is not None:
            _save_cached_codec_features(cache_file, item)
        return item

    if len(clipped_paths) == 1:
        return _encode_one(prompts[0], clipped_paths[0], video_paths[0], anchor_frames[0])

    items = [
        _encode_one(prompt_text, clipped_path, src_video_path, anchor_frame)
        for prompt_text, clipped_path, src_video_path, anchor_frame in zip(
            prompts, clipped_paths, video_paths, anchor_frames
        )
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
