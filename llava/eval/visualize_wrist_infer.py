#!/usr/bin/env python3
"""
Visualize inference: at each hist_end frame, overlay ONLY the next K-step trajectories.

- Green: dataset GT for future K frames
- Red: model prediction for future K frames
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from datasets.epoch_reader import WristEpisodeReader
from datasets.wrist_video_sft import WristVideoSFTDataset, discover_episode_pairs
from llava.eval.eval_wrist import load_model
from llava.wrist.collator import WristLlavaCollator
from llava.wrist.viz import render_future_traj_frame


def _video_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _pad_future_chunk(
    wrists: np.ndarray,
    mask: np.ndarray,
    start: int,
    future_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice [start:start+valid) and pad to (future_k, 2, 3) with invalid tail steps."""
    valid = min(future_k, max(0, wrists.shape[0] - start))
    gt_fut = np.full((future_k, 2, 3), np.nan, dtype=np.float32)
    gt_m = np.zeros((future_k, 2), dtype=bool)
    if valid > 0:
        gt_fut[:valid] = wrists[start : start + valid]
        gt_m[:valid] = mask[start : start + valid]
    return gt_fut, gt_m


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def episode_sample_indices_for_ep(dataset: WristVideoSFTDataset, episode_idx: int) -> list[int]:
    return [i for i, s in enumerate(dataset.samples) if s["episode_idx"] == episode_idx]


@torch.no_grad()
def collect_episode_predictions(
    model,
    processor,
    dataset: WristVideoSFTDataset,
    episode_idx: int,
    device: torch.device,
    *,
    frames_upbound: int,
    future_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """pred_wrists (T,2,3), pred_mask (T,2) — filled for frames that are predicted targets."""
    indices = episode_sample_indices_for_ep(dataset, episode_idx)
    ann = np.load(dataset.episode_pairs[episode_idx]["ann_path"], allow_pickle=True).item()
    num_frames = len(ann["video_decode_frame"])
    pred_wrists = np.full((num_frames, 2, 3), np.nan, dtype=np.float32)
    pred_mask = np.zeros((num_frames, 2), dtype=bool)

    collator = WristLlavaCollator(processor=processor, frames_upbound=frames_upbound, future_k=future_k)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, collate_fn=collator)

    model.eval()
    if model.config.freeze_llava:
        model.llava.eval()

    for batch in loader:
        vdtype = _video_dtype(device)
        batch = {
            k: v.to(device, dtype=vdtype) if k == "pixel_values_videos" and torch.is_tensor(v)
            else v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        out = model(
            pixel_values_videos=batch["pixel_values_videos"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            history_wrists=batch["history_wrists"],
            history_wrist_mask=batch["history_wrist_mask"],
            history_len=batch["history_len"],
        )
        pred = out["pred"].detach().cpu().numpy()
        pmask = batch["future_wrist_mask"].detach().cpu().numpy()
        t = int(batch["hist_ends"][0].item())
        for k in range(future_k):
            j = t + 1 + k
            if j >= num_frames:
                break
            if not pmask[0, k].any():
                continue
            for hand in range(2):
                if pmask[0, k, hand]:
                    pred_wrists[j, hand] = pred[0, k, hand]
                    pred_mask[j, hand] = True

    return pred_wrists, pred_mask


def render_episode_infer_video(
    ep,
    pred_wrists: np.ndarray,
    pred_mask: np.ndarray,
    out_path: str,
    *,
    future_k: int,
    fps: int = 10,
) -> None:
    """
    For each hist_end t, on video frame t draw future K GT (green) vs pred (red).
    Episode tail uses fewer than K valid steps; padded steps are not drawn.
    """
    num_frames = ep.num_frames
    frames_out = []

    for fi in range(num_frames):
        remaining = num_frames - 1 - fi
        if remaining <= 0:
            frames_out.append(ep.frames[fi])
            continue

        gt_fut, gt_m = _pad_future_chunk(ep.wrists, ep.wrist_mask, fi + 1, future_k)
        pred_fut, pred_m = _pad_future_chunk(pred_wrists, pred_mask, fi + 1, future_k)

        frame = render_future_traj_frame(
            ep.frames[fi],
            fi,
            num_frames,
            gt_fut,
            pred_fut,
            gt_m,
            pred_m,
            ep.intrinsics,
            future_k=future_k,
        )
        frames_out.append(frame)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames_out, fps=fps, codec="libx264", quality=8)
    print(f"Saved: {out_path}")


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(description="Future-K wrist traj viz (GT green / pred red only)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    p.add_argument("--data_root", type=str, default=str(root / "data"))
    p.add_argument("--output_dir", type=str, default=str(root / "outputs" / "wrist_infer_viz"))
    p.add_argument("--episode", default=0)
    p.add_argument("--all", action="store_true")
    p.add_argument("--future_k", type=int, default=16)
    p.add_argument("--frames_upbound", type=int, default=8)
    p.add_argument("--max_history", type=int, default=0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--norm_stats",
        type=str,
        default=str(root / "outputs" / "wrist_norm_stats.json"),
        help="Fallback if checkpoint has no embedded norm_stats",
    )
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = load_model(args.checkpoint, args.model_name_or_path, device, norm_stats_path=args.norm_stats)

    dataset = WristVideoSFTDataset(
        data_root=args.data_root,
        future_k=args.future_k,
        max_history=args.max_history,
    )
    reader = WristEpisodeReader(data_root=args.data_root)
    pairs = discover_episode_pairs(args.data_root)

    if args.all:
        ep_ids = list(range(len(pairs)))
    elif isinstance(args.episode, str) and not str(args.episode).isdigit():
        ep_ids = [i for i, ep in enumerate(pairs) if args.episode in ep["ann_path"] or args.episode in ep["video_name"]]
    else:
        ep_ids = [int(args.episode)]

    os.makedirs(args.output_dir, exist_ok=True)

    for ep_idx in ep_ids:
        print(f"\n=== Episode {ep_idx}: {pairs[ep_idx]['video_name']} ===")
        pred_wrists, pred_mask = collect_episode_predictions(
            model, processor, dataset, ep_idx, device,
            frames_upbound=args.frames_upbound, future_k=args.future_k,
        )
        ep = reader.load(ep_idx)
        stem = os.path.splitext(ep.video_name)[0]
        out_mp4 = os.path.join(args.output_dir, f"{stem}_future16_gt_green_pred_red.mp4")
        render_episode_infer_video(ep, pred_wrists, pred_mask, out_mp4, future_k=args.future_k, fps=args.fps)


if __name__ == "__main__":
    main()
