"""Evaluate wrist overfit checkpoints (mlp / llava cached-ctx modes)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.epoch_reader import WristEpisodeReader
from datasets.wrist_video_sft import WristVideoSFTCollator, WristVideoSFTDataset, discover_episode_pairs
from llava.eval.visualize_wrist_infer import episode_sample_indices_for_ep, render_episode_infer_video
from llava.train.train_wrist_overfit import OverfitCollator, build_video_cache, evaluate
from llava.wrist.constants import DEFAULT_CODEC_MAX_PIXELS, DEFAULT_OV2_CKPT
from llava.wrist.normalize import WristNormStats
from llava.wrist.overfit_model import WristOverfitConfig, WristOverfitMLP


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_overfit_model(
    checkpoint: str,
    device: torch.device,
    *,
    norm_stats_path: str | None = None,
) -> tuple[WristOverfitMLP, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = WristOverfitConfig(**ckpt["config"])
    norm_stats = None
    if ckpt.get("norm_stats") is not None:
        norm_stats = WristNormStats.from_dict(ckpt["norm_stats"])
    elif norm_stats_path and Path(norm_stats_path).is_file():
        norm_stats = WristNormStats.load(norm_stats_path)
    model = WristOverfitMLP(cfg, norm_stats=norm_stats)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_overfit_episode_predictions(
    model: WristOverfitMLP,
    dataset: WristVideoSFTDataset,
    collator: OverfitCollator,
    episode_idx: int,
    device: torch.device,
    *,
    future_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """pred_wrists (T,2,3), pred_mask (T,2) for one episode."""
    indices = episode_sample_indices_for_ep(dataset, episode_idx)
    ann = np.load(dataset.episode_pairs[episode_idx]["ann_path"], allow_pickle=True).item()
    num_frames = len(ann["video_decode_frame"])
    pred_wrists = np.full((num_frames, 2, 3), np.nan, dtype=np.float32)
    pred_mask = np.zeros((num_frames, 2), dtype=bool)

    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, collate_fn=collator)
    model.eval()

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        kwargs = dict(
            history_wrists=batch["history_wrists"],
            history_wrist_mask=batch["history_wrist_mask"],
            history_len=batch["history_len"],
            episode_idx=batch["episode_idx"],
            hist_ends=batch["hist_ends"],
        )
        if "video_ctx" in batch:
            kwargs["video_ctx"] = batch["video_ctx"]
        out = model(**kwargs)
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


def save_codec_infer_video(
    model: WristOverfitMLP,
    dataset: WristVideoSFTDataset,
    collator: OverfitCollator,
    episode_idx: int,
    output_dir: str,
    device: torch.device,
    *,
    future_k: int,
    fps: int,
) -> str:
    reader = WristEpisodeReader(data_root=dataset.data_root)
    pred_wrists, pred_mask = collect_overfit_episode_predictions(
        model, dataset, collator, episode_idx, device, future_k=future_k
    )
    ep = reader.load(episode_idx)
    stem = os.path.splitext(ep.video_name)[0]
    out_mp4 = os.path.join(output_dir, f"{stem}_codec.mp4")
    render_episode_infer_video(ep, pred_wrists, pred_mask, out_mp4, future_k=future_k, fps=fps)
    return out_mp4


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(description="Evaluate wrist overfit checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default=str(root / "data"))
    p.add_argument("--output_dir", default=str(root / "outputs" / "wrist_overfit_eval"))
    p.add_argument("--model_name_or_path", default=DEFAULT_OV2_CKPT, help="For llava mode video cache")
    p.add_argument("--video_cache_path", default=None)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--future_k", type=int, default=16)
    p.add_argument("--max_history", type=int, default=0)
    p.add_argument("--max_pixels", type=int, default=DEFAULT_CODEC_MAX_PIXELS)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--norm_stats", default=str(root / "outputs" / "wrist_norm_stats.json"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--viz_episode", type=int, default=0, help="Episode index for codec viz mp4 (-1 to skip)")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckpt = load_overfit_model(args.checkpoint, device, norm_stats_path=args.norm_stats)
    mode = ckpt.get("mode", "mlp")

    ds = WristVideoSFTDataset(
        data_root=args.data_root,
        future_k=args.future_k,
        max_history=args.max_history,
        load_video=mode == "llava",
    )
    t_max = max(s["hist_len"] for s in ds.samples)

    video_cache = None
    if mode == "llava":
        cache_path = args.video_cache_path or str(Path(args.checkpoint).parent / "video_ctx_cache.pt")
        if os.path.isfile(cache_path) and not args.rebuild_cache:
            print(f"Load video cache: {cache_path}")
            video_cache = torch.load(cache_path, map_location="cpu", weights_only=False)["cache"]
        else:
            print("Building LLaVA-OneVision-2 codec video_ctx cache ...")
            video_cache = build_video_cache(
                args.model_name_or_path,
                ds,
                device,
                future_k=args.future_k,
                max_pixels=args.max_pixels,
            )

    collator = OverfitCollator(video_cache=video_cache, pad_to_max_hist=t_max)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    metrics = evaluate(model, loader, device)
    pairs = discover_episode_pairs(args.data_root)
    out = {
        "checkpoint": args.checkpoint,
        "mode": mode,
        "n_samples": len(ds),
        "n_episodes": len(pairs),
        "metrics": metrics,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics_train.json")
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"[overfit eval] mode={mode} samples={len(ds)} "
        f"mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f}"
    )
    print(f"Wrote {metrics_path}")

    if args.viz_episode >= 0:
        if args.viz_episode >= len(pairs):
            raise ValueError(f"viz_episode={args.viz_episode} out of range (n_episodes={len(pairs)})")
        viz_path = save_codec_infer_video(
            model,
            ds,
            collator,
            args.viz_episode,
            args.output_dir,
            device,
            future_k=args.future_k,
            fps=args.fps,
        )
        out["viz_codec_mp4"] = viz_path
        with open(metrics_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote codec viz: {viz_path}")


if __name__ == "__main__":
    main()
