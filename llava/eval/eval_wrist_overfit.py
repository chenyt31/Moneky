"""Evaluate wrist overfit checkpoints (mlp / llava cached-ctx modes)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.wrist_video_sft import WristVideoSFTCollator, WristVideoSFTDataset, discover_episode_pairs
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


if __name__ == "__main__":
    main()
