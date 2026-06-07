"""Inference evaluation for LLaVA-OneVision wrist checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from datasets.wrist_video_sft import WristVideoSFTDataset, discover_episode_pairs
from llava.train.train_wrist import episode_sample_indices, evaluate, split_episode_indices
from llava.wrist.collator import WristLlavaCollator
from llava.wrist.metrics import compute_wrist_metrics
from llava.wrist.model import WristLlavaOVConfig, WristLlavaOneVisionModel
from llava.wrist.normalize import WristNormStats


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_model(
    checkpoint: str,
    model_name_or_path: str,
    device: torch.device,
    norm_stats_path: str | None = None,
) -> WristLlavaOneVisionModel:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    full_finetune = ckpt.get("full_finetune", "llava" in ckpt)
    cfg = WristLlavaOVConfig(
        model_name_or_path=model_name_or_path or cfg_dict.get("model_name_or_path", "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"),
        future_k=cfg_dict.get("future_k", 16),
        freeze_llava=cfg_dict.get("freeze_llava", not full_finetune),
    )
    norm_stats = None
    if ckpt.get("norm_stats") is not None:
        norm_stats = WristNormStats.from_dict(ckpt["norm_stats"])
    elif norm_stats_path and Path(norm_stats_path).is_file():
        norm_stats = WristNormStats.load(norm_stats_path)
    model = WristLlavaOneVisionModel(cfg, norm_stats=norm_stats)
    model.wrist_encoder.load_state_dict(ckpt["wrist_head"]["wrist_encoder"])
    model.head.load_state_dict(ckpt["wrist_head"]["head"])
    if "video_ctx_norm" in ckpt["wrist_head"]:
        model.video_ctx_norm.load_state_dict(ckpt["wrist_head"]["video_ctx_norm"])
    if full_finetune and "llava" in ckpt:
        model.llava.load_state_dict(ckpt["llava"], strict=False)
    if device.type == "cuda":
        model.llava.to(device)
    model.wrist_encoder.to(device=device)
    model.head.to(device=device)
    model.video_ctx_norm.to(device=device)
    model.eval()
    if cfg.freeze_llava:
        model.llava.eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device, max_batches=None):
    predictions = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if "pixel_values_videos" in batch:
            vdtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            batch["pixel_values_videos"] = batch["pixel_values_videos"].to(device, dtype=vdtype)
        out = model(
            pixel_values_videos=batch["pixel_values_videos"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            history_wrists=batch["history_wrists"],
            history_wrist_mask=batch["history_wrist_mask"],
            history_len=batch["history_len"],
        )
        pred = out["pred"].cpu()
        target = batch["future_wrists"].cpu()
        mask = batch["future_wrist_mask"].cpu()
        for b in range(pred.shape[0]):
            predictions.append(
                {
                    "video_path": batch["video_paths"][b],
                    "hist_end": int(batch["hist_ends"][b].item()),
                    "mae": compute_wrist_metrics(pred[b : b + 1], target[b : b + 1], mask[b : b + 1])["mae"],
                    "pred": pred[b].tolist(),
                    "target": target[b].tolist(),
                    "mask": mask[b].tolist(),
                }
            )
    return predictions


def main(argv=None):
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    p.add_argument("--data_root", type=str, default=str(root / "data"))
    p.add_argument("--output_dir", type=str, default=str(root / "outputs" / "wrist_llava_ov_eval"))
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--future_k", type=int, default=16)
    p.add_argument("--max_history", type=int, default=0)
    p.add_argument("--frames_upbound", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help="Must match training; 0 = all episodes belong to train split",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_batches", type=int, default=None)
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

    pairs = discover_episode_pairs(args.data_root)
    train_eps, val_eps = split_episode_indices(len(pairs), args.val_ratio, args.seed)
    ep_ids = train_eps if args.split == "train" else list(val_eps) if args.split == "val" else list(range(len(pairs)))

    ds = WristVideoSFTDataset(data_root=args.data_root, future_k=args.future_k, max_history=args.max_history)
    indices = episode_sample_indices(ds, ep_ids)
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=WristLlavaCollator(processor=processor, frames_upbound=args.frames_upbound, future_k=args.future_k),
    )

    model = load_model(args.checkpoint, args.model_name_or_path, device, norm_stats_path=args.norm_stats)
    agg = evaluate(model, loader, device)
    print(f"[{args.split}] mae={agg['mae']:.5f} rmse={agg['rmse']:.5f}")

    preds = run_inference(model, loader, device, max_batches=args.max_batches)
    os.makedirs(args.output_dir, exist_ok=True)
    out = {"checkpoint": args.checkpoint, "split": args.split, "aggregate": agg, "n_samples": len(preds)}
    with open(os.path.join(args.output_dir, f"metrics_{args.split}.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.output_dir, f"predictions_{args.split}.json"), "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote metrics and predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
