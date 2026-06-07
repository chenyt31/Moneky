"""
Train wrist trajectory prediction with LLaVA-OneVision (HF) video modality.

Model: llava-hf/llava-onevision-qwen2-0.5b-ov-hf
Data: datasets/wrist_video_sft.py + data/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from datasets.wrist_video_sft import WristVideoSFTDataset, discover_episode_pairs
from llava.wrist.collator import WristLlavaCollator
from llava.wrist.metrics import compute_wrist_metrics
from llava.wrist.model import WristLlavaOVConfig, WristLlavaOneVisionModel
from llava.wrist.normalize import WristNormStats, compute_wrist_norm_stats


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def split_episode_indices(n_episodes: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Return (train_episode_ids, val_episode_ids). val_ratio=0 uses all episodes for training."""
    if n_episodes <= 0:
        return [], []
    if val_ratio <= 0:
        return list(range(n_episodes)), []
    rng = random.Random(seed)
    indices = list(range(n_episodes))
    rng.shuffle(indices)
    n_val = max(1, int(n_episodes * val_ratio))
    if n_val >= n_episodes:
        n_val = max(1, n_episodes - 1)
    val_eps = set(indices[:n_val])
    train_eps = [i for i in indices if i not in val_eps]
    if not train_eps:
        train_eps = [i for i in range(n_episodes) if i not in val_eps] or [0]
    return train_eps, sorted(val_eps)


def episode_sample_indices(dataset: WristVideoSFTDataset, episode_ids: list[int]) -> list[int]:
    out = []
    for idx, sample in enumerate(dataset.samples):
        if sample["episode_idx"] in episode_ids:
            out.append(idx)
    return out


def _video_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if k == "pixel_values_videos":
                out[k] = v.to(device, dtype=_video_dtype(device))
            else:
                out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(model: WristLlavaOneVisionModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    if model.config.freeze_llava:
        model.llava.eval()
    agg = {"mae": 0.0, "rmse": 0.0, "n_valid": 0.0, "left_mae": 0.0, "right_mae": 0.0, "batches": 0, "valid_batches": 0}
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(
            pixel_values_videos=batch["pixel_values_videos"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            history_wrists=batch["history_wrists"],
            history_wrist_mask=batch["history_wrist_mask"],
            history_len=batch["history_len"],
            future_wrists=batch["future_wrists"],
            future_wrist_mask=batch["future_wrist_mask"],
        )
        m = compute_wrist_metrics(out["pred"], batch["future_wrists"], batch["future_wrist_mask"])
        if m.get("mae") != m.get("mae"):
            agg["batches"] += 1
            continue
        for k in ("mae", "rmse", "n_valid", "left_mae", "right_mae"):
            if k in m and m[k] == m[k]:
                agg[k] += m[k]
        agg["batches"] += 1
        agg["valid_batches"] += 1
    if agg["valid_batches"] == 0:
        return {**agg, "mae": float("nan"), "rmse": float("nan")}
    for k in ("mae", "rmse", "left_mae", "right_mae"):
        agg[k] /= agg["valid_batches"]
    return agg


def _build_optimizer(model: WristLlavaOneVisionModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    if model.config.freeze_llava:
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        model.trainable_parameter_groups(
            lr=args.lr,
            llava_lr=args.llava_lr,
            weight_decay=args.weight_decay,
        )
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    pairs = discover_episode_pairs(args.data_root)
    train_eps, val_eps = split_episode_indices(len(pairs), args.val_ratio, args.seed)

    full_ds = WristVideoSFTDataset(
        data_root=args.data_root,
        future_k=args.future_k,
        max_history=args.max_history,
        image_size=(224, 224),
    )
    train_idx = episode_sample_indices(full_ds, train_eps)
    val_idx = episode_sample_indices(full_ds, val_eps) if val_eps else []

    collator = WristLlavaCollator(
        processor=processor,
        frames_upbound=args.frames_upbound,
        future_k=args.future_k,
    )
    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=False,
    )
    val_loader = (
        DataLoader(
            Subset(full_ds, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        if val_idx
        else None
    )

    norm_stats: WristNormStats | None = None
    if args.no_normalize:
        print("Training without wrist xyz normalization.")
    else:
        norm_path = Path(args.norm_stats)
        if args.compute_norm or not norm_path.is_file():
            print(f"Computing wrist norm stats from {args.data_root} ...")
            norm_stats = compute_wrist_norm_stats(args.data_root)
            norm_stats.save(norm_path)
        else:
            norm_stats = WristNormStats.load(norm_path)
        print(f"Wrist norm: mean={norm_stats.mean} std={norm_stats.std} ({norm_path})")

    cfg = WristLlavaOVConfig(
        model_name_or_path=args.model_name_or_path,
        future_k=args.future_k,
        freeze_llava=args.freeze_llava,
    )
    model = WristLlavaOneVisionModel(cfg, norm_stats=norm_stats)
    if args.gradient_checkpointing and not cfg.freeze_llava:
        model.enable_gradient_checkpointing()
    if device.type == "cuda":
        model.llava.to(device)
    model.wrist_encoder.to(device=device)
    model.head.to(device=device)
    model.video_ctx_norm.to(device=device)

    optim = _build_optimizer(model, args)
    llava_n, head_n = model.n_trainable_params()

    os.makedirs(args.output_dir, exist_ok=True)
    meta = {
        "model_name_or_path": args.model_name_or_path,
        "data_root": args.data_root,
        "frames_upbound": args.frames_upbound,
        "n_train_samples": len(train_idx),
        "n_val_samples": len(val_idx),
        "train_episodes": train_eps,
        "val_episodes": list(val_eps),
        "freeze_llava": cfg.freeze_llava,
        "full_finetune": not cfg.freeze_llava,
        "norm_stats": norm_stats.to_dict() if norm_stats is not None else None,
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Device: {device}")
    print(f"Model: {args.model_name_or_path} (video modality via pixel_values_videos)")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    if cfg.freeze_llava:
        print("Finetune: wrist head only (LLaVA frozen)")
    else:
        print(f"Finetune: full model — LLaVA trainable={llava_n/1e6:.1f}M, head={head_n/1e6:.2f}M")
        print(f"  lr(head)={args.lr}  lr(llava)={args.llava_lr}  amp={use_amp} dtype={amp_dtype}")

    global_step = 0
    best_eval_mae = float("inf")
    # When val_ratio=0, monitor/save best on train set MAE.
    monitor_loader = val_loader if val_loader is not None else train_loader
    monitor_tag = "val" if val_loader is not None else "train"

    for epoch in range(args.epochs):
        model.train()
        if model.config.freeze_llava:
            model.llava.eval()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = _to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    pixel_values_videos=batch["pixel_values_videos"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    history_wrists=batch["history_wrists"],
                    history_wrist_mask=batch["history_wrist_mask"],
                    history_len=batch["history_len"],
                    future_wrists=batch["future_wrists"],
                    future_wrist_mask=batch["future_wrist_mask"],
                )
                loss = out["loss"]
            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss (epoch {epoch+1} batch {n_batches+1}), skipping")
                optim.zero_grad(set_to_none=True)
                continue
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
                )
            grads_ok = all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            if not grads_ok:
                print(f"  [warn] non-finite grad (epoch {epoch+1} step {n_batches+1}), skip step")
                optim.zero_grad(set_to_none=True)
                continue
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            if global_step % args.log_steps == 0:
                print(f"epoch {epoch+1} step {global_step} train_loss={loss.item():.5f}")

            if global_step % args.eval_steps == 0:
                eval_metrics = evaluate(model, monitor_loader, device)
                mae = eval_metrics["mae"]
                if mae == mae:
                    print(
                        f"  [eval step {global_step}] {monitor_tag}_mae={mae:.5f} "
                        f"{monitor_tag}_rmse={eval_metrics['rmse']:.5f}"
                    )
                    if mae < best_eval_mae:
                        best_eval_mae = mae
                        _save_checkpoint(model, cfg, args.output_dir, "best")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} avg_train_loss={avg_loss:.5f}")

        eval_metrics = evaluate(model, monitor_loader, device)
        print(
            f"  [epoch {epoch+1} {monitor_tag}] mae={eval_metrics['mae']:.5f} rmse={eval_metrics['rmse']:.5f} "
            f"left_mae={eval_metrics['left_mae']:.5f} right_mae={eval_metrics['right_mae']:.5f}"
        )
        metrics_path = os.path.join(
            args.output_dir,
            f"{'val' if val_loader is not None else 'train'}_metrics_epoch{epoch+1}.json",
        )
        with open(metrics_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        if eval_metrics["mae"] == eval_metrics["mae"] and eval_metrics["mae"] < best_eval_mae:
            best_eval_mae = eval_metrics["mae"]
            _save_checkpoint(model, cfg, args.output_dir, "best")

        _save_checkpoint(model, cfg, args.output_dir, f"epoch{epoch+1}")

    _save_checkpoint(model, cfg, args.output_dir, "last")
    print(f"Done. Checkpoints in {args.output_dir}")


def _save_checkpoint(model: WristLlavaOneVisionModel, cfg: WristLlavaOVConfig, output_dir: str, tag: str) -> None:
    path = os.path.join(output_dir, f"wrist_llava_ov_{tag}.pt")
    state = {
        "wrist_encoder": model.wrist_encoder.state_dict(),
        "head": model.head.state_dict(),
    }
    if hasattr(model, "video_ctx_norm"):
        state["video_ctx_norm"] = model.video_ctx_norm.state_dict()
    payload = {
        "wrist_head": state,
        "config": cfg.__dict__,
        "full_finetune": not cfg.freeze_llava,
        "norm_stats": model.norm_stats.to_dict() if model.norm_stats is not None else None,
    }
    if not cfg.freeze_llava:
        payload["llava"] = model.llava.state_dict()
    torch.save(payload, path)
    print(f"  saved {path}" + (" (full LLaVA + wrist head)" if not cfg.freeze_llava else " (wrist head only)"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = _repo_root()
    p = argparse.ArgumentParser(description="LLaVA-OneVision wrist SFT (video modality)")
    p.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    p.add_argument("--data_root", type=str, default=str(root / "data"))
    p.add_argument("--output_dir", type=str, default=str(root / "outputs" / "wrist_llava_ov_train"))
    p.add_argument("--future_k", type=int, default=16)
    p.add_argument("--max_history", type=int, default=0)
    p.add_argument("--frames_upbound", type=int, default=8, help="Max video frames (LLaVA uniform sample)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4, help="LR for wrist head (and all params if frozen LLaVA)")
    p.add_argument("--llava_lr", type=float, default=1e-5, help="LR for LLaVA backbone when full finetune")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help="Fraction of episodes for validation; 0 = use all data for training",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--freeze_llava",
        action="store_true",
        help="Freeze LLaVA backbone; default is full-parameter finetune",
    )
    p.add_argument("--gradient_checkpointing", action="store_true", help="Save VRAM when training LLaVA")
    p.add_argument("--no_amp", action="store_true", help="Disable autocast (fp16) during training")
    p.add_argument("--log_steps", type=int, default=5)
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--norm_stats",
        type=str,
        default=str(root / "outputs" / "wrist_norm_stats.json"),
        help="Wrist xyz mean/std JSON (computed on data/ if missing)",
    )
    p.add_argument("--compute_norm", action="store_true", help="Recompute norm stats even if file exists")
    p.add_argument("--no_normalize", action="store_true", help="Train in raw meters (no xyz normalization)")
    return p.parse_args(argv)


if __name__ == "__main__":
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    train(parse_args())
