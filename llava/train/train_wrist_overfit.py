"""
Overfit wrist trajectory model on tiny data (e.g. 2 episodes / 130 samples).

Modes:
  mlp        — full history wrists only (fast, should reach ~0 train MAE)
  llava      — frozen LLaVA video_ctx cache + fat MLP
  llava_full — end-to-end full-parameter LLaVA + wrist head (no feature cache)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from datasets.wrist_video_sft import WristVideoSFTCollator, WristVideoSFTDataset, discover_episode_pairs
from llava.wrist.collator import WristLlavaCollator
from llava.wrist.metrics import compute_wrist_metrics
from llava.wrist.model import WristLlavaOVConfig, WristLlavaOneVisionModel
from llava.wrist.normalize import WristNormStats, compute_wrist_norm_stats
from llava.wrist.overfit_model import WristOverfitConfig, WristOverfitMLP


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class OverfitCollator:
    def __init__(self, video_cache: dict[int, torch.Tensor] | None = None, pad_to_max_hist: int = 0):
        self.wrist = WristVideoSFTCollator(pad_to_max_hist=pad_to_max_hist)
        self.video_cache = video_cache

    def __call__(self, instances):
        batch = self.wrist(instances)
        if self.video_cache is not None:
            batch["video_ctx"] = torch.stack(
                [self.video_cache[int(i["sample_idx"])].float() for i in instances], dim=0
            )
        return batch


@torch.no_grad()
def build_video_cache(
    model_name: str,
    dataset: WristVideoSFTDataset,
    device: torch.device,
    *,
    frames_upbound: int,
    future_k: int,
) -> dict[int, torch.Tensor]:
    processor = AutoProcessor.from_pretrained(model_name)
    ov = WristLlavaOneVisionModel(WristLlavaOVConfig(model_name_or_path=model_name, freeze_llava=True))
    ov.llava.to(device)
    ov.eval()

    collator = WristLlavaCollator(processor=processor, frames_upbound=frames_upbound, future_k=future_k)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
    cache: dict[int, torch.Tensor] = {}
    for batch in loader:
        batch = {
            k: v.to(device, dtype=torch.float16) if k == "pixel_values_videos" and torch.is_tensor(v)
            else v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        ctx = ov._encode_video_context(
            batch["pixel_values_videos"], batch["input_ids"], batch["attention_mask"]
        )
        sid = int(batch["sample_idx"][0].item())
        cache[sid] = ctx[0].cpu()
    return cache


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    agg = {"mae": 0.0, "rmse": 0.0, "batches": 0}
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        kwargs = dict(
            history_wrists=batch["history_wrists"],
            history_wrist_mask=batch["history_wrist_mask"],
            history_len=batch["history_len"],
            episode_idx=batch["episode_idx"],
            hist_ends=batch["hist_ends"],
            future_wrists=batch["future_wrists"],
            future_wrist_mask=batch["future_wrist_mask"],
        )
        if "video_ctx" in batch:
            kwargs["video_ctx"] = batch["video_ctx"]
        out = model(**kwargs)
        m = compute_wrist_metrics(out["pred"], batch["future_wrists"], batch["future_wrist_mask"])
        if m["mae"] == m["mae"]:
            agg["mae"] += m["mae"]
            agg["rmse"] += m["rmse"]
        agg["batches"] += 1
    if agg["batches"]:
        agg["mae"] /= agg["batches"]
        agg["rmse"] /= agg["batches"]
    return agg


def _save_ckpt(
    output_dir: str,
    filename: str,
    model: WristOverfitMLP,
    cfg: WristOverfitConfig,
    args: argparse.Namespace,
    norm_stats: WristNormStats | None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "mode": args.mode,
        "model_name_or_path": args.model_name_or_path if args.mode == "llava" else None,
        "norm_stats": norm_stats.to_dict() if norm_stats is not None else None,
    }
    torch.save(payload, os.path.join(output_dir, filename))


def _to_device_llava(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if k == "pixel_values_videos":
                out[k] = v.to(device, dtype=torch.float16)
            else:
                out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_llava(model: WristLlavaOneVisionModel, loader, device) -> dict:
    model.eval()
    if model.config.freeze_llava:
        model.llava.eval()
    agg = {"mae": 0.0, "rmse": 0.0, "batches": 0}
    for batch in loader:
        batch = _to_device_llava(batch, device)
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
        if m["mae"] == m["mae"]:
            agg["mae"] += m["mae"]
            agg["rmse"] += m["rmse"]
        agg["batches"] += 1
    if agg["batches"]:
        agg["mae"] /= agg["batches"]
        agg["rmse"] /= agg["batches"]
    return agg


def _save_llava_ckpt(output_dir: str, filename: str, model: WristLlavaOneVisionModel, cfg: WristLlavaOVConfig) -> None:
    payload = {
        "wrist_head": {
            "wrist_encoder": model.wrist_encoder.state_dict(),
            "head": model.head.state_dict(),
            "video_ctx_norm": model.video_ctx_norm.state_dict(),
        },
        "llava": model.llava.state_dict(),
        "config": cfg.__dict__,
        "full_finetune": True,
        "mode": "llava_full",
    }
    torch.save(payload, os.path.join(output_dir, filename))


def train_llava_full(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda"

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    ds = WristVideoSFTDataset(
        data_root=args.data_root,
        future_k=args.future_k,
        max_history=args.max_history,
        load_video=True,
    )
    collator = WristLlavaCollator(
        processor=processor,
        frames_upbound=args.frames_upbound,
        future_k=args.future_k,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
    )

    cfg = WristLlavaOVConfig(
        model_name_or_path=args.model_name_or_path,
        future_k=args.future_k,
        freeze_llava=False,
    )
    model = WristLlavaOneVisionModel(cfg)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    if device.type == "cuda":
        model.llava.to(device)
    model.wrist_encoder.to(device=device)
    model.head.to(device=device)
    model.video_ctx_norm.to(device=device)

    optim = torch.optim.AdamW(
        model.trainable_parameter_groups(lr=args.lr, llava_lr=args.llava_lr, weight_decay=0.0)
    )
    llava_n, head_n = model.n_trainable_params()
    os.makedirs(args.output_dir, exist_ok=True)
    best_mae = float("inf")

    print(
        f"Mode=llava_full samples={len(ds)} "
        f"LLaVA={llava_n/1e6:.1f}M head={head_n/1e6:.2f}M lr={args.lr} llava_lr={args.llava_lr}"
    )

    for epoch in range(args.epochs):
        model.train()
        model.llava.train()
        epoch_loss = 0.0
        n = 0
        for batch in loader:
            batch = _to_device_llava(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
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
                continue
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
                )
            optim.step()
            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / max(n, 1)
        do_eval = (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        if do_eval:
            metrics = evaluate_llava(model, loader, device)
            print(f"epoch {epoch+1}/{args.epochs} loss={avg:.6f} train_mae={metrics['mae']:.6f}")
            if metrics["mae"] == metrics["mae"] and metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                _save_llava_ckpt(args.output_dir, "wrist_overfit_best.pt", model, cfg)
            if metrics["mae"] < args.target_mae:
                print(f"Reached target MAE < {args.target_mae} m")
                break
        else:
            print(f"epoch {epoch+1}/{args.epochs} loss={avg:.6f}")

    _save_llava_ckpt(args.output_dir, "wrist_overfit_last.pt", model, cfg)
    final = evaluate_llava(model, loader, device)
    print(f"Done llava_full. Final MAE={final['mae']:.6f} m -> {args.output_dir}")


def train(args: argparse.Namespace) -> None:
    if args.mode == "llava_full":
        train_llava_full(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pairs = discover_episode_pairs(args.data_root)
    n_ep = len(pairs)

    ds = WristVideoSFTDataset(
        data_root=args.data_root,
        future_k=args.future_k,
        max_history=args.max_history,
        load_video=args.mode == "llava",
    )
    t_max = max(s["hist_len"] for s in ds.samples)

    video_cache = None
    video_dim = 0
    if args.mode == "llava":
        cache_path = args.video_cache_path or os.path.join(
            args.output_dir, "video_ctx_cache.pt"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        if os.path.isfile(cache_path) and not args.rebuild_cache:
            print(f"Load video cache: {cache_path}")
            video_cache = torch.load(cache_path, map_location="cpu")["cache"]
        else:
            print("Building LLaVA video_ctx cache (one-time)...")
            video_cache = build_video_cache(
                args.model_name_or_path, ds, device,
                frames_upbound=args.frames_upbound, future_k=args.future_k,
            )
            torch.save({"cache": video_cache, "future_k": args.future_k}, cache_path)
            print(f"Saved cache: {cache_path}")
        video_dim = int(next(iter(video_cache.values())).numel())

    norm_stats: WristNormStats | None = None
    if args.no_normalize:
        print("Training without wrist normalization.")
    else:
        norm_path = Path(args.norm_stats)
        if args.compute_norm or not norm_path.is_file():
            print(f"Computing wrist norm stats from {args.data_root} ...")
            norm_stats = compute_wrist_norm_stats(args.data_root)
            norm_stats.save(norm_path)
            print(f"  mean={norm_stats.mean} std={norm_stats.std} n={norm_stats.n_valid}")
        else:
            norm_stats = WristNormStats.load(norm_path)
            print(f"Loaded norm stats: {norm_path}")
        print(f"  mean={norm_stats.mean} std={norm_stats.std}")

    cfg = WristOverfitConfig(
        future_k=args.future_k,
        max_history=t_max,
        hidden=args.hidden,
        depth=args.depth,
        n_episodes=max(n_ep, 2),
        video_ctx_dim=video_dim,
    )
    model = WristOverfitMLP(cfg, norm_stats=norm_stats).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    collator = OverfitCollator(video_cache=video_cache, pad_to_max_hist=t_max)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_mae = float("inf")

    print(f"Mode={args.mode} samples={len(ds)} t_max={t_max} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            kwargs = dict(
                history_wrists=batch["history_wrists"],
                history_wrist_mask=batch["history_wrist_mask"],
                history_len=batch["history_len"],
                episode_idx=batch["episode_idx"],
                hist_ends=batch["hist_ends"],
                future_wrists=batch["future_wrists"],
                future_wrist_mask=batch["future_wrist_mask"],
            )
            if "video_ctx" in batch:
                kwargs["video_ctx"] = batch["video_ctx"]
            out = model(**kwargs)
            loss = out["loss"]
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / max(n, 1)
        do_eval = (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        if do_eval:
            metrics = evaluate(model, loader, device)
            print(f"epoch {epoch+1}/{args.epochs} loss={avg:.6f} train_mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f}")
        else:
            metrics = {"mae": float("nan")}
            print(f"epoch {epoch+1}/{args.epochs} loss={avg:.6f}")

        if do_eval and metrics["mae"] == metrics["mae"] and metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            _save_ckpt(
                args.output_dir,
                "wrist_overfit_best.pt",
                model,
                cfg,
                args,
                norm_stats,
            )

        if do_eval and metrics["mae"] == metrics["mae"] and metrics["mae"] < args.target_mae:
            print(f"Reached target MAE < {args.target_mae} m, stopping.")
            break

    _save_ckpt(args.output_dir, "wrist_overfit_last.pt", model, cfg, args, norm_stats)
    final = evaluate(model, loader, device)
    print(f"Done. Final MAE={final['mae']:.6f} m  -> {args.output_dir}")


def parse_args():
    root = _repo_root()
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mlp", "llava", "llava_full"], default="mlp")
    p.add_argument("--data_root", default=str(root / "data"))
    p.add_argument("--output_dir", default=str(root / "outputs" / "wrist_overfit"))
    p.add_argument("--model_name_or_path", default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    p.add_argument("--video_cache_path", default=None)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--future_k", type=int, default=16)
    p.add_argument("--max_history", type=int, default=0)
    p.add_argument("--frames_upbound", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="Head LR (mlp/llava_full) or global LR (llava cache)")
    p.add_argument("--llava_lr", type=float, default=2e-5, help="LLaVA backbone LR for llava_full")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--target_mae", type=float, default=0.005, help="Stop when train MAE below this (meters)")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--norm_stats",
        default=str(root / "outputs" / "wrist_norm_stats.json"),
        help="Path to save/load wrist mean/std JSON",
    )
    p.add_argument("--compute_norm", action="store_true", help="Recompute norm stats even if file exists")
    p.add_argument("--no_normalize", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    if str(_repo_root()) not in sys.path:
        sys.path.insert(0, str(_repo_root()))
    train(parse_args())
