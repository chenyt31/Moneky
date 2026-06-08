"""
Single-GPU overfit trainer for VITRA + LLaVA-OneVision-2 (history codec video + state -> hand actions).

Freezes the LLaVA backbone by default and trains the DiT action head (+ cognition token / fov).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VITRA_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VITRA_ROOT) not in sys.path:
    sys.path.insert(0, str(VITRA_ROOT))

from vitra.utils.hf_env import enable_hf_offline

enable_hf_offline()

import torch
from torch.utils.data import DataLoader

from vitra.datasets.llava_ov2_collator import LlavaOV2HandCollator, build_llava_ov2_collator
from vitra.datasets.llava_ov2_dataset import LlavaOV2HumanDataset
from vitra.models.vla_builder import build_vla


def _vision_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _to_device(batch: dict, device: torch.device) -> dict:
    vdtype = _vision_dtype(device)
    out = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            out[k] = v
        elif k == "pixel_values":
            out[k] = v.to(device, dtype=vdtype)
        else:
            out[k] = v.to(device)
    return out


def _n_trainable(model) -> tuple[int, int]:
    bb = sum(p.numel() for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad)
    head = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad)
    return bb, head


def _build_optimizer(model, cfg: dict) -> torch.optim.AdamW:
    trainer_cfg = cfg.get("trainer", {})
    weight_decay = trainer_cfg.get("weight_decay", 0.0)
    head_lr = trainer_cfg.get("action_model_learning_rate", 1e-4)
    bb_lr = trainer_cfg.get("learning_rate", 1e-5)
    freeze_option = cfg.get("train_setup", {}).get("freeze_option", "only_head_and_token")

    bb_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]

    if freeze_option == "full_finetune" and bb_params:
        print(f"optimizer: backbone lr={bb_lr} ({len(bb_params)} tensors), head lr={head_lr} ({len(head_params)} tensors)")
        return torch.optim.AdamW(
            [
                {"params": bb_params, "lr": bb_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )

    params = head_params or [p for p in model.parameters() if p.requires_grad]
    print(f"optimizer: head-only lr={head_lr} ({len(params)} tensors)")
    return torch.optim.AdamW(params, lr=head_lr, weight_decay=weight_decay)


def _maybe_enable_gradient_checkpointing(model, cfg: dict) -> None:
    if not cfg.get("trainer", {}).get("enable_gradient_checkpointing", False):
        return
    if cfg.get("train_setup", {}).get("freeze_option") != "full_finetune":
        return
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
        print("enabled backbone gradient checkpointing", flush=True)


@torch.no_grad()
def evaluate(model, loader, device, *, max_batches: int | None = None) -> dict:
    model.eval()
    model.backbone.eval()
    total = 0.0
    n = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = _to_device(batch, device)
        out = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image_grid_thw=batch["image_grid_thw"],
            patch_positions=batch["patch_positions"],
            action_labels=batch["actions"],
            action_masks=batch["action_masks"],
            current_state=batch["current_state"],
            current_state_mask=batch["current_state_mask"],
            fov=batch["fov"],
            mode="train",
        )
        loss = out["loss"]
        if torch.isfinite(loss):
            total += loss.item()
            n += 1
        print(f"  eval batch {batch_idx + 1} loss={loss.item():.6f}", flush=True)
    return {"eval_loss": total / max(n, 1), "eval_batches": n}


def _build_collator(cfg: dict, processor, output_dir: str) -> LlavaOV2HandCollator:
    cache_root = os.path.join(output_dir, "codec_cache")
    os.makedirs(cache_root, exist_ok=True)
    return build_llava_ov2_collator(cfg, processor, cache_root)


def prewarm_codec_cache(dataset, collator: LlavaOV2HandCollator) -> None:
    print(f"=== Pre-warming codec cache for {len(dataset)} samples ===", flush=True)
    for idx in range(len(dataset)):
        collator([dataset[idx]])
        if (idx + 1) % 4 == 0 or idx + 1 == len(dataset):
            print(f"  codec cache {idx + 1}/{len(dataset)}", flush=True)
    print("=== Codec cache ready ===", flush=True)


def save_init_checkpoint(model, cfg: dict, output_dir: str) -> str:
    path = os.path.join(output_dir, "vitra_llava_ov2_init.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": 0,
            "eval_loss": None,
            "tag": "init_before_training",
        },
        path,
    )
    return path


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda"

    with open(args.config) as f:
        cfg = json.load(f)

    ds_cfg = cfg["train_dataset"]
    dataset = LlavaOV2HumanDataset(
        data_root=ds_cfg["data_root_dir"],
        statistics_path=ds_cfg["statistics_path"],
        action_future_window_size=ds_cfg.get("action_future_window_size", 15),
        max_samples=ds_cfg.get("max_samples", 0),
        augmentation=ds_cfg.get("augmentation", False),
        normalization=ds_cfg.get("normalization", True),
    )

    model = build_vla(cfg)
    model.trainable_params_setup()
    _maybe_enable_gradient_checkpointing(model, cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    collator = _build_collator(cfg, model.processor, args.output_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
    )
    model.use_bf16 = cfg.get("use_bf16", True)
    model.to(device)
    model.backbone.to(device)
    bb_n, head_n = _n_trainable(model)
    print(f"samples={len(dataset)} backbone_trainable={bb_n/1e6:.2f}M head_trainable={head_n/1e6:.2f}M")

    if not args.skip_prewarm:
        prewarm_codec_cache(dataset, collator)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = _build_optimizer(model, cfg)

    init_ckpt = save_init_checkpoint(model, cfg, args.output_dir)
    baseline_max_batches = cfg["trainer"].get("baseline_max_batches")
    if baseline_max_batches in (0, "0"):
        baseline_max_batches = None
    print(f"=== Baseline eval (max_batches={baseline_max_batches}) ===", flush=True)
    baseline = evaluate(model, loader, device, max_batches=baseline_max_batches)
    print(f"baseline eval_loss={baseline['eval_loss']:.6f} (init ckpt -> {init_ckpt})")

    history = {
        "config": args.config,
        "num_samples": len(dataset),
        "epochs": args.epochs,
        "baseline": {"epoch": 0, **baseline},
        "epochs_log": [],
    }

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        if cfg["train_setup"].get("freeze_option") != "full_finetune":
            model.backbone.eval()
        epoch_loss = 0.0
        n = 0
        for batch_idx, batch in enumerate(loader):
            batch = _to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=_vision_dtype(device), enabled=use_amp):
                out = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    image_grid_thw=batch["image_grid_thw"],
                    patch_positions=batch["patch_positions"],
                    action_labels=batch["actions"],
                    action_masks=batch["action_masks"],
                    current_state=batch["current_state"],
                    current_state_mask=batch["current_state_mask"],
                    fov=batch["fov"],
                    mode="train",
                )
                loss = out["loss"]
            if not torch.isfinite(loss):
                print("skip non-finite loss", flush=True)
                continue
            loss.backward()
            if cfg["trainer"].get("gradient_clip_val", 0) > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg["trainer"]["gradient_clip_val"])
            optim.step()
            epoch_loss += loss.item()
            n += 1
            print(
                f"  train batch {batch_idx + 1}/{len(loader)} loss={loss.item():.6f}",
                flush=True,
            )

        print(f"=== Epoch {epoch+1} eval ===", flush=True)
        metrics = evaluate(model, loader, device, max_batches=baseline_max_batches)
        avg = epoch_loss / max(n, 1)
        print(f"epoch {epoch+1}/{args.epochs} train_loss={avg:.6f} eval_loss={metrics['eval_loss']:.6f}")
        history["epochs_log"].append(
            {
                "epoch": epoch + 1,
                "train_loss": avg,
                **metrics,
            }
        )
        if metrics["eval_loss"] < best_loss:
            best_loss = metrics["eval_loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch + 1,
                    "eval_loss": metrics["eval_loss"],
                },
                os.path.join(args.output_dir, "vitra_llava_ov2_best.pt"),
            )

    torch.save(
        {"model": model.state_dict(), "config": cfg, "epoch": args.epochs, "eval_loss": metrics["eval_loss"]},
        os.path.join(args.output_dir, "vitra_llava_ov2_last.pt"),
    )
    history["best_eval_loss"] = best_loss
    history["final"] = history["epochs_log"][-1] if history["epochs_log"] else baseline
    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done -> {args.output_dir}")
    print(
        f"baseline eval_loss={baseline['eval_loss']:.6f} -> best eval_loss={best_loss:.6f} "
        f"(delta={baseline['eval_loss'] - best_loss:.6f})"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=str(VITRA_ROOT / "vitra/configs/human_llava_ov2_overfit.json"),
    )
    p.add_argument("--output_dir", default=str(VITRA_ROOT / "outputs/vitra_llava_ov2_overfit"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip_prewarm", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
