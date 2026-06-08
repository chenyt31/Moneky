"""
Memorize trainer: freeze LLaVA backbone, aggressively train DiT head until sampled action MAE
on the training set approaches zero (cfg_scale=1, no CFG at inference).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VITRA_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VITRA_ROOT) not in sys.path:
    sys.path.insert(0, str(VITRA_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from vitra.utils.hf_env import enable_hf_offline

enable_hf_offline()

import torch
from torch.utils.data import DataLoader

from vitra.datasets.llava_ov2_collator import build_llava_ov2_collator
from vitra.datasets.llava_ov2_dataset import LlavaOV2HumanDataset
from vitra.models.vla_builder import build_vla
from vitra.utils.memorize_eval import evaluate_action_mae
from train_llava_ov2_overfit import (
    _build_optimizer,
    _n_trainable,
    _to_device,
    _vision_dtype,
    prewarm_codec_cache,
    save_init_checkpoint,
)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda"

    with open(args.config) as f:
        cfg = json.load(f)

    ds_cfg = cfg["train_dataset"]
    max_samples = args.max_samples if args.max_samples is not None else ds_cfg.get("max_samples", 0)
    dataset = LlavaOV2HumanDataset(
        data_root=ds_cfg["data_root_dir"],
        statistics_path=ds_cfg["statistics_path"],
        action_future_window_size=ds_cfg.get("action_future_window_size", 15),
        max_samples=max_samples,
        augmentation=False,
        normalization=ds_cfg.get("normalization", True),
    )

    model = build_vla(cfg)
    model.trainable_params_setup()
    os.makedirs(args.output_dir, exist_ok=True)
    collator = build_llava_ov2_collator(cfg, model.processor, os.path.join(args.output_dir, "codec_cache"))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        drop_last=False,
    )

    model.use_bf16 = cfg.get("use_bf16", True)
    model.to(device)
    model.backbone.to(device)
    bb_n, head_n = _n_trainable(model)
    trainer_cfg = cfg["trainer"]
    target_mae = float(trainer_cfg.get("target_action_mae_norm", 0.05))
    mae_every = int(trainer_cfg.get("action_mae_every", 5))

    print(
        f"memorize: samples={len(dataset)} backbone_trainable={bb_n/1e6:.2f}M "
        f"head_trainable={head_n/1e6:.2f}M repeated_diffusion_steps={cfg.get('repeated_diffusion_steps')} "
        f"head_lr={trainer_cfg.get('action_model_learning_rate')} cfg_scale={trainer_cfg.get('predict_cfg_scale', 1.0)}",
        flush=True,
    )

    if not args.skip_prewarm:
        prewarm_codec_cache(dataset, collator)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = _build_optimizer(model, cfg)
    init_ckpt = save_init_checkpoint(model, cfg, args.output_dir)

    run_baseline_mae = not args.skip_baseline_mae
    if run_baseline_mae:
        print("=== Baseline action MAE (sampled, cfg=1) ===", flush=True)
        baseline_mae = evaluate_action_mae(model, dataset, collator, device, cfg, verbose=True)
        print(
            f"baseline action_mae_norm={baseline_mae['action_mae_norm_mean']:.6f} (init -> {init_ckpt})",
            flush=True,
        )
    else:
        baseline_mae = {"action_mae_norm_mean": None}
        print("=== Skip baseline action MAE (codec cache already warm) ===", flush=True)

    history = {
        "config": args.config,
        "num_samples": len(dataset),
        "epochs": args.epochs,
        "target_action_mae_norm": target_mae,
        "baseline_action_mae": baseline_mae,
        "epochs_log": [],
    }

    best_mae = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
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
            if trainer_cfg.get("gradient_clip_val", 0) > 0:
                torch.nn.utils.clip_grad_norm_(params, trainer_cfg["gradient_clip_val"])
            optim.step()
            epoch_loss += loss.item()
            n += 1
            if (batch_idx + 1) % 16 == 0 or batch_idx + 1 == len(loader):
                print(
                    f"  epoch {epoch+1} train {batch_idx + 1}/{len(loader)} loss={loss.item():.6f}",
                    flush=True,
                )

        avg = epoch_loss / max(n, 1)
        log_entry = {"epoch": epoch + 1, "train_loss": avg}
        run_mae = (epoch + 1) % mae_every == 0 or epoch + 1 == args.epochs
        if run_mae:
            print(f"=== Epoch {epoch+1} action MAE eval ===", flush=True)
            mae_metrics = evaluate_action_mae(model, dataset, collator, device, cfg, verbose=True)
            log_entry.update(
                {
                    "action_mae_norm_mean": mae_metrics["action_mae_norm_mean"],
                    "action_mae_norm_max": mae_metrics["action_mae_norm_max"],
                    "action_mae_norm_p95": mae_metrics["action_mae_norm_p95"],
                }
            )
            if mae_metrics["action_mae_norm_mean"] < best_mae:
                best_mae = mae_metrics["action_mae_norm_mean"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg,
                        "epoch": epoch + 1,
                        "action_mae_norm_mean": best_mae,
                        "tag": "best_action_mae",
                    },
                    os.path.join(args.output_dir, "vitra_llava_ov2_memorize_best.pt"),
                )
                print(f"  new best action_mae_norm={best_mae:.6f}", flush=True)
            if best_mae <= target_mae:
                print(f"  reached target action_mae_norm <= {target_mae}", flush=True)
                history["epochs_log"].append(log_entry)
                break
        else:
            print(f"epoch {epoch+1}/{args.epochs} train_loss={avg:.6f}", flush=True)

        history["epochs_log"].append(log_entry)

    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": history["epochs_log"][-1]["epoch"] if history["epochs_log"] else args.epochs,
            "tag": "last",
        },
        os.path.join(args.output_dir, "vitra_llava_ov2_memorize_last.pt"),
    )

    history["best_action_mae_norm"] = best_mae
    history["best_epoch"] = best_epoch
    history["final"] = history["epochs_log"][-1] if history["epochs_log"] else {}
    with open(os.path.join(args.output_dir, "memorize_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Done -> {args.output_dir}", flush=True)
    print(
        f"baseline action_mae={baseline_mae['action_mae_norm_mean']:.6f} -> "
        f"best action_mae={best_mae:.6f} (epoch {best_epoch})",
        flush=True,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Memorize train set actions (head-only, cfg=1 eval)")
    p.add_argument(
        "--config",
        default=str(VITRA_ROOT / "vitra/configs/human_llava_ov2_memorize.json"),
    )
    p.add_argument("--output_dir", default=str(VITRA_ROOT / "outputs/vitra_llava_ov2_memorize"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--max_samples", type=int, default=None, help="Override config max_samples; 0 = all")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip_prewarm", action="store_true")
    p.add_argument(
        "--skip_baseline_mae",
        action="store_true",
        help="Skip baseline action MAE (reuse existing codec cache on restart)",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
