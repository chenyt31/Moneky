"""
Memorize eval: full training-set action reproduction check (sampled MAE, cfg_scale=1).
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

import numpy as np
import torch

from vitra.datasets.llava_ov2_collator import build_llava_ov2_collator
from vitra.datasets.llava_ov2_dataset import LlavaOV2HumanDataset
from vitra.models.vla_builder import build_vla
from vitra.utils.data_utils import GaussianNormalizer, read_dataset_statistics
from vitra.utils.memorize_eval import action_mae, evaluate_action_mae, predict_one


def load_model(checkpoint: str, config: dict, device: torch.device):
    model = build_vla(config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.use_bf16 = config.get("use_bf16", True)
    model.to(device)
    model.eval()
    model.backbone.eval()
    return model, ckpt


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
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
        normalization=True,
    )
    normalizer = GaussianNormalizer(read_dataset_statistics(ds_cfg["statistics_path"]))
    os.makedirs(args.output_dir, exist_ok=True)

    model, ckpt = load_model(args.checkpoint, cfg, device)
    collator = build_llava_ov2_collator(cfg, model.processor, os.path.join(args.output_dir, "codec_cache"))

    print(f"=== Memorize eval on {len(dataset)} training samples ===", flush=True)
    summary = evaluate_action_mae(model, dataset, collator, device, cfg, verbose=True)

    unnorm_maes = []
    for item in summary["per_sample"]:
        idx = item["index"]
        batch = collator([dataset[idx]])
        pred = predict_one(model, batch, device, cfg)
        gt = batch["actions"][0].numpy()
        mask = batch["action_masks"][0].numpy()
        pred_u = pred.copy()
        gt_u = gt.copy()
        for t in range(pred_u.shape[0]):
            pred_u[t, :102] = normalizer.unnormalize_action(pred_u[t, :102])
            gt_u[t, :102] = normalizer.unnormalize_action(gt_u[t, :102])
        item["action_mae_unnorm"] = action_mae(pred_u, gt_u, mask)
        unnorm_maes.append(item["action_mae_unnorm"])

    metrics = {
        "checkpoint": args.checkpoint,
        "num_samples": len(dataset),
        "predict_cfg_scale": cfg["trainer"].get("predict_cfg_scale", 1.0),
        "predict_ddim_steps": cfg["trainer"].get("predict_ddim_steps", 10),
        "action_mae_norm_mean": summary["action_mae_norm_mean"],
        "action_mae_norm_std": summary["action_mae_norm_std"],
        "action_mae_norm_max": summary["action_mae_norm_max"],
        "action_mae_norm_p95": summary["action_mae_norm_p95"],
        "action_mae_unnorm_mean": float(np.nanmean(unnorm_maes)),
        "action_mae_unnorm_max": float(np.nanmax(unnorm_maes)),
        "memorized": summary["action_mae_norm_mean"] <= float(cfg["trainer"].get("target_action_mae_norm", 0.05)),
        "epoch": ckpt.get("epoch"),
        "per_sample": summary["per_sample"],
    }
    out_path = os.path.join(args.output_dir, "memorize_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_sample"}, indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--config",
        default=str(VITRA_ROOT / "vitra/configs/human_llava_ov2_memorize.json"),
    )
    p.add_argument("--output_dir", default=str(VITRA_ROOT / "outputs/vitra_llava_ov2_memorize_eval"))
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
