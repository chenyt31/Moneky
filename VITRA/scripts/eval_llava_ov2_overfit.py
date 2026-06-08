"""
Evaluate VITRA + LLaVA-OV2 checkpoint (history codec video + state -> hand actions).
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
from torch.utils.data import DataLoader, Subset

from vitra.datasets.llava_ov2_collator import build_llava_ov2_collator
from vitra.datasets.llava_ov2_dataset import LlavaOV2HumanDataset
from vitra.models.vla_builder import build_vla
from vitra.utils.data_utils import GaussianNormalizer, read_dataset_statistics
from vitra.utils.memorize_eval import predict_one
from vitra.visualization.llava_ov2_viz import render_sample_future_video


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


def load_model(checkpoint: str, config: dict, device: torch.device, *, fresh_init: bool = False):
    model = build_vla(config)
    if fresh_init:
        model.trainable_params_setup()
        ckpt = {"epoch": 0, "eval_loss": None, "tag": "fresh_init"}
    else:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    model.backbone.eval()
    return model, ckpt


@torch.no_grad()
def predict_batch(model, batch, device, cfg: dict):
    batch = _to_device(batch, device)
    b = batch["input_ids"].shape[0]
    preds = [predict_one(model, batch, device, cfg, sample_idx=i) for i in range(b)]
    return np.stack(preds, axis=0)


def action_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    return float(np.abs(pred[m] - gt[m]).mean())


def _parse_indices(text: str, n: int) -> list[int]:
    text = text.strip().lower()
    if not text or text == "all":
        return list(range(n))
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return [i for i in out if 0 <= i < n]


def _render_gt_viz(dataset, dataset_idx: int, normalizer: GaussianNormalizer, out_dir: str, mano_path: str) -> str:
    viz = dataset.get_norm_viz_sample(dataset_idx)
    out_mp4 = os.path.join(
        out_dir,
        f"sample_{dataset_idx:03d}_ep_{viz['episode_id']}_anchor{viz['frame_id']}_gt_future.mp4",
    )
    render_sample_future_video(
        video_path=viz["video_path"],
        anchor=int(viz["frame_id"]),
        norm_state=viz["norm_state"],
        norm_gt_actions=viz["norm_actions"],
        action_masks=viz["action_masks"],
        intrinsics=viz["intrinsics"],
        normalizer=normalizer,
        out_path=out_mp4,
        beta_left=viz["beta_left"],
        beta_right=viz["beta_right"],
        hand_state_mask=viz["hand_state_mask"],
        gt_only=True,
        mano_model_path=mano_path,
    )
    return out_mp4


def _render_pred_viz(
    dataset,
    dataset_idx: int,
    norm_pred: np.ndarray,
    normalizer: GaussianNormalizer,
    out_dir: str,
    mano_path: str,
) -> str:
    viz = dataset.get_norm_viz_sample(dataset_idx)
    out_mp4 = os.path.join(
        out_dir,
        f"sample_{dataset_idx:03d}_ep_{viz['episode_id']}_anchor{viz['frame_id']}_gt_pred_future.mp4",
    )
    render_sample_future_video(
        video_path=viz["video_path"],
        anchor=int(viz["frame_id"]),
        norm_state=viz["norm_state"],
        norm_gt_actions=viz["norm_actions"],
        action_masks=viz["action_masks"],
        intrinsics=viz["intrinsics"],
        normalizer=normalizer,
        out_path=out_mp4,
        beta_left=viz["beta_left"],
        beta_right=viz["beta_right"],
        hand_state_mask=viz["hand_state_mask"],
        norm_pred_actions=norm_pred,
        mano_model_path=mano_path,
    )
    return out_mp4


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    with open(args.config) as f:
        cfg = json.load(f)
    ds_cfg = cfg["train_dataset"]

    dataset = LlavaOV2HumanDataset(
        data_root=ds_cfg["data_root_dir"],
        statistics_path=ds_cfg["statistics_path"],
        action_future_window_size=ds_cfg.get("action_future_window_size", 15),
        max_samples=ds_cfg.get("max_samples", 32),
        augmentation=False,
        normalization=True,
    )
    normalizer = GaussianNormalizer(read_dataset_statistics(ds_cfg["statistics_path"]))

    eval_count = len(dataset) if args.num_eval <= 0 else min(args.num_eval, len(dataset))
    eval_indices = list(range(eval_count))
    gt_viz_indices = eval_indices if args.gt_viz_all else []
    pred_indices = _parse_indices(args.pred_indices, len(eval_indices))
    pred_indices = [eval_indices[i] for i in pred_indices if i < len(eval_indices)]

    os.makedirs(args.output_dir, exist_ok=True)
    gt_dir = os.path.join(args.output_dir, "gt_viz")
    pred_dir = os.path.join(args.output_dir, "pred_viz")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    if gt_viz_indices:
        print(f"=== GT mesh viz for {len(gt_viz_indices)} samples (norm->unnorm, no forward) ===")
        for bi in gt_viz_indices:
            path = _render_gt_viz(dataset, bi, normalizer, gt_dir, args.mano_path)
            print(f"  gt sample {bi} -> {path}")

    need_model = (not args.fresh_init or args.pred_indices) and len(pred_indices) > 0
    if not need_model and args.fresh_init:
        pred_indices = pred_indices or [eval_indices[0]]

    if len(pred_indices) == 0:
        print("No pred_indices selected; skipping model forward.")
        return

    model, ckpt = load_model(args.checkpoint, cfg, device, fresh_init=args.fresh_init)
    cache_root = os.path.join(args.output_dir, "codec_cache")
    collator = build_llava_ov2_collator(cfg, model.processor, cache_root)
    pred_loader = DataLoader(
        Subset(dataset, pred_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    maes, unnorm_maes = [], []
    print(f"=== Model forward + pred viz for {len(pred_indices)} samples ===")
    for bi, batch in zip(pred_indices, pred_loader):
        pred = predict_batch(model, batch, device, cfg)[0]
        gt = batch["actions"][0].numpy()
        mask = batch["action_masks"][0].numpy()
        mae = action_mae(pred, gt, mask)
        maes.append(mae)

        pred_u = pred.copy()
        gt_u = gt.copy()
        for i in range(pred_u.shape[0]):
            pred_u[i, :102] = normalizer.unnormalize_action(pred_u[i, :102])
            gt_u[i, :102] = normalizer.unnormalize_action(gt_u[i, :102])
        unnorm_maes.append(action_mae(pred_u, gt_u, mask))
        print(
            f"sample {bi} ep={batch['episode_id'][0]} anchor={int(batch['frame_id'][0])} "
            f"mae_norm={mae:.6f} mae_unnorm={unnorm_maes[-1]:.6f}"
        )

        viz_rank = pred_indices.index(bi)
        if args.max_pred_viz > 0 and viz_rank >= args.max_pred_viz:
            continue
        path = _render_pred_viz(dataset, bi, pred, normalizer, pred_dir, args.mano_path)
        print(f"  pred viz -> {path}")

    metrics = {
        "checkpoint": None if args.fresh_init else args.checkpoint,
        "fresh_init": args.fresh_init,
        "tag": args.tag,
        "pred_indices": pred_indices,
        "gt_viz_count": len(gt_viz_indices),
        "num_pred_samples": len(maes),
        "action_mae_norm_mean": float(np.nanmean(maes)) if maes else None,
        "action_mae_norm_std": float(np.nanstd(maes)) if maes else None,
        "action_mae_unnorm_mean": float(np.nanmean(unnorm_maes)) if unnorm_maes else None,
        "action_mae_unnorm_std": float(np.nanstd(unnorm_maes)) if unnorm_maes else None,
        "epoch": ckpt.get("epoch"),
        "eval_loss": ckpt.get("eval_loss"),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="")
    p.add_argument("--fresh_init", action="store_true", help="Evaluate randomly initialized action head")
    p.add_argument(
        "--config",
        default=str(VITRA_ROOT / "vitra/configs/human_llava_ov2_overfit.json"),
    )
    p.add_argument("--output_dir", default=str(VITRA_ROOT / "outputs/vitra_llava_ov2_eval"))
    p.add_argument("--num_eval", type=int, default=32, help="Number of samples to evaluate; <=0 means all")
    p.add_argument(
        "--pred_indices",
        default="0,1,2",
        help="Comma-separated dataset indices (within num_eval), or 'all'",
    )
    p.add_argument("--gt_viz_all", action="store_true", default=True)
    p.add_argument("--no_gt_viz_all", action="store_false", dest="gt_viz_all")
    p.add_argument("--max_pred_viz", type=int, default=8, help="Max pred overlay videos; <=0 means all")
    p.add_argument("--tag", default="")
    p.add_argument("--mano_path", default=str(VITRA_ROOT / "weights/mano"))
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if not args.fresh_init and not args.checkpoint:
        p.error("--checkpoint is required unless --fresh_init is set")
    return args


if __name__ == "__main__":
    main(parse_args())
