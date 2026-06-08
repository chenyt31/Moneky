"""Action-level metrics for memorize / overfit runs (sampled actions, not diffusion noise MSE)."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import torch


def action_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    return float(np.abs(pred[m] - gt[m]).mean())


def predict_settings(cfg: dict) -> dict:
    trainer = cfg.get("trainer", {})
    return {
        "cfg_scale": float(trainer.get("predict_cfg_scale", 1.0)),
        "num_ddim_steps": int(trainer.get("predict_ddim_steps", 10)),
        "sample_times": int(trainer.get("predict_sample_times", 1)),
    }


SEQ_KEYS = frozenset({
    "input_ids",
    "attention_mask",
    "actions",
    "action_masks",
    "current_state",
    "current_state_mask",
    "fov",
    "intrinsics",
    "frame_id",
})


def _vision_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def slice_batch_one(batch: dict, index: int, device: torch.device) -> dict:
    vdtype = _vision_dtype(device)
    one = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            one[k] = v
        elif k in SEQ_KEYS:
            one[k] = v[index : index + 1].to(device)
        elif k == "pixel_values":
            one[k] = v.to(device, dtype=vdtype)
        else:
            one[k] = v.to(device)
    return one


@torch.no_grad()
def predict_one(
    model,
    batch: dict,
    device: torch.device,
    cfg: dict,
    *,
    sample_idx: int = 0,
) -> np.ndarray:
    one = slice_batch_one(batch, sample_idx, device)

    settings = predict_settings(cfg)
    samples = model.predict_action(
        pixel_values=one["pixel_values"],
        input_ids=one["input_ids"],
        attention_mask=one["attention_mask"],
        image_grid_thw=one["image_grid_thw"],
        patch_positions=one["patch_positions"],
        current_state=one["current_state"],
        current_state_mask=one["current_state_mask"],
        fov=one["fov"],
        action_mask_torch=one["action_masks"],
        cfg_scale=settings["cfg_scale"],
        num_ddim_steps=settings["num_ddim_steps"],
        sample_times=settings["sample_times"],
    )
    return samples[0]


@torch.no_grad()
def evaluate_action_mae(
    model,
    dataset,
    collator,
    device: torch.device,
    cfg: dict,
    *,
    indices: Optional[Iterable[int]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    model.eval()
    model.backbone.eval()
    if indices is None:
        index_list = list(range(len(dataset)))
    else:
        index_list = list(indices)

    norm_maes = []
    per_sample = []
    for rank, idx in enumerate(index_list):
        batch = collator([dataset[idx]])
        pred = predict_one(model, batch, device, cfg)
        gt = batch["actions"][0].numpy()
        mask = batch["action_masks"][0].numpy()
        mae = action_mae(pred, gt, mask)
        norm_maes.append(mae)
        per_sample.append({"index": int(idx), "action_mae_norm": mae})
        if verbose and ((rank + 1) % 16 == 0 or rank + 1 == len(index_list)):
            print(f"  action_mae progress {rank + 1}/{len(index_list)}", flush=True)

    arr = np.asarray(norm_maes, dtype=np.float64)
    summary = {
        "action_mae_norm_mean": float(np.nanmean(arr)),
        "action_mae_norm_std": float(np.nanstd(arr)),
        "action_mae_norm_max": float(np.nanmax(arr)),
        "action_mae_norm_p95": float(np.nanpercentile(arr, 95)),
        "num_samples": int(len(arr)),
        "per_sample": per_sample,
    }
    if verbose:
        print(
            f"  action_mae_norm mean={summary['action_mae_norm_mean']:.6f} "
            f"max={summary['action_mae_norm_max']:.6f} p95={summary['action_mae_norm_p95']:.6f}",
            flush=True,
        )
    return summary
