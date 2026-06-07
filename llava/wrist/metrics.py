"""Evaluation metrics for wrist trajectory prediction."""

from __future__ import annotations

from typing import Dict

import torch


def masked_wrist_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Masked L1 loss.

    Args:
        pred: (B, K, 2, 3)
        target: (B, K, 2, 3)
        mask: (B, K, 2) bool, True where target is valid
    """
    if mask.sum() == 0:
        return pred.sum() * 0.0
    if not torch.isfinite(pred).all():
        return pred.sum() * float("nan")
    # NaN in target must be cleared before diff — (nan - x) * 0 is still nan in PyTorch.
    target_safe = target.nan_to_num(0.0)
    diff = (pred - target_safe).abs()
    diff = diff * mask.unsqueeze(-1)
    if reduction == "mean":
        return diff.sum() / mask.sum().clamp(min=1) / 3
    return diff.sum()


@torch.no_grad()
def compute_wrist_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute masked MAE / RMSE (meters) over valid future wrist targets.

    Returns dict with keys: mae, rmse, n_valid, left_mae, right_mae.
    """
    metrics: Dict[str, float] = {}
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n_valid": 0.0, "left_mae": float("nan"), "right_mae": float("nan")}
    if not torch.isfinite(pred).all():
        return {"mae": float("nan"), "rmse": float("nan"), "n_valid": float(mask.sum().item()), "left_mae": float("nan"), "right_mae": float("nan")}

    target_safe = target.nan_to_num(0.0)
    pred_safe = pred.nan_to_num(0.0)
    err = (pred_safe - target_safe) * mask.unsqueeze(-1)
    sq = err.pow(2)
    n = mask.sum().item() * 3
    metrics["mae"] = err.abs().sum().item() / n
    metrics["rmse"] = (sq.sum().item() / n) ** 0.5
    metrics["n_valid"] = float(mask.sum().item())

    for hand_idx, name in enumerate(("left_mae", "right_mae")):
        hand_mask = mask[:, :, hand_idx]
        if hand_mask.sum() == 0:
            metrics[name] = float("nan")
            continue
        hand_err = (pred_safe[:, :, hand_idx] - target_safe[:, :, hand_idx]).abs()
        metrics[name] = (hand_err * hand_mask.unsqueeze(-1)).sum().item() / (hand_mask.sum().item() * 3)

    return metrics
