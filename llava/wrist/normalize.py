"""Wrist XYZ normalization from dataset statistics (camera frame, meters)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from datasets.epoch_reader import WristEpisodeReader


@dataclass
class WristNormStats:
    """Per-axis mean/std over all valid wrist positions in camera frame."""

    mean: np.ndarray  # (3,)
    std: np.ndarray  # (3,)
    n_valid: int = 0
    min_xyz: Optional[np.ndarray] = None
    max_xyz: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_valid": self.n_valid,
            "min_xyz": self.min_xyz.tolist() if self.min_xyz is not None else None,
            "max_xyz": self.max_xyz.tolist() if self.max_xyz is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WristNormStats":
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float32),
            std=np.asarray(d["std"], dtype=np.float32),
            n_valid=int(d.get("n_valid", 0)),
            min_xyz=np.asarray(d["min_xyz"], dtype=np.float32) if d.get("min_xyz") is not None else None,
            max_xyz=np.asarray(d["max_xyz"], dtype=np.float32) if d.get("max_xyz") is not None else None,
        )

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WristNormStats":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(self.mean, device=device, dtype=torch.float32)
        std = torch.as_tensor(self.std, device=device, dtype=torch.float32)
        return mean, std


def compute_wrist_norm_stats(data_root: str = "data", eps: float = 1e-6) -> WristNormStats:
    """
    Compute mean/std over every valid left/right wrist xyz in all episodes.
    """
    reader = WristEpisodeReader(data_root=data_root)
    values = []
    for ep_idx in range(len(reader.episode_pairs)):
        ep = reader.load(ep_idx)
        for t in range(ep.num_frames):
            for h in range(2):
                if ep.wrist_mask[t, h]:
                    values.append(ep.wrists[t, h])

    if not values:
        raise ValueError(f"No valid wrist points under {data_root}")

    arr = np.stack(values, axis=0).astype(np.float64)
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps)

    return WristNormStats(
        mean=mean,
        std=std,
        n_valid=len(values),
        min_xyz=arr.min(axis=0).astype(np.float32),
        max_xyz=arr.max(axis=0).astype(np.float32),
    )


def normalize_wrist_tensor(
    wrists: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    wrists: (..., 2, 3), mask: (..., 2) bool — only valid entries are scaled.
    """
    out = wrists.clone()
    m = mask.unsqueeze(-1)
    out = torch.where(m, (wrists - mean) / std, wrists)
    return out


def denormalize_wrist_tensor(
    wrists: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return wrists * std + mean


def main():
    import argparse

    p = argparse.ArgumentParser(description="Compute wrist normalization stats from data/")
    p.add_argument("--data_root", default="data")
    p.add_argument("--output", default="outputs/wrist_norm_stats.json")
    args = p.parse_args()

    stats = compute_wrist_norm_stats(args.data_root)
    stats.save(args.output)
    print(f"Saved {args.output}")
    print(f"  n_valid={stats.n_valid}")
    print(f"  mean={stats.mean}")
    print(f"  std={stats.std}")
    print(f"  min={stats.min_xyz}")
    print(f"  max={stats.max_xyz}")


if __name__ == "__main__":
    main()
