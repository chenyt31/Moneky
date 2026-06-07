"""Draw future-K GT / pred wrist trajectories only."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GT_COLOR = (0, 220, 0)
PRED_COLOR = (255, 40, 40)


def project_cam_to_pixel(points_xyz: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """(N,3) camera -> (N,2) uv; invalid -> NaN."""
    out = np.full((points_xyz.shape[0], 2), np.nan, dtype=np.float32)
    valid = np.isfinite(points_xyz).all(axis=1) & (points_xyz[:, 2] > 1e-4)
    if not np.any(valid):
        return out
    p = points_xyz[valid]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    out[valid, 0] = fx * p[:, 0] / p[:, 2] + cx
    out[valid, 1] = fy * p[:, 1] / p[:, 2] + cy
    return out


def _draw_hand_trajectory(
    draw: ImageDraw.ImageDraw,
    uv: np.ndarray,
    valid: np.ndarray,
    color: Tuple[int, int, int],
    w: int,
    h: int,
) -> None:
    """Polyline + dots for one hand over K future steps."""
    pts = []
    for i in range(len(uv)):
        if valid[i] and np.isfinite(uv[i]).all():
            u, v = float(uv[i, 0]), float(uv[i, 1])
            if 0 <= u < w and 0 <= v < h:
                pts.append((int(round(u)), int(round(v))))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=3)
    for p in pts:
        r = 5
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=color, outline=(255, 255, 255), width=1)


def render_future_traj_frame(
    frame_rgb: np.ndarray,
    fi: int,
    num_frames: int,
    gt_future: np.ndarray,
    pred_future: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    future_k: int,
) -> np.ndarray:
    """
    On frame ``fi`` (hist_end), draw only future K wrist trajectories.

    gt_future / pred_future: (K, 2, 3) camera coords for frames fi+1 .. fi+K
    gt_mask / pred_mask: (K, 2) bool
    """
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    h, w = frame_rgb.shape[:2]

    for hi in range(2):
        gt_uv = project_cam_to_pixel(gt_future[:, hi, :], intrinsics)
        _draw_hand_trajectory(draw, gt_uv, gt_mask[:, hi], GT_COLOR, w, h)
        pred_uv = project_cam_to_pixel(pred_future[:, hi, :], intrinsics)
        _draw_hand_trajectory(draw, pred_uv, pred_mask[:, hi], PRED_COLOR, w, h)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((4, 4, 380, 50), fill=(0, 0, 0))
    valid_k = int(np.sum(np.any(gt_mask, axis=1) | np.any(pred_mask, axis=1)))
    if valid_k <= 0:
        valid_k = int(np.sum(np.any(gt_mask, axis=1)))
    draw.text(
        (8, 6),
        f"frame {fi}/{num_frames - 1}  (next {valid_k}/{future_k} valid steps)",
        fill=(255, 255, 255),
        font=font,
    )
    draw.text((8, 26), "GT future green | Pred future red", fill=(200, 200, 200), font=font)
    return np.asarray(img)
