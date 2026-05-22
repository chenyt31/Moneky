#!/usr/bin/env python3
"""
Visualize a full wrist-video epoch:
  1) wrist XYZ + validity masks (left / right)
  2) overlay wrist points & spline-smoothed trajectories on video
"""

from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.epoch_reader import WristEpisode, WristEpisodeReader

HAND_NAMES = ("left", "right")
HAND_COLORS_RGB = {
    "left": (0, 180, 255),
    "right": (255, 80, 32),
}


def project_cam_to_pixel(points_xyz: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """(N,3) camera coords -> (N,2) pixel uv. NaN rows stay NaN."""
    out = np.full((points_xyz.shape[0], 2), np.nan, dtype=np.float32)
    valid = np.isfinite(points_xyz).all(axis=1) & (points_xyz[:, 2] > 1e-4)
    if not np.any(valid):
        return out
    p = points_xyz[valid]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = fx * p[:, 0] / p[:, 2] + cx
    v = fy * p[:, 1] / p[:, 2] + cy
    out[valid, 0] = u
    out[valid, 1] = v
    return out


def interpolate_wrist_spline(
    t: np.ndarray,
    wrists: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic spline through valid wrist samples; evaluate at all frame indices."""
    t = np.asarray(t, dtype=np.float64)
    out = np.full_like(wrists, np.nan)
    interp_mask = np.zeros(len(t), dtype=bool)

    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        return out, interp_mask
    if len(valid_idx) == 1:
        out[valid_idx] = wrists[valid_idx]
        return out, interp_mask

    tv = t[valid_idx].astype(np.float64)
    wv = wrists[valid_idx].astype(np.float64)
    if len(np.unique(tv)) < 2:
        out[valid_idx] = wv
        return out, interp_mask

    for dim in range(3):
        cs = CubicSpline(tv, wv[:, dim], extrapolate=False)
        out[:, dim] = cs(t)

    interp_mask[:] = True
    interp_mask[valid_idx] = False
    return out, interp_mask


def plot_wrist_and_mask(ep: WristEpisode, out_path: str) -> None:
    t = np.arange(ep.num_frames)
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Wrist cam-space + mask — {ep.video_name} (anno: {ep.anno_type})", fontsize=12)

    for col, (name, wrists, mask) in enumerate(
        zip(HAND_NAMES, [ep.left_wrists, ep.right_wrists], [ep.left_mask, ep.right_mask])
    ):
        color = "#{:02x}{:02x}{:02x}".format(*HAND_COLORS_RGB[name])
        for row, axis_label in enumerate(["x", "y", "z"]):
            ax = axes[row, col]
            y = wrists[:, row]
            ax.plot(t, y, color=color, linewidth=1.2, label=axis_label)
            ax.scatter(t[mask], y[mask], c=color, s=12, zorder=3)
            ax.set_ylabel(f"{name} {axis_label}")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"{name} wrist (camera frame)")
        ax_m = axes[3, col]
        ax_m.fill_between(t, 0, mask.astype(float), color=color, alpha=0.35, step="mid")
        ax_m.plot(t, mask.astype(float), color=color, drawstyle="steps-mid", linewidth=1.5)
        ax_m.set_ylim(-0.1, 1.1)
        ax_m.set_ylabel("valid mask")
        ax_m.set_xlabel("frame index")
        ax_m.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")


def _draw_trail_pil(
    draw: ImageDraw.ImageDraw,
    uv_path: np.ndarray,
    valid_path: np.ndarray,
    color: tuple,
    current_idx: int,
    trail_len: int,
    w: int,
    h: int,
) -> None:
    start = max(0, current_idx - trail_len)

    pts = []
    for i in range(start, current_idx + 1):
        if np.isfinite(uv_path[i]).all():
            u, v = int(round(uv_path[i, 0])), int(round(uv_path[i, 1]))
            if 0 <= u < w and 0 <= v < h:
                pts.append((u, v))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=2)

    for i in range(start, current_idx + 1):
        if valid_path[i] and np.isfinite(uv_path[i]).all():
            u, v = int(round(uv_path[i, 0])), int(round(uv_path[i, 1]))
            if 0 <= u < w and 0 <= v < h:
                r = 4
                draw.ellipse((u - r, v - r, u + r, v + r), fill=color, outline=color)

    if valid_path[current_idx] and np.isfinite(uv_path[current_idx]).all():
        u, v = int(round(uv_path[current_idx, 0])), int(round(uv_path[current_idx, 1]))
        if 0 <= u < w and 0 <= v < h:
            r = 6
            draw.ellipse((u - r, v - r, u + r, v + r), fill=color, outline=(255, 255, 255), width=2)


def render_overlay_video(ep: WristEpisode, out_path: str, trail_len: int = 40, fps: int = 10) -> None:
    t = np.arange(ep.num_frames, dtype=np.float64)
    h, w = ep.frames.shape[1:3]

    uv_spline = {}
    for hi, name in enumerate(HAND_NAMES):
        wrists = ep.wrists[:, hi, :]
        mask = ep.wrist_mask[:, hi]
        xyz_interp, _ = interpolate_wrist_spline(t, wrists, mask)
        uv_spline[name] = project_cam_to_pixel(xyz_interp, ep.intrinsics)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames_out = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for fi in range(ep.num_frames):
        img = Image.fromarray(ep.frames[fi])
        draw = ImageDraw.Draw(img)
        for name in HAND_NAMES:
            _draw_trail_pil(
                draw,
                uv_spline[name],
                ep.wrist_mask[:, HAND_NAMES.index(name)],
                HAND_COLORS_RGB[name],
                fi,
                trail_len,
                w,
                h,
            )
        label = f"frame {fi}/{ep.num_frames - 1}  L={int(ep.left_mask[fi])} R={int(ep.right_mask[fi])}"
        draw.rectangle((4, 4, 320, 28), fill=(0, 0, 0, 180))
        draw.text((8, 6), label, fill=(255, 255, 255), font=font)
        frames_out.append(np.asarray(img))

    imageio.mimsave(out_path, frames_out, fps=fps, codec="libx264", quality=8)
    print(f"Saved video: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize full wrist episode")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--episode", default=0, help="Episode index (int) or ann/video name substring")
    parser.add_argument("--out_dir", type=str, default="outputs/wrist_viz")
    parser.add_argument("--trail_len", type=int, default=40)
    parser.add_argument("--all", action="store_true", help="Visualize every episode")
    args = parser.parse_args()

    reader = WristEpisodeReader(data_root=args.data_root)
    print("Episodes:", reader.list_episodes())

    indices = range(len(reader.episode_pairs)) if args.all else [args.episode]
    os.makedirs(args.out_dir, exist_ok=True)

    for ep_id in indices:
        ep = reader.load(ep_id)
        stem = os.path.splitext(ep.video_name)[0]
        plot_path = os.path.join(args.out_dir, f"{stem}_wrist_mask.png")
        video_path = os.path.join(args.out_dir, f"{stem}_wrist_overlay.mp4")

        plot_wrist_and_mask(ep, plot_path)
        render_overlay_video(ep, video_path, trail_len=args.trail_len)

        print(f"\n[{ep_id}] {ep.video_name}")
        print(f"  frames: {ep.num_frames}")
        print(f"  left valid: {ep.left_mask.sum()} / {ep.num_frames}")
        print(f"  right valid: {ep.right_mask.sum()} / {ep.num_frames}")


if __name__ == "__main__":
    main()
