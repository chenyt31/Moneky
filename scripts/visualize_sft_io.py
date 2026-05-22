#!/usr/bin/env python3
"""
Visualize one WristVideoSFT training sample: model INPUT vs TARGET.

Run:
  python3 scripts/visualize_sft_io.py --hist_end 40
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.epoch_reader import WristEpisodeReader
from datasets.wrist_video_sft import WristVideoSFTDataset, discover_episode_pairs, parse_wrist_pair

HAND_COLORS = {"left": "#00b4ff", "right": "#ff5020"}
HAND_LABELS = ("left", "right")


def project_uv(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    out = np.full((xyz.shape[0], 2), np.nan)
    ok = np.isfinite(xyz).all(1) & (xyz[:, 2] > 1e-4)
    if not ok.any():
        return out
    p = xyz[ok]
    out[ok, 0] = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
    out[ok, 1] = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    return out


def find_sample_index(dataset: WristVideoSFTDataset, episode: int, hist_end: int) -> int:
    for i, s in enumerate(dataset.samples):
        if s["episode_idx"] == episode and s["hist_end"] == hist_end:
            return i
    raise ValueError(f"No sample with episode={episode} hist_end={hist_end}")


def load_raw_frames(ep, hist_indices):
    from datasets.wrist_video_sft import _load_video_frames

    decode = [ep.ann_raw["video_decode_frame"][i] for i in hist_indices]
    return _load_video_frames(ep.video_path, decode)


def draw_io_diagram(
    sample: dict,
    ep,
    hist_indices: list,
    out_path: str,
    future_k: int,
    max_frames_show: int = 8,
) -> None:
    K = ep.intrinsics
    hist_end = sample["hist_end"]
    t_hist = sample["hist_len"]

    # subsample history frames for display
    if len(hist_indices) <= max_frames_show:
        show_idx = hist_indices
    else:
        pick = np.linspace(0, len(hist_indices) - 1, max_frames_show, dtype=int)
        show_idx = [hist_indices[i] for i in pick]

    raw_frames = load_raw_frames(ep, show_idx)

    hf = sample["history_frames"].numpy()  # T,3,h,w
    hw = sample["history_wrists"].numpy()
    hm = sample["history_wrist_mask"].numpy()
    fw = sample["future_wrists"].numpy()
    fm = sample["future_wrist_mask"].numpy()

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 1, height_ratios=[1.0, 2.2, 1.6], hspace=0.28)
    ax_tl = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1])
    ax_z = fig.add_subplot(gs[2])

    # --- Timeline schematic ---
    ax_tl.set_xlim(-1, t_hist + future_k + 2)
    ax_tl.set_ylim(0, 4)
    ax_tl.axis("off")
    ax_tl.set_title(
        f"SFT sample @ hist_end t={hist_end}  |  episode: {sample['video_name']}\n"
        f"INPUT: {t_hist} video frames + synced wrists  →  TARGET: next {future_k} wrist positions (camera frame)",
        fontsize=12,
        fontweight="bold",
    )

    # input bar
    ax_tl.add_patch(mpatches.FancyBboxPatch((0, 2.2), t_hist, 0.9, boxstyle="round,pad=0.02", fc="#d4edda", ec="#28a745", lw=2))
    ax_tl.text(t_hist / 2, 2.65, f"INPUT: frames 0…{hist_end}  +  wrists (T={t_hist})", ha="center", va="center", fontsize=11, fontweight="bold")
    ax_tl.text(t_hist / 2, 1.5, "video + wrist (synced)", ha="center", fontsize=9, color="#555")

    # target bar
    ax_tl.add_patch(
        mpatches.FancyBboxPatch((t_hist + 0.5, 2.2), future_k, 0.9, boxstyle="round,pad=0.02", fc="#f8d7da", ec="#dc3545", lw=2)
    )
    ax_tl.text(t_hist + 0.5 + future_k / 2, 2.65, f"TARGET: wrists {hist_end+1}…{hist_end+future_k}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax_tl.text(t_hist + 0.5 + future_k / 2, 1.5, "no future video", ha="center", fontsize=9, color="#555")

    ax_tl.annotate("", xy=(t_hist + 0.3, 2.65), xytext=(t_hist + 0.5, 2.65), arrowprops=dict(arrowstyle="->", lw=2))
    ax_tl.text(t_hist + 0.4, 3.15, "predict", ha="center", fontsize=10)

    # frame ticks
    for i in range(0, min(t_hist, 12)):
        ax_tl.plot(i, 1.0, "o", color="#28a745", ms=4)
    if t_hist > 12:
        ax_tl.text(6, 0.6, "…", ha="center", fontsize=14)
    for i in range(future_k):
        ax_tl.plot(t_hist + 1 + i, 1.0, "s", color="#dc3545", ms=4)

    # shape cheat sheet
    shapes = (
        f"history_frames   {tuple(hf.shape)}\n"
        f"history_wrists   {tuple(hw.shape)}  + mask {tuple(hm.shape)}\n"
        f"future_wrists    {tuple(fw.shape)}  + mask {tuple(fm.shape)}"
    )
    ax_tl.text(0.02, 0.15, shapes, transform=ax_tl.transAxes, fontsize=9, family="monospace", va="bottom",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # --- History frame strip (INPUT) ---
    ax_strip.set_title("INPUT — history video frames (wrists overlaid)", fontsize=11)
    n = len(show_idx)
    ax_strip.set_xlim(0, n)
    ax_strip.set_ylim(0, 1)
    ax_strip.axis("off")

    for col, (fi, frame) in enumerate(zip(show_idx, raw_frames)):
        ax = ax_strip.inset_axes([col / n + 0.01, 0.05, 1 / n - 0.02, 0.9])
        ax.imshow(frame)
        ax.axis("off")
        # wrists at this annotation index
        pair = parse_wrist_pair(ep.ann_raw, fi)
        uv_list = []
        for hi, name in enumerate(HAND_LABELS):
            w = pair[hi]
            if w is not None:
                u, v = project_uv(w[None, :], K)[0]
                if np.isfinite(u):
                    ax.plot(u, v, "o", color=HAND_COLORS[name], ms=6, markeredgecolor="white", markeredgewidth=0.8)
        ax.set_title(f"t={fi}", fontsize=8)

    # --- Future target plot (TARGET only wrists) ---
    fut_t = np.arange(hist_end + 1, hist_end + 1 + future_k)
    ax_z.set_title("TARGET — future wrist trajectory (camera XYZ; no images)", fontsize=11)
    for hi, name in enumerate(HAND_LABELS):
        y = fw[:, hi, 2]  # z
        m = fm[:, hi]
        ax_z.plot(fut_t, y, color=HAND_COLORS[name], lw=2, label=f"{name} z")
        ax_z.scatter(fut_t[m], y[m], c=HAND_COLORS[name], s=40, zorder=3, edgecolors="white", linewidths=0.5)
    ax_z.axvline(hist_end, color="gray", ls="--", lw=1, label="now (t)")
    ax_z.set_xlabel("frame index")
    ax_z.set_ylabel("z in camera frame (m)")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc="upper right", fontsize=9)

    # inset: xy projection of future
    ax_xy = ax_z.inset_axes([0.02, 0.55, 0.35, 0.42])
    for hi, name in enumerate(HAND_LABELS):
        xy = fw[:, hi, :2]
        m = fm[:, hi]
        ax_xy.plot(xy[m, 0], xy[m, 1], "-o", color=HAND_COLORS[name], ms=3, label=name)
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title("future x-y", fontsize=8)
    ax_xy.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def draw_compare_panel(sample: dict, out_path: str) -> None:
    """Side-by-side: what goes in vs what model must predict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    hf = sample["history_frames"].numpy()
    mid = hf[len(hf) // 2].transpose(1, 2, 0)  # HWC

    axes[0].imshow(np.clip(mid, 0, 1))
    axes[0].set_title(
        f"INPUT (example frame)\n"
        f"history_frames [{hf.shape[0]}, 3, {hf.shape[2]}, {hf.shape[3]}]\n"
        f"+ history_wrists [{sample['history_wrists'].shape}]",
        fontsize=10,
    )
    axes[0].axis("off")

    fw = sample["future_wrists"].numpy()
    fm = sample["future_wrist_mask"].numpy()
    axes[1].axis("off")
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].set_title(
        f"TARGET (no pixels)\n"
        f"future_wrists [{fw.shape}] — next K absolute 3D positions",
        fontsize=10,
    )
    text = "Future wrist (camera frame):\n\n"
    for k in range(min(5, fw.shape[0])):
        L = "L✓" if fm[k, 0] else "L—"
        R = "R✓" if fm[k, 1] else "R—"
        text += f"  t+{k+1}: {L}  {R}\n"
        if fm[k, 0]:
            text += f"       L xyz = {fw[k,0]}\n"
        if fm[k, 1]:
            text += f"       R xyz = {fw[k,1]}\n"
    if fw.shape[0] > 5:
        text += f"  … ({fw.shape[0]-5} more frames)\n"
    axes[1].text(0.05, 0.95, text, transform=axes[1].transAxes, va="top", fontsize=9, family="monospace",
                 bbox=dict(facecolor="#fff3cd", edgecolor="#ffc107"))

    # simple 3D path sketch
    ax2 = axes[1].inset_axes([0.55, 0.1, 0.42, 0.75])
    for hi, name in enumerate(HAND_LABELS):
        m = fm[:, hi]
        if m.any():
            ax2.plot(fw[m, hi, 0], fw[m, hi, 2], "-o", color=HAND_COLORS[name], ms=4, label=name)
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("target path (x-z)", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--hist_end", type=int, default=40, help="History end index t for the sample")
    parser.add_argument("--future_k", type=int, default=16)
    parser.add_argument("--out_dir", default="outputs/sft_io_viz")
    args = parser.parse_args()

    pairs = discover_episode_pairs(args.data_root)
    ds = WristVideoSFTDataset(data_root=args.data_root, future_k=args.future_k, image_size=(224, 224), episode_pairs=pairs)
    idx = find_sample_index(ds, args.episode, args.hist_end)
    sample = ds[idx]

    reader = WristEpisodeReader(args.data_root)
    ep = reader.load(args.episode)
    hist_start = sample["hist_start"]
    hist_indices = list(range(hist_start, sample["hist_end"] + 1))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(sample["video_name"])[0]

    draw_io_diagram(
        sample,
        ep,
        hist_indices,
        os.path.join(args.out_dir, f"{stem}_t{args.hist_end}_io_diagram.png"),
        args.future_k,
    )
    draw_compare_panel(sample, os.path.join(args.out_dir, f"{stem}_t{args.hist_end}_in_vs_out.png"))

    print("\n=== Sample summary ===")
    print(f"index={idx}  hist_end t={args.hist_end}  hist_len={sample['hist_len']}")
    print(f"INPUT  history_frames {tuple(sample['history_frames'].shape)}")
    print(f"       history_wrists {tuple(sample['history_wrists'].shape)}  valid={sample['history_wrist_mask'].sum().item()} hand-steps")
    print(f"TARGET future_wrists {tuple(sample['future_wrists'].shape)}  valid={sample['future_wrist_mask'].sum().item()} hand-steps")


if __name__ == "__main__":
    main()
