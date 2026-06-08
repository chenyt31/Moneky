"""Visualize LLaVA-OV2 dataset samples for full episodes (history / anchor / masked future)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

VITRA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VITRA_ROOT.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_STATS = DEFAULT_DATA_ROOT / "worldmodelv1_angle_statistics.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VITRA_ROOT) not in sys.path:
    sys.path.insert(0, str(VITRA_ROOT))

from vitra.datasets.llava_ov2_dataset import LlavaOV2HumanDataset
from vitra.visualization.llava_ov2_viz import render_episode_dataset_diagnostics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render per-anchor dataset diagnostic videos for episodes.")
    p.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--statistics_path", default=str(DEFAULT_STATS))
    p.add_argument(
        "--output_dir",
        default=str(VITRA_ROOT / "outputs/llava_ov2_dataset_viz"),
    )
    p.add_argument(
        "--episodes",
        default="",
        help="Comma-separated episode ids. Empty = pick top-N longest episodes.",
    )
    p.add_argument("--num_episodes", type=int, default=2)
    p.add_argument("--max_anchors", type=int, default=0, help="0 = all anchors (T-1 videos)")
    p.add_argument("--action_future_window_size", type=int, default=15)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument(
        "--mano_path",
        default=str(VITRA_ROOT / "weights/mano"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LlavaOV2HumanDataset(
        data_root=args.data_root,
        statistics_path=args.statistics_path,
        action_future_window_size=args.action_future_window_size,
        max_samples=0,
        augmentation=False,
        normalization=True,
    )

    if args.episodes.strip():
        episode_ids = [x.strip() for x in args.episodes.split(",") if x.strip()]
    else:
        episode_ids = [ep for ep, _ in dataset.list_episodes()[: args.num_episodes]]

    os.makedirs(args.output_dir, exist_ok=True)
    all_summaries = []
    for episode_id in episode_ids:
        print(f"=== episode {episode_id} ===")
        summary = render_episode_dataset_diagnostics(
            dataset,
            episode_id,
            args.output_dir,
            dataset.normalizer,
            mano_model_path=args.mano_path,
            fps=args.fps,
            max_anchors=args.max_anchors,
        )
        mismatch = summary["num_anchor_videos"] != summary["expected_num_videos"]
        print(
            f"  total_frames={summary['total_frames']}  "
            f"videos={summary['num_anchor_videos']}  "
            f"expected={summary['expected_num_videos']}  "
            f"count_ok={not mismatch}"
        )
        if mismatch:
            print(
                "  WARNING: anchor video count != total_frames-1; "
                "check episode_frame_index vs episode length."
            )
        all_summaries.append(summary)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
