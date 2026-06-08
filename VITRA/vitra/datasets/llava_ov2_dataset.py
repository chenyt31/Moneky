"""Debug/overfit dataset: VITRA hand actions + history video [0..anchor] for LLaVA codec."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from vitra.datasets.human_dataset import EpisodicDatasetCore
from vitra.utils.data_utils import read_dataset_statistics, GaussianNormalizer


def resolve_llava_ov2_data_paths(data_root: str) -> Tuple[str, str, str]:
    """Resolve annotation index, label folder, and video root under ``data_root``."""
    data_root = os.path.abspath(data_root)
    flat_index = os.path.join(data_root, "episode_frame_index.npz")
    if os.path.isfile(flat_index):
        return (
            flat_index,
            os.path.join(data_root, "episodic_annotations"),
            os.path.join(data_root, "videos"),
        )
    return (
        os.path.join(data_root, "Annotation/WorldModelV1/episode_frame_index.npz"),
        os.path.join(data_root, "Annotation/WorldModelV1/episodic_annotations"),
        os.path.join(data_root, "Video/WorldModelV1_root"),
    )


def default_llava_ov2_statistics_path(data_root: str) -> str:
    data_root = os.path.abspath(data_root)
    preferred = os.path.join(data_root, "worldmodelv1_angle_statistics.json")
    if os.path.isfile(preferred):
        return preferred
    legacy = os.path.join(data_root, "WorldModelV1_mix_50k_angle_weighted_statistics.json")
    if os.path.isfile(legacy):
        return legacy
    return preferred


class LlavaOV2HumanDataset(Dataset):
    """WorldModelV1 episodic dataset: full episode mp4 + hand actions."""

    def __init__(
        self,
        data_root: str,
        statistics_path: str,
        *,
        action_future_window_size: int = 15,
        max_samples: int = 0,
        training_index_path: Optional[str] = None,
        augmentation: bool = False,
        normalization: bool = True,
    ):
        self.data_root = os.path.abspath(data_root)
        annotation_file, label_folder, video_root = resolve_llava_ov2_data_paths(self.data_root)
        if statistics_path is None:
            statistics_path = default_llava_ov2_statistics_path(self.data_root)
        self.statistics_path = statistics_path

        self.core = EpisodicDatasetCore(
            video_root=video_root,
            annotation_file=annotation_file,
            label_folder=label_folder,
            training_path=training_index_path,
            statistics_path=statistics_path,
            augmentation=augmentation,
            flip_augmentation=False,
            set_none_ratio=0.0,
            action_type="angle",
            use_rel=False,
            clip_len=None,
            action_past_window_size=0,
            action_future_window_size=action_future_window_size,
            image_past_window_size=0,
            image_future_window_size=0,
            rel_mode="step",
            load_images=False,
        )
        stats = read_dataset_statistics(statistics_path)
        self.core.set_global_data_statistics(stats)
        self.normalizer = GaussianNormalizer(stats)
        self.normalization = normalization
        self.future_k = action_future_window_size + 1

        n = len(self.core)
        self._indices = list(range(n))
        if max_samples > 0:
            self._indices = self._indices[:max_samples]

        self._episode_to_frame_ids: Dict[str, List[int]] = {}
        self._episode_frame_to_data_id: Dict[Tuple[str, int], int] = {}
        for data_id in range(len(self.core)):
            corr = self.core.index_frame_pair[data_id]
            episode_id = self.core.index_to_episode_id[corr[0]]
            frame_id = int(corr[1])
            self._episode_to_frame_ids.setdefault(episode_id, []).append(frame_id)
            self._episode_frame_to_data_id[(episode_id, frame_id)] = data_id
        for episode_id in self._episode_to_frame_ids:
            self._episode_to_frame_ids[episode_id].sort()

    def __len__(self):
        return len(self._indices)

    def list_episodes(self) -> List[Tuple[str, int]]:
        """Return (episode_id, num_indexed_frames) sorted by frame count descending."""
        return sorted(
            ((ep, len(frames)) for ep, frames in self._episode_to_frame_ids.items()),
            key=lambda x: (-x[1], x[0]),
        )

    def get_episode_frame_ids(self, episode_id: str) -> List[int]:
        frames = self._episode_to_frame_ids.get(episode_id)
        if frames is None:
            raise KeyError(f"Unknown episode_id: {episode_id}")
        return frames.copy()

    def get_episode_anchor_frames(self, episode_id: str) -> List[int]:
        """Anchor frames used for dataset samples: all indexed frames except the last."""
        frames = self.get_episode_frame_ids(episode_id)
        if len(frames) <= 1:
            return []
        return frames[:-1]

    def get_episode_length(self, episode_id: str) -> int:
        epi, _, _ = self.core._load_or_cache_episode(episode_id)
        return len(epi["extrinsics"])

    def _data_id_for_episode_frame(self, episode_id: str, frame_id: int) -> int:
        key = (episode_id, int(frame_id))
        if key not in self._episode_frame_to_data_id:
            raise KeyError(f"No dataset sample for episode={episode_id} frame={frame_id}")
        return self._episode_frame_to_data_id[key]

    def dataset_index_for_episode_frame(self, episode_id: str, frame_id: int) -> int:
        data_id = self._data_id_for_episode_frame(episode_id, frame_id)
        try:
            return self._indices.index(data_id)
        except ValueError as exc:
            raise KeyError(
                f"Sample episode={episode_id} frame={frame_id} not in active dataset subset"
            ) from exc

    def _episode_video_path(self, episode_id: str, epi: dict) -> str:
        dataset_name = episode_id.split("_")[0]
        return self.core._resolve_video_path(dataset_name, epi["video_name"])

    def get_norm_viz_sample(self, idx: int) -> dict:
        """Normalized tensors for viz (same as model training/eval pipeline)."""
        item = self[idx]
        return self._pack_norm_viz_sample(item)

    def get_norm_viz_sample_by_frame(self, episode_id: str, frame_id: int) -> dict:
        """Normalized tensors for a specific episode anchor frame."""
        idx = self.dataset_index_for_episode_frame(episode_id, frame_id)
        return self.get_norm_viz_sample(idx)

    def _pack_norm_viz_sample(self, item: dict) -> dict:
        actions = item["actions"].numpy()
        action_masks = item["action_masks"].numpy()
        return {
            "video_path": item["video_path"],
            "norm_state": item["current_state"].numpy(),
            "norm_actions": actions,
            "action_masks": action_masks,
            "hand_state_mask": item["hand_state_mask"].numpy(),
            "intrinsics": item["intrinsics"].numpy(),
            "beta_left": item["beta_left"].numpy(),
            "beta_right": item["beta_right"].numpy(),
            "episode_id": item["episode_id"],
            "frame_id": int(item["frame_id"]),
            "sample_idx": int(item["sample_idx"]),
            "future_k": int(actions.shape[0]),
            "valid_future_steps": int(action_masks[:, :102].any(axis=1).sum()),
        }

    def get_raw_viz_sample(self, idx: int) -> dict:
        """Unnormalized state/actions for visualization (before padding/normalize)."""
        data_id = self._indices[idx]
        corr = self.core.index_frame_pair[data_id]
        episode_id = self.core.index_to_episode_id[corr[0]]
        frame_id = int(corr[1])
        sample = self.core.get_item_frame(
            episode_id,
            frame_id,
            action_past_window_size=0,
            action_future_window_size=self.core.action_future_window_size,
            image_past_window_size=0,
            image_future_window_size=0,
            rel_mode="step",
            load_images=False,
        )
        epi, _, _ = self.core._load_or_cache_episode(episode_id)
        return {
            "video_path": self._episode_video_path(episode_id, epi),
            "raw_state": np.asarray(sample["current_state"], dtype=np.float32),
            "raw_actions": np.asarray(sample["action_list"], dtype=np.float32),
            "raw_action_mask": np.asarray(sample["action_mask"], dtype=bool),
            "hand_state_mask": np.asarray(sample["current_state_mask"], dtype=bool),
            "intrinsics": np.asarray(sample["intrinsics"], dtype=np.float32),
            "beta_left": np.asarray(epi["left"]["beta"], dtype=np.float32),
            "beta_right": np.asarray(epi["right"]["beta"], dtype=np.float32),
            "episode_id": episode_id,
            "frame_id": frame_id,
            "sample_idx": data_id,
        }

    def __getitem__(self, idx: int) -> dict:
        data_id = self._indices[idx]
        corr = self.core.index_frame_pair[data_id]
        episode_id = self.core.index_to_episode_id[corr[0]]
        frame_id = int(corr[1])

        sample = self.core.get_item_frame(
            episode_id,
            frame_id,
            action_past_window_size=0,
            action_future_window_size=self.core.action_future_window_size,
            image_past_window_size=0,
            image_future_window_size=0,
            rel_mode="step",
            load_images=False,
        )
        epi, _, _ = self.core._load_or_cache_episode(episode_id)
        hand_state_mask = np.asarray(sample["current_state_mask"], dtype=bool)
        betas_left = np.asarray(epi["left"]["beta"], dtype=np.float32)
        betas_right = np.asarray(epi["right"]["beta"], dtype=np.float32)

        if self.normalization:
            sample = self.core.transform_trajectory(sample, normalization=True)

        def _as_tensor(x, dtype):
            if torch.is_tensor(x):
                return x.detach().clone().to(dtype=dtype)
            return torch.tensor(x, dtype=dtype)

        video_path = self._episode_video_path(episode_id, epi)

        return {
            "video_path": video_path,
            "instruction": sample["instruction"],
            "actions": _as_tensor(sample["action_list"], torch.float32),
            "action_masks": _as_tensor(sample["action_mask"], torch.bool),
            "current_state": _as_tensor(sample["current_state"], torch.float32),
            "current_state_mask": _as_tensor(sample["current_state_mask"], torch.bool),
            "fov": torch.tensor(sample["fov"], dtype=torch.float32),
            "intrinsics": torch.tensor(sample["intrinsics"], dtype=torch.float32),
            "episode_id": episode_id,
            "frame_id": frame_id,
            "sample_idx": data_id,
            "hand_state_mask": torch.tensor(hand_state_mask, dtype=torch.bool),
            "beta_left": torch.tensor(betas_left, dtype=torch.float32),
            "beta_right": torch.tensor(betas_right, dtype=torch.float32),
        }
