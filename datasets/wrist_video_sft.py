"""
SFT dataloader for wrist trajectory prediction from WorldModel episodic data.

Each sample uses all history video frames and synchronized wrist positions in the
camera frame as input, and predicts the next K frames of absolute wrist positions
(3D per hand; missing hands are None / masked in batched tensors).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset

# MANO root translation (transl_worldspace) is the wrist; NOT joints_camspace[:, 0].
LEFT_IDX = 0
RIGHT_IDX = 1


Wrist3D = Optional[np.ndarray]  # shape (3,) or None
WristPair = Tuple[Wrist3D, Wrist3D]  # (left, right)


@dataclass
class WristVideoSFTConfig:
    data_root: str = "data"
    annotation_subdir: str = "episodic_annotations"
    video_subdir: str = "videos"
    future_k: int = 16
    # Cap history length (most recent frames). <= 0 means no cap (full history).
    max_history: int = 0
    min_history: int = 1
    image_size: Tuple[int, int] = (224, 224)
    num_workers: int = 4
    batch_size: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = True


def discover_episode_pairs(data_root: str) -> List[Dict[str, str]]:
    """Pair each .npy annotation with its video via annotation['video_name']."""
    ann_dir = os.path.join(data_root, "episodic_annotations")
    vid_dir = os.path.join(data_root, "videos")
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    pairs = []
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".npy"):
            continue
        ann_path = os.path.join(ann_dir, fname)
        ann = np.load(ann_path, allow_pickle=True).item()
        video_name = ann["video_name"]
        video_path = os.path.join(vid_dir, video_name)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video for {fname} not found: {video_path}")
        pairs.append(
            {
                "ann_path": ann_path,
                "video_path": video_path,
                "video_name": video_name,
            }
        )
    if not pairs:
        raise FileNotFoundError(f"No .npy annotations found under {ann_dir}")
    return pairs


def _hand_valid_at_frame(hand: dict, frame_idx: int) -> bool:
    kept = hand["kept_frames"][frame_idx]
    if isinstance(kept, (bool, np.bool_)):
        return bool(kept)
    return int(kept) == 1


def _wrist_camspace_at_frame(ann: dict, hand_key: str, frame_idx: int) -> Wrist3D:
    """
    Wrist = MANO root translation in camera frame.

    Use transl_worldspace transformed by per-frame extrinsics. joints_camspace[:, 0]
    is not the wrist in this dataset (index 0 is offset ~10cm from the true root).
    """
    hand = ann[hand_key]
    if not _hand_valid_at_frame(hand, frame_idx):
        return None
    transl = hand["transl_worldspace"][frame_idx]
    if np.allclose(transl, 0.0):
        return None
    ext = ann["extrinsics"][frame_idx]
    pos = (ext @ np.append(transl.astype(np.float64), 1.0))[:3]
    if pos[2] <= 1e-4 or np.allclose(pos, 0.0):
        return None
    return pos.astype(np.float32)


def parse_wrist_pair(ann: dict, frame_idx: int) -> WristPair:
    """Return (left_wrist, right_wrist) in camera coordinates; None if missing."""
    left = _wrist_camspace_at_frame(ann, "left", frame_idx)
    right = _wrist_camspace_at_frame(ann, "right", frame_idx)
    return left, right


def _wrist_pair_to_array(pair: WristPair) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (left, right) to float32 (2,3) and bool validity mask (2,)."""
    out = np.full((2, 3), np.nan, dtype=np.float32)
    mask = np.zeros(2, dtype=bool)
    for i, w in enumerate(pair):
        if w is not None:
            out[i] = w
            mask[i] = True
    return out, mask


def _load_video_frames(video_path: str, frame_indices: Sequence[int]) -> np.ndarray:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    if len(frame_indices) == 0:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    idx = list(frame_indices)
    frames = vr.get_batch(idx).asnumpy()  # (T, H, W, C)
    vr.seek(0)
    return frames


def _resize_frames(frames: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    if frames.shape[0] == 0:
        return frames
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("PIL is required for frame resizing") from e

    h, w = image_size
    out = []
    for f in frames:
        img = Image.fromarray(f).resize((w, h), Image.BILINEAR)
        out.append(np.asarray(img))
    return np.stack(out, axis=0)


class WristVideoSFTDataset(Dataset):
    """
    Sliding-window SFT samples over episodic video + wrist annotations.

    For history end index ``t`` (0-based, inclusive):
      - Input: video frames ``[hist_start, t]`` and wrist pairs for the same indices
      - Target: wrist pairs at frames ``t+1 .. t+future_k`` (absolute camera coords)

    Wrist layout per timestep: left (3,) and right (3,); missing hand -> None.
    Batched tensors use NaN + ``*_mask`` for missing values.
    """

    def __init__(
        self,
        data_root: str = "data",
        future_k: int = 16,
        max_history: int = 0,
        min_history: int = 1,
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        episode_pairs: Optional[List[Dict[str, str]]] = None,
    ):
        self.data_root = data_root
        self.future_k = future_k
        self.max_history = max_history
        self.min_history = min_history
        self.image_size = image_size
        self.transform = transform

        self.episode_pairs = episode_pairs or discover_episode_pairs(data_root)
        self.samples: List[Dict] = []
        self._index_samples()

    def _index_samples(self) -> None:
        self.samples.clear()
        for ep_idx, ep in enumerate(self.episode_pairs):
            ann = np.load(ep["ann_path"], allow_pickle=True).item()
            decode_frames = ann["video_decode_frame"]
            num_frames = len(decode_frames)
            # Need t+1 history frames (0..t) and t+1..t+K future wrist targets.
            for t in range(num_frames - self.future_k):
                hist_len = t + 1
                if hist_len < self.min_history:
                    continue
                self.samples.append(
                    {
                        "episode_idx": ep_idx,
                        "hist_end": t,
                        "hist_len": hist_len,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _history_start(self, hist_end: int, hist_len: int) -> int:
        if self.max_history > 0 and hist_len > self.max_history:
            return hist_end + 1 - self.max_history
        return 0

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        ep = self.episode_pairs[sample["episode_idx"]]
        ann = np.load(ep["ann_path"], allow_pickle=True).item()
        decode_frames = ann["video_decode_frame"]

        hist_end = sample["hist_end"]
        hist_start = self._history_start(hist_end, sample["hist_len"])
        hist_indices = list(range(hist_start, hist_end + 1))
        decode_hist = [decode_frames[i] for i in hist_indices]

        frames = _load_video_frames(ep["video_path"], decode_hist)
        frames = _resize_frames(frames, self.image_size)

        history_wrists = []
        history_masks = []
        for i in hist_indices:
            w, m = _wrist_pair_to_array(parse_wrist_pair(ann, i))
            history_wrists.append(w)
            history_masks.append(m)

        future_wrists = []
        future_masks = []
        for i in range(hist_end + 1, hist_end + 1 + self.future_k):
            w, m = _wrist_pair_to_array(parse_wrist_pair(ann, i))
            future_wrists.append(w)
            future_masks.append(m)

        history_wrists = np.stack(history_wrists, axis=0)  # (T, 2, 3)
        history_masks = np.stack(history_masks, axis=0)  # (T, 2)
        future_wrists = np.stack(future_wrists, axis=0)  # (K, 2, 3)
        future_masks = np.stack(future_masks, axis=0)  # (K, 2)

        if self.transform is not None:
            history_frames = self.transform(frames)
        else:
            # (T, H, W, C) uint8 -> (T, C, H, W) float32 in [0, 1]
            history_frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        return {
            "history_frames": history_frames,
            "history_wrists": torch.from_numpy(history_wrists),
            "history_wrist_mask": torch.from_numpy(history_masks),
            "future_wrists": torch.from_numpy(future_wrists),
            "future_wrist_mask": torch.from_numpy(future_masks),
            "hist_len": len(hist_indices),
            "hist_start": hist_start,
            "hist_end": hist_end,
            "video_path": ep["video_path"],
            "ann_path": ep["ann_path"],
            "video_name": ann["video_name"],
        }


@dataclass
class WristVideoSFTCollator:
    """Pad variable-length history; stack fixed-length future targets."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(instances)
        max_hist = max(inst["hist_len"] for inst in instances)
        k = instances[0]["future_wrists"].shape[0]
        _, _, h, w = instances[0]["history_frames"].shape

        history_frames = torch.zeros(batch_size, max_hist, 3, h, w, dtype=instances[0]["history_frames"].dtype)
        history_wrists = torch.full((batch_size, max_hist, 2, 3), float("nan"))
        history_wrist_mask = torch.zeros(batch_size, max_hist, 2, dtype=torch.bool)
        history_len = torch.zeros(batch_size, dtype=torch.long)

        future_wrists = torch.stack([inst["future_wrists"] for inst in instances], dim=0)
        future_wrist_mask = torch.stack([inst["future_wrist_mask"] for inst in instances], dim=0)

        for b, inst in enumerate(instances):
            t = inst["hist_len"]
            history_len[b] = t
            # Right-align so the most recent frames share the same end index in the batch.
            offset = max_hist - t
            history_frames[b, offset:] = inst["history_frames"]
            history_wrists[b, offset:] = inst["history_wrists"]
            history_wrist_mask[b, offset:] = inst["history_wrist_mask"]

        return {
            "history_frames": history_frames,
            "history_wrists": history_wrists,
            "history_wrist_mask": history_wrist_mask,
            "history_len": history_len,
            "future_wrists": future_wrists,
            "future_wrist_mask": future_wrist_mask,
            "future_k": k,
            "video_paths": [inst["video_path"] for inst in instances],
            "ann_paths": [inst["ann_path"] for inst in instances],
            "hist_ends": torch.tensor([inst["hist_end"] for inst in instances], dtype=torch.long),
        }


def make_wrist_sft_dataloader(
    config: Optional[WristVideoSFTConfig] = None,
    *,
    transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    episode_pairs: Optional[List[Dict[str, str]]] = None,
) -> DataLoader:
    cfg = config or WristVideoSFTConfig()
    dataset = WristVideoSFTDataset(
        data_root=cfg.data_root,
        future_k=cfg.future_k,
        max_history=cfg.max_history,
        min_history=cfg.min_history,
        image_size=cfg.image_size,
        transform=transform,
        episode_pairs=episode_pairs,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=WristVideoSFTCollator(),
    )
