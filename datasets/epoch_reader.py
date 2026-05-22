"""Load a full episodic epoch (video + wrist annotations) for inspection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from decord import VideoReader, cpu

from datasets.wrist_video_sft import discover_episode_pairs, parse_wrist_pair


@dataclass
class WristEpisode:
    """One full episode aligned by annotation frame index."""

    ann_path: str
    video_path: str
    video_name: str
    num_frames: int
    decode_frame_ids: np.ndarray  # (T,) indices into video file
    frames: np.ndarray  # (T, H, W, 3) uint8 RGB
    wrists: np.ndarray  # (T, 2, 3) float32, NaN if missing
    wrist_mask: np.ndarray  # (T, 2) bool
    intrinsics: np.ndarray  # (3, 3)
    extrinsics: np.ndarray  # (T, 4, 4)
    anno_type: str
    ann_raw: dict

    @property
    def left_wrists(self) -> np.ndarray:
        return self.wrists[:, 0, :]

    @property
    def right_wrists(self) -> np.ndarray:
        return self.wrists[:, 1, :]

    @property
    def left_mask(self) -> np.ndarray:
        return self.wrist_mask[:, 0]

    @property
    def right_mask(self) -> np.ndarray:
        return self.wrist_mask[:, 1]


def _pair_to_arrays(pair) -> Tuple[np.ndarray, np.ndarray]:
    out = np.full((2, 3), np.nan, dtype=np.float32)
    mask = np.zeros(2, dtype=bool)
    for i, w in enumerate(pair):
        if w is not None:
            out[i] = w
            mask[i] = True
    return out, mask


class WristEpisodeReader:
    def __init__(self, data_root: str = "data"):
        self.data_root = data_root
        self.episode_pairs = discover_episode_pairs(data_root)

    def list_episodes(self) -> List[str]:
        return [os.path.basename(p["ann_path"]) for p in self.episode_pairs]

    def load(self, episode: int | str = 0) -> WristEpisode:
        if isinstance(episode, str):
            ep = next(p for p in self.episode_pairs if episode in p["ann_path"] or episode in p["video_name"])
        else:
            ep = self.episode_pairs[int(episode)]

        ann = np.load(ep["ann_path"], allow_pickle=True).item()
        decode_ids = np.asarray(ann["video_decode_frame"], dtype=np.int64)
        num_frames = len(decode_ids)

        vr = VideoReader(ep["video_path"], ctx=cpu(0), num_threads=1)
        frames = vr.get_batch(decode_ids.tolist()).asnumpy()
        vr.seek(0)

        wrists = np.full((num_frames, 2, 3), np.nan, dtype=np.float32)
        mask = np.zeros((num_frames, 2), dtype=bool)
        for t in range(num_frames):
            w, m = _pair_to_arrays(parse_wrist_pair(ann, t))
            wrists[t] = w
            mask[t] = m

        return WristEpisode(
            ann_path=ep["ann_path"],
            video_path=ep["video_path"],
            video_name=ann["video_name"],
            num_frames=num_frames,
            decode_frame_ids=decode_ids,
            frames=frames,
            wrists=wrists,
            wrist_mask=mask,
            intrinsics=np.asarray(ann["intrinsics"], dtype=np.float32),
            extrinsics=np.asarray(ann["extrinsics"], dtype=np.float32),
            anno_type=str(ann.get("anno_type", "")),
            ann_raw=ann,
        )
