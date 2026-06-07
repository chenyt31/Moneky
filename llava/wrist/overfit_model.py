"""Small models that can memorize wrist trajectories on tiny datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from llava.wrist.metrics import compute_wrist_metrics, masked_wrist_loss
from llava.wrist.normalize import WristNormStats, denormalize_wrist_tensor, normalize_wrist_tensor


@dataclass
class WristOverfitConfig:
    future_k: int = 16
    max_history: int = 65
    hidden: int = 2048
    depth: int = 4
    use_episode_embed: bool = True
    n_episodes: int = 8
    video_ctx_dim: int = 0  # >0 when using cached LLaVA features


class WristOverfitMLP(nn.Module):
    """
    Flatten padded history wrists (+ mask + optional episode id + video ctx) -> future K.

    Designed to overfit ~100–200 sliding-window samples.
    """

    def __init__(
        self,
        config: Optional[WristOverfitConfig] = None,
        norm_stats: Optional[WristNormStats] = None,
    ):
        super().__init__()
        self.config = config or WristOverfitConfig()
        self.norm_stats = norm_stats
        self._norm_mean: Optional[torch.Tensor] = None
        self._norm_std: Optional[torch.Tensor] = None
        c = self.config
        t, k = c.max_history, c.future_k

        in_dim = t * 6 + t * 2  # wrists + per-step hand-valid mask (2,)
        if c.use_episode_embed:
            in_dim += c.n_episodes
        in_dim += 1  # hist_end / T_norm
        in_dim += c.video_ctx_dim

        layers = []
        dim = in_dim
        for _ in range(c.depth):
            layers += [nn.Linear(dim, c.hidden), nn.GELU()]
            dim = c.hidden
        layers.append(nn.Linear(dim, k * 2 * 3))
        self.net = nn.Sequential(*layers)

    def _pack_input(
        self,
        history_wrists: torch.Tensor,
        history_wrist_mask: torch.Tensor,
        history_len: torch.Tensor,
        episode_idx: Optional[torch.Tensor] = None,
        hist_ends: Optional[torch.Tensor] = None,
        video_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t_max = history_wrists.shape[:2]
        device = history_wrists.device
        chunks = []

        flat_w = torch.zeros(b, t_max * 6, device=device)
        flat_m = torch.zeros(b, t_max * 2, device=device)
        for bi in range(b):
            t_valid = int(history_len[bi].item())
            off = t_max - t_valid
            w = history_wrists[bi, off:].nan_to_num(0.0).reshape(t_valid, 6)
            m = history_wrist_mask[bi, off:].float().reshape(t_valid, 2)
            flat_w[bi, off * 6 : (off + t_valid) * 6] = w.reshape(-1)
            flat_m[bi, off * 2 : (off + t_valid) * 2] = m.reshape(-1)
        chunks.extend([flat_w, flat_m])

        if self.config.use_episode_embed and episode_idx is not None:
            ep_oh = torch.zeros(b, self.config.n_episodes, device=device)
            ep_oh.scatter_(1, episode_idx.long().clamp(0, self.config.n_episodes - 1).unsqueeze(1), 1.0)
            chunks.append(ep_oh)

        if hist_ends is not None:
            norm = hist_ends.float().unsqueeze(1) / max(self.config.max_history, 1)
            chunks.append(norm)

        if video_ctx is not None:
            chunks.append(video_ctx.float())

        return torch.cat(chunks, dim=-1)

    def _norm_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._norm_mean is None or self._norm_mean.device != device:
            if self.norm_stats is None:
                raise ValueError("norm_stats required for normalized forward")
            self._norm_mean, self._norm_std = self.norm_stats.to_tensors(device)
        return self._norm_mean, self._norm_std

    def forward(
        self,
        history_wrists: torch.Tensor,
        history_wrist_mask: torch.Tensor,
        history_len: torch.Tensor,
        episode_idx: Optional[torch.Tensor] = None,
        hist_ends: Optional[torch.Tensor] = None,
        video_ctx: Optional[torch.Tensor] = None,
        future_wrists: Optional[torch.Tensor] = None,
        future_wrist_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        hist_in = history_wrists
        fut_tgt = future_wrists
        if self.norm_stats is not None:
            mean, std = self._norm_tensors(history_wrists.device)
            hist_in = normalize_wrist_tensor(history_wrists, history_wrist_mask, mean, std)
            if future_wrists is not None:
                fut_tgt = normalize_wrist_tensor(future_wrists, future_wrist_mask, mean, std)

        x = self._pack_input(hist_in, history_wrist_mask, history_len, episode_idx, hist_ends, video_ctx)
        pred_norm = self.net(x).view(-1, self.config.future_k, 2, 3)

        if self.norm_stats is not None:
            mean, std = self._norm_tensors(pred_norm.device)
            pred = denormalize_wrist_tensor(pred_norm, mean, std)
        else:
            pred = pred_norm

        out = {"pred": pred, "pred_norm": pred_norm}
        if fut_tgt is not None and future_wrist_mask is not None:
            out["loss"] = masked_wrist_loss(pred_norm, fut_tgt, future_wrist_mask)
            if future_wrists is not None:
                out["metrics"] = compute_wrist_metrics(pred, future_wrists, future_wrist_mask)
        return out
