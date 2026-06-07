"""Wrist trajectory head on top of LLaVA-OneVision (HF) with official video encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import LlavaOnevisionForConditionalGeneration

from llava.wrist.metrics import compute_wrist_metrics, masked_wrist_loss
from llava.wrist.normalize import WristNormStats, denormalize_wrist_tensor, normalize_wrist_tensor


def _resolve_llava_dtype(torch_dtype: str) -> torch.dtype:
    if torch_dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if torch_dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


@dataclass
class WristLlavaOVConfig:
    model_name_or_path: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    future_k: int = 16
    freeze_llava: bool = True
    use_lm_hidden: bool = True
    dropout: float = 0.1
    torch_dtype: str = "auto"  # auto -> bfloat16 on CUDA if supported, else float16


class WristLlavaOneVisionModel(nn.Module):
    """
    Uses LlavaOnevisionForConditionalGeneration video pathway:
      pixel_values_videos -> get_video_features / full forward with video tokens
    Then fuses pooled LM hidden state with history wrist features for regression.
    """

    def __init__(
        self,
        config: Optional[WristLlavaOVConfig] = None,
        norm_stats: Optional[WristNormStats] = None,
    ):
        super().__init__()
        self.config = config or WristLlavaOVConfig()
        self.norm_stats = norm_stats
        self._norm_mean: Optional[torch.Tensor] = None
        self._norm_std: Optional[torch.Tensor] = None
        dtype = _resolve_llava_dtype(self.config.torch_dtype)

        self.llava = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=dtype,
        )
        hidden = self.llava.config.text_config.hidden_size

        self.wrist_encoder = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.video_ctx_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, self.config.future_k * 2 * 3),
        )

        if self.config.freeze_llava:
            self.llava.requires_grad_(False)
            self.llava.eval()
        else:
            self.llava.requires_grad_(True)
            self.llava.train()

        # Wrist head in fp32 for stable regression; LLaVA stays in fp16/bf16.
        self.wrist_encoder.to(dtype=torch.float32)
        self.head.to(dtype=torch.float32)
        self.video_ctx_norm.to(dtype=torch.float32)

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.llava, "gradient_checkpointing_enable"):
            self.llava.gradient_checkpointing_enable()
        elif hasattr(self.llava, "model") and hasattr(self.llava.model, "gradient_checkpointing_enable"):
            self.llava.model.gradient_checkpointing_enable()

    def trainable_parameter_groups(
        self,
        *,
        lr: float,
        llava_lr: float,
        weight_decay: float,
    ) -> list[dict]:
        """Separate LR for LLaVA backbone vs wrist head."""
        llava_params, head_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("llava."):
                llava_params.append(p)
            else:
                head_params.append(p)
        groups = []
        if head_params:
            groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})
        if llava_params:
            groups.append({"params": llava_params, "lr": llava_lr, "weight_decay": weight_decay})
        return groups

    def n_trainable_params(self) -> tuple[int, int]:
        llava_n = sum(p.numel() for n, p in self.named_parameters() if n.startswith("llava.") and p.requires_grad)
        head_n = sum(p.numel() for n, p in self.named_parameters() if not n.startswith("llava.") and p.requires_grad)
        return llava_n, head_n

    @property
    def llava_dtype(self):
        return next(self.llava.parameters()).dtype

    def _norm_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._norm_mean is None or self._norm_mean.device != device:
            if self.norm_stats is None:
                raise ValueError("norm_stats required for normalized forward")
            self._norm_mean, self._norm_std = self.norm_stats.to_tensors(device)
        return self._norm_mean, self._norm_std

    def _pool_history_wrists(
        self,
        history_wrists: torch.Tensor,
        history_wrist_mask: torch.Tensor,
        history_len: torch.Tensor,
    ) -> torch.Tensor:
        b, t_max = history_wrists.shape[:2]
        out = []
        for bi in range(b):
            t_valid = int(history_len[bi].item())
            offset = t_max - t_valid
            wrists = history_wrists[bi, offset:].nan_to_num(0.0).float()
            mask = history_wrist_mask[bi, offset:].float()
            flat = wrists.reshape(t_valid, 6)
            emb = self.wrist_encoder(flat)
            step_w = mask.amax(dim=-1).clamp(min=1e-6).unsqueeze(-1)
            pooled = (emb * step_w).sum(dim=0) / step_w.sum().clamp(min=1e-6)
            out.append(pooled)
        return torch.stack(out, dim=0)

    def _encode_video_context(
        self,
        pixel_values_videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """LLaVA-OneVision video modality: embed video tokens and run the language model."""
        pixel_values_videos = pixel_values_videos.to(device=self.llava.device, dtype=self.llava_dtype)
        input_ids = input_ids.to(self.llava.device)
        attention_mask = attention_mask.to(self.llava.device)

        if self.config.freeze_llava:
            with torch.no_grad():
                outputs = self.llava(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values_videos=pixel_values_videos,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            outputs = self.llava(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden = outputs.hidden_states[-1].float()
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            ctx = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            ctx = hidden.mean(dim=1)
        ctx = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
        return ctx

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        history_wrists: torch.Tensor,
        history_wrist_mask: torch.Tensor,
        history_len: torch.Tensor,
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

        video_ctx = self.video_ctx_norm(self._encode_video_context(pixel_values_videos, input_ids, attention_mask))
        wrist_ctx = self._pool_history_wrists(hist_in, history_wrist_mask, history_len)

        fused = torch.cat([self.dropout(video_ctx), self.dropout(wrist_ctx)], dim=-1)
        pred_norm = self.head(fused).view(-1, self.config.future_k, 2, 3)

        if self.norm_stats is not None:
            mean, std = self._norm_tensors(pred_norm.device)
            pred = denormalize_wrist_tensor(pred_norm, mean, std)
        else:
            pred = pred_norm

        out = {"pred": pred, "pred_norm": pred_norm, "video_ctx": video_ctx}
        if fut_tgt is not None and future_wrist_mask is not None:
            out["loss"] = masked_wrist_loss(pred_norm, fut_tgt, future_wrist_mask)
            if future_wrists is not None:
                out["metrics"] = compute_wrist_metrics(pred, future_wrists, future_wrist_mask)
        return out
