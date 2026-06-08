"""VITRA VLA with LLaVA-OneVision-2 history codec video backbone and DiT action head."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from vitra.models.vlm_builder import build_vlm
from vitra.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


class VITRA_LlavaOV2(nn.Module):
    def __init__(
        self,
        configs,
        train_setup_configs=None,
        act_model_configs=None,
        fwd_pred_next_n=1,
        repeated_diffusion_steps: int = 8,
        use_state="DiT",
        use_fov=True,
        use_bf16=False,
        **kwargs,
    ):
        super().__init__()

        self.configs = configs
        self.train_setup_configs = train_setup_configs or {}
        self.act_model_configs = act_model_configs or {}
        self.use_state = use_state
        self.use_fov = use_fov
        self.repeated_diffusion_steps = repeated_diffusion_steps
        self.past_action_window_size = 0
        self.chunk_size = self.configs.get("fwd_pred_next_n", 16)
        self.future_action_window_size = self.chunk_size - 1
        self.state_mask_prob = self.configs.get("state_mask_prob", 0.1)
        self.action_type = self.configs["train_dataset"].get("action_type", "angle")
        self.use_bf16 = use_bf16

        self.processor, self.backbone = self._init_backbone()
        self.tokenizer = self.processor.tokenizer
        self.act_model = self._init_act_model()

        if self.use_state == "VLM":
            self.state_and_mask_dim = 2 * self.configs["state_encoder"]["state_dim"]
            self.vlm_state_encoder = self._init_state_encoder()

        if self.use_fov:
            self.fov_encoder = self._init_fov_encoder()

        self.cognition_token_id = self.configs.get("cognition_token_id", self.tokenizer.pad_token_id or 0)
        untied_cognition_token = self.configs.get("untied_cognition_token", True)
        if untied_cognition_token:
            init_id = self.configs.get("cognition_token_init_id", self.cognition_token_id)
            ebd = self.model.get_input_embeddings().weight.data[init_id].clone()
            self.cognition_token = nn.Parameter(ebd)
        else:
            self.cognition_token = None

        if self.act_model_configs.get("token_size", -1) <= 0:
            self.act_model_configs["token_size"] = self.hidden_size

    def _init_backbone(self):
        return build_vlm(self.configs["vlm"])

    def _init_fov_encoder(self):
        from vitra.utils.nn_utils import MLPProjector

        mlp = MLPProjector(2, self.hidden_size)
        for layer in (0, 2):
            nn.init.normal_(mlp.projector[layer].weight, mean=0.0, std=0.02)
            nn.init.normal_(mlp.projector[layer].bias, mean=0.0, std=0.02)
        return mlp

    def _init_state_encoder(self):
        from vitra.utils.nn_utils import MLPProjector

        mlp = MLPProjector(self.state_and_mask_dim, self.hidden_size)
        for layer in (0, 2):
            nn.init.normal_(mlp.projector[layer].weight, mean=0.0, std=0.02)
            nn.init.normal_(mlp.projector[layer].bias, mean=0.0, std=0.02)
        return mlp

    def _init_act_model(self):
        from vitra.models.action_model.diffusion_policy import DiffusionPolicy

        token_size = self.act_model_configs.get("token_size", -1)
        if token_size <= 0:
            token_size = self.hidden_size
        action_head = DiffusionPolicy(
            model_type=self.act_model_configs.get("model_type", "DiT-B"),
            token_size=token_size,
            in_channels=self.act_model_configs.get("action_dim", 192),
            future_action_window_size=self.future_action_window_size,
            past_action_window_size=self.past_action_window_size,
            use_state=self.use_state,
            action_type=self.configs["train_dataset"].get("action_type", "angle"),
            state_dim=self.configs["state_encoder"]["state_dim"] if self.use_state == "DiT" else None,
            loss_type=self.configs.get("loss_type", "human"),
        )
        for param in action_head.parameters():
            assert param.dtype == torch.float32
        return action_head

    def trainable_params_setup(self):
        freeze_option = self.train_setup_configs.get("freeze_option", "only_head_and_token")
        self.model.config.use_cache = False

        if freeze_option == "full_finetune":
            self.model.requires_grad_(True)
        elif freeze_option == "freeze_vision_encoder":
            self.model.requires_grad_(True)
            self.vision_tower.requires_grad_(False)
        else:
            self.model.requires_grad_(False)

        if self.act_model is not None:
            self.act_model.requires_grad_(True)
        if self.use_state == "VLM":
            self.vlm_state_encoder.requires_grad_(True)
        if self.use_fov:
            self.fov_encoder.requires_grad_(True)
        if self.cognition_token is not None:
            self.cognition_token.requires_grad_(True)

    @property
    def model(self):
        return self.backbone

    @property
    def hidden_size(self):
        return self.model.config.text_config.hidden_size

    @property
    def word_embedding(self):
        return self.model.get_input_embeddings()

    @property
    def vision_tower(self):
        return self.model.model.visual

    @property
    def text_tower(self):
        return self.model.model.language_model

    def extract_cognition_token(self, output_hs, attention_mask):
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, output_hs.size(-1))
        return output_hs.gather(1, expanded_indices.unsqueeze(1))

    def _append_condition_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        current_state: Optional[torch.Tensor],
        current_state_mask: Optional[torch.Tensor],
        fov: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = input_ids.shape[0]
        embed_layer = self.model.get_input_embeddings()
        word_embeds = embed_layer(input_ids)
        embeds_list = [word_embeds]
        masks_list = [attention_mask]
        extra = 0

        if self.use_state == "VLM":
            current_state = current_state * current_state_mask.to(current_state.dtype)
            state_embeds = self.vlm_state_encoder(
                torch.cat([current_state, current_state_mask.to(current_state.dtype)], dim=1)
            )
            embeds_list.append(state_embeds.unsqueeze(1))
            masks_list.append(torch.ones((b, 1), dtype=torch.bool, device=input_ids.device))
            extra += 1

        if self.use_fov:
            fov_embeds = self.fov_encoder(fov)
            embeds_list.append(fov_embeds.unsqueeze(1))
            masks_list.append(torch.ones((b, 1), dtype=torch.bool, device=input_ids.device))
            extra += 1

        cog_embeds = embed_layer(
            torch.full((b, 1), self.cognition_token_id, dtype=input_ids.dtype, device=input_ids.device)
        )
        if self.cognition_token is not None:
            cog_embeds = self.cognition_token.unsqueeze(0).unsqueeze(0).expand(b, -1, -1)
        embeds_list.append(cog_embeds)
        masks_list.append(torch.ones((b, 1), dtype=torch.bool, device=input_ids.device))
        extra += 1

        inputs_embeds = torch.cat(embeds_list, dim=1)
        inputs_masks = torch.cat(masks_list, dim=1)

        pad_id = self.tokenizer.pad_token_id or 0
        pad_ids = torch.full((b, extra), pad_id, dtype=input_ids.dtype, device=input_ids.device)
        ext_input_ids = torch.cat([input_ids, pad_ids], dim=1)
        return ext_input_ids, inputs_embeds, inputs_masks

    def prepare_vlm_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        patch_positions: torch.Tensor,
        current_state_mask: Optional[torch.Tensor] = None,
        current_state: Optional[torch.Tensor] = None,
        fov: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        ext_input_ids, inputs_embeds, inputs_masks = self._append_condition_tokens(
            input_ids, attention_mask, current_state, current_state_mask, fov
        )

        dtype = next(self.model.parameters()).dtype
        inputs_embeds = inputs_embeds.to(dtype=dtype)
        forward_kwargs = dict(
            input_ids=ext_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=inputs_masks,
            pixel_values=pixel_values.to(dtype=dtype),
            image_grid_thw=image_grid_thw,
            patch_positions=patch_positions,
            output_hidden_states=True,
            return_dict=True,
            use_cache=use_cache,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            outputs = self.model(**forward_kwargs)
        return outputs.hidden_states[-1], inputs_masks

    def _forward_act_model(
        self,
        vlm_features: torch.Tensor,
        action_labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        mode: str = "train",
        repeated_diffusion_steps: int = 1,
        cfg_scale: float = 5.0,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        **kwargs,
    ):
        actions = None
        action_loss = None
        b = vlm_features.shape[0]
        action_features = self.extract_cognition_token(vlm_features, attention_mask)
        model_dtype = next(self.act_model.net.parameters()).dtype
        action_features = action_features.to(model_dtype)

        action_features_repeated = action_features.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
        action_masks_repeated = action_masks.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
        action_features_repeated = action_features_repeated.view(
            b * repeated_diffusion_steps, 1, action_features.shape[-1]
        )
        action_masks_repeated = action_masks_repeated.view(
            b * repeated_diffusion_steps, action_masks.shape[1], action_masks.shape[2]
        )

        if self.use_state == "DiT":
            current_state_repeated = current_state.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_repeated = current_state_repeated.view(
                b * repeated_diffusion_steps, 1, current_state.shape[1]
            )
            current_state_mask_repeated = current_state_mask.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_mask_repeated = current_state_mask_repeated.view(
                b * repeated_diffusion_steps, 1, current_state_mask.shape[1]
            )
        else:
            current_state_repeated = None
            current_state_mask_repeated = None

        if mode == "train":
            actions_repeated = action_labels.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
            actions_repeated = actions_repeated.view(
                b * repeated_diffusion_steps, action_labels.shape[1], action_labels.shape[2]
            )
            if self.use_state == "DiT":
                action_loss = self.act_model.loss(
                    actions_repeated,
                    action_features_repeated,
                    action_masks_repeated,
                    current_state_repeated,
                    current_state_mask_repeated,
                )
            else:
                action_loss = self.act_model.loss(
                    actions_repeated, action_features_repeated, action_masks_repeated
                )
            return actions, action_loss

        actions = self.act_model.sample(
            action_features_repeated,
            cfg_scale,
            current_state_repeated,
            current_state_mask_repeated,
            use_ddim,
            num_ddim_steps,
            action_masks_repeated,
        )
        return actions, action_loss

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        patch_positions: torch.Tensor,
        action_labels: torch.Tensor = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        mode="train",
        **kwargs,
    ):
        assert mode == "train"
        loss = {}
        output_hs, inputs_masks = self.prepare_vlm_features(
            pixel_values,
            input_ids,
            attention_mask,
            image_grid_thw,
            patch_positions,
            current_state_mask,
            current_state,
            fov,
            use_cache,
            **kwargs,
        )
        _, action_loss = self._forward_act_model(
            vlm_features=output_hs,
            action_labels=action_labels,
            attention_mask=inputs_masks,
            action_masks=action_masks,
            current_state=current_state,
            current_state_mask=current_state_mask,
            mode=mode,
            repeated_diffusion_steps=self.repeated_diffusion_steps,
        )
        self._update_loss(loss, action_loss)
        return loss

    @staticmethod
    def _update_loss(loss, new_loss, suffix=None):
        def get_key(k, d):
            if suffix is not None:
                return f"{k}_{suffix}"
            ind = 0
            while True:
                new_k = k if ind == 0 else f"{k}_{ind}"
                if new_k not in d:
                    return new_k
                ind += 1

        for k in new_loss:
            loss[get_key(k, loss)] = new_loss[k]
        loss["loss"] = sum(v for k, v in loss.items() if "loss" in k)
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        image_grid_thw,
        patch_positions,
        current_state,
        current_state_mask,
        use_ddim=True,
        num_ddim_steps=10,
        cfg_scale=5.0,
        action_mask_torch=None,
        fov=None,
        sample_times=1,
    ) -> np.ndarray:
        b = current_state.shape[0]
        assert b == 1

        if action_mask_torch is None:
            x_mask = torch.zeros(b, self.chunk_size, self.act_model.in_channels, device=input_ids.device)
            x_mask[:, :, 51:102] = 1.0
        else:
            x_mask = action_mask_torch.to(input_ids.device)

        output_hs, inputs_masks = self.prepare_vlm_features(
            pixel_values,
            input_ids,
            attention_mask,
            image_grid_thw,
            patch_positions,
            current_state_mask,
            current_state,
            fov,
        )
        samples, _ = self._forward_act_model(
            vlm_features=output_hs,
            attention_mask=inputs_masks,
            action_masks=x_mask,
            current_state=current_state,
            current_state_mask=current_state_mask,
            mode="eval",
            repeated_diffusion_steps=sample_times,
            cfg_scale=cfg_scale,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
        )
        return samples.cpu().numpy() * x_mask.cpu().numpy()
