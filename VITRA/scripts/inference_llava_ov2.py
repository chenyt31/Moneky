"""
LLaVA-OV2 VITRA inference: history video [0..anchor] + current state -> future action chunk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VITRA_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VITRA_ROOT) not in sys.path:
    sys.path.insert(0, str(VITRA_ROOT))

from vitra.utils.hf_env import enable_hf_offline

enable_hf_offline()

import numpy as np
import torch

from vitra.datasets.llava_ov2_collator import build_llava_ov2_collator
from vitra.datasets.llava_ov2_dataset import LlavaOV2HumanDataset
from vitra.models.vla_builder import build_vla
from vitra.utils.data_utils import GaussianNormalizer, read_dataset_statistics
from vitra.visualization.llava_ov2_viz import render_sample_future_video


def _vision_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _to_device(batch: dict, device: torch.device) -> dict:
    vdtype = _vision_dtype(device)
    out = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            out[k] = v
        elif k == "pixel_values":
            out[k] = v.to(device, dtype=vdtype)
        else:
            out[k] = v.to(device)
    return out


def load_model(checkpoint: str, config: dict, device: torch.device):
    model = build_vla(config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    model.backbone.eval()
    return model, ckpt


@torch.no_grad()
def predict_one(model, batch: dict, device: torch.device) -> np.ndarray:
    batch = _to_device(batch, device)
    samples = model.predict_action(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        image_grid_thw=batch["image_grid_thw"],
        patch_positions=batch["patch_positions"],
        current_state=batch["current_state"],
        current_state_mask=batch["current_state_mask"],
        fov=batch["fov"],
        action_mask_torch=batch["action_masks"],
        sample_times=1,
    )
    return samples[0]


def unnormalize_action_chunk(
    norm_actions: np.ndarray,
    normalizer: GaussianNormalizer,
) -> np.ndarray:
    out = norm_actions.copy()
    for i in range(out.shape[0]):
        out[i, :102] = normalizer.unnormalize_action(out[i, :102])
    return out


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    with open(args.config) as f:
        cfg = json.load(f)
    ds_cfg = cfg["train_dataset"]

    dataset = LlavaOV2HumanDataset(
        data_root=ds_cfg["data_root_dir"],
        statistics_path=ds_cfg["statistics_path"],
        action_future_window_size=ds_cfg.get("action_future_window_size", 15),
        max_samples=ds_cfg.get("max_samples", 32),
        augmentation=False,
        normalization=True,
    )
    normalizer = GaussianNormalizer(read_dataset_statistics(ds_cfg["statistics_path"]))

    model, ckpt = load_model(args.checkpoint, cfg, device)
    cache_root = os.path.join(args.output_dir, "codec_cache")
    collator = build_llava_ov2_collator(cfg, model.processor, cache_root)

    sample_idx = args.sample_idx
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise ValueError(f"sample_idx {sample_idx} out of range [0, {len(dataset)})")
    item = dataset[sample_idx]
    batch = collator([item])

    pred_norm = predict_one(model, batch, device)
    pred = unnormalize_action_chunk(pred_norm, normalizer)
    gt_norm = batch["actions"][0].numpy()
    gt = unnormalize_action_chunk(gt_norm, normalizer)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"sample_{sample_idx:03d}_ep_{item['episode_id']}_anchor{item['frame_id']}"
    np.save(os.path.join(args.output_dir, f"{tag}_pred_actions.npy"), pred)
    np.save(os.path.join(args.output_dir, f"{tag}_gt_actions.npy"), gt)

    mask = batch["action_masks"][0].numpy()
    valid = mask.astype(bool)
    mae = float(np.abs(pred[valid] - gt[valid]).mean()) if valid.any() else float("nan")

    meta = {
        "checkpoint": args.checkpoint,
        "sample_idx": sample_idx,
        "episode_id": item["episode_id"],
        "frame_id": int(item["frame_id"]),
        "video_path": item["video_path"],
        "instruction": item["instruction"],
        "action_mae": mae,
        "pred_shape": list(pred.shape),
        "epoch": ckpt.get("epoch"),
    }
    with open(os.path.join(args.output_dir, f"{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))

    if args.visualize:
        viz = dataset.get_norm_viz_sample(sample_idx)
        out_mp4 = os.path.join(args.output_dir, f"{tag}_gt_pred_mesh.mp4")
        render_sample_future_video(
            video_path=viz["video_path"],
            anchor=int(viz["frame_id"]),
            norm_state=viz["norm_state"],
            norm_gt_actions=viz["norm_actions"],
            action_masks=viz["action_masks"],
            intrinsics=viz["intrinsics"],
            normalizer=normalizer,
            out_path=out_mp4,
            beta_left=viz["beta_left"],
            beta_right=viz["beta_right"],
            hand_state_mask=viz["hand_state_mask"],
            norm_pred_actions=pred_norm,
            mano_model_path=args.mano_path,
            fps=args.fps,
        )
        print(f"saved visualization: {out_mp4}")


def parse_args():
    p = argparse.ArgumentParser(description="LLaVA-OV2 VITRA inference (full video + state -> actions)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--config",
        default=str(VITRA_ROOT / "vitra/configs/human_llava_ov2_overfit.json"),
    )
    p.add_argument("--output_dir", default=str(VITRA_ROOT / "outputs/vitra_llava_ov2_infer"))
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--mano_path", default=str(VITRA_ROOT / "weights/mano"))
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
