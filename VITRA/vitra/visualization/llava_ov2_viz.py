"""Visualize future hand meshes on anchor frame (all 16 steps overlaid)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

VITRA_ROOT = Path(__file__).resolve().parents[2]
if str(VITRA_ROOT) not in sys.path:
    sys.path.insert(0, str(VITRA_ROOT))


def _patch_numpy_aliases_for_chumpy() -> None:
    import numpy as np

    aliases = {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": object,
        "unicode": np.str_,
        "str": np.str_,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


from libs.models.mano_wrapper import MANO
from visualization.render_utils import Renderer
from visualization.video_utils import read_video_frames, resize_frames_to_long_side, save_to_video
from vitra.datasets.llava_ov2_video import DEFAULT_CODEC_HISTORY_FRAMES
from vitra.utils.data_utils import GaussianNormalizer, recon_traj

GT_COLOR = np.array([0.0, 0.86, 0.20], dtype=np.float32)
PRED_COLOR = np.array([0.92, 0.18, 0.18], dtype=np.float32)
HISTORY_COLOR = (80, 220, 120)
ANCHOR_COLOR = (40, 220, 255)
MASKED_COLOR = (120, 120, 120)
RENDER_LONG_SIDE = 480
# use_rel=False: trans/rot step-relative, hand joints absolute
RECON_ABS_JOINT = True
RECON_REL_MODE = "step"


def unnormalize_padded_state(
    norm_state: np.ndarray,
    normalizer: GaussianNormalizer,
) -> Tuple[np.ndarray, np.ndarray]:
    """212-dim padded norm state -> unnormalized left/right 51-dim hand states."""
    norm_state = np.asarray(norm_state, dtype=np.float32)
    norm_full = np.zeros(122, dtype=np.float32)
    norm_full[:51] = norm_state[:51]
    norm_full[61:112] = norm_state[51:102]
    raw_full = normalizer.unnormalize_state(norm_full)
    return raw_full[:51].copy(), raw_full[61:112].copy()


def unnormalize_padded_actions(
    norm_actions: np.ndarray,
    normalizer: GaussianNormalizer,
) -> np.ndarray:
    """(K,192) normalized actions -> (K,102) physical actions."""
    out = np.asarray(norm_actions, dtype=np.float32).copy()
    for i in range(out.shape[0]):
        out[i, :102] = normalizer.unnormalize_action(out[i, :102])
    return out[:, :102]


def trajectories_from_normalized(
    norm_state: np.ndarray,
    norm_actions: np.ndarray,
    normalizer: GaussianNormalizer,
) -> Tuple[np.ndarray, np.ndarray]:
    """norm state + norm actions -> unnormalize -> recon_traj (same as training pipeline)."""
    state_l, state_r = unnormalize_padded_state(norm_state, normalizer)
    raw_actions = unnormalize_padded_actions(norm_actions, normalizer)
    act_l, act_r = split_hand_actions(raw_actions)
    left = recon_traj(state_l, act_l, abs_joint=RECON_ABS_JOINT, rel_mode=RECON_REL_MODE)
    right = recon_traj(state_r, act_r, abs_joint=RECON_ABS_JOINT, rel_mode=RECON_REL_MODE)
    return left, right


def split_hand_actions(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    return actions[:, :51], actions[:, 51:102]


def process_single_hand_labels(
    hand_labels: dict,
    hand_mask: np.ndarray,
    mano: MANO,
    *,
    is_left: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    t = len(hand_mask)
    wrist_worldspace = hand_labels["transl_worldspace"].reshape(-1, 1, 3)
    wrist_orientation = hand_labels["global_orient_worldspace"]
    beta = hand_labels["beta"]
    pose = hand_labels["hand_pose"].copy()

    identity = np.eye(3, dtype=pose.dtype)
    identity_block = np.broadcast_to(identity, (pose.shape[1], 3, 3))
    mask_indices = hand_mask == 0
    if np.any(mask_indices):
        pose[mask_indices] = identity_block

    beta_torch = torch.from_numpy(beta).float().cuda().unsqueeze(0).repeat(t, 1)
    pose_torch = torch.from_numpy(pose).float().cuda()
    global_rot_placeholder = torch.eye(3).float().unsqueeze(0).unsqueeze(0).cuda().repeat(t, 1, 1, 1)
    mano_out = mano(betas=beta_torch, hand_pose=pose_torch, global_orient=global_rot_placeholder)
    verts = mano_out.vertices.cpu().numpy()
    joints = mano_out.joints.cpu().numpy()

    if is_left:
        verts[:, :, 0] *= -1
        joints[:, :, 0] *= -1

    verts_cam = (
        wrist_orientation @ (verts - joints[:, 0][:, None]).transpose(0, 2, 1)
    ).transpose(0, 2, 1) + wrist_worldspace
    return verts_cam, joints[:, 0]


def euler_traj_to_rotmat_traj(euler_traj: np.ndarray, t: int) -> np.ndarray:
    hand_pose = euler_traj.reshape(-1, 3)
    pose_matrices = R.from_euler("xyz", hand_pose).as_matrix()
    return pose_matrices.reshape(t, 15, 3, 3)


def build_step_hand_masks(
    hand_state_mask: np.ndarray,
    action_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    k = action_mask.shape[0]
    t = k + 1
    left = np.zeros(t, dtype=np.int32)
    right = np.zeros(t, dtype=np.int32)
    left[0] = int(hand_state_mask[0])
    right[0] = int(hand_state_mask[1])
    for i in range(k):
        left[i + 1] = int(action_mask[i, :51].any())
        right[i + 1] = int(action_mask[i, 51:102].any())
    return left, right


def _scale_intrinsics(intrinsics: np.ndarray, old_hw: Tuple[int, int], new_hw: Tuple[int, int]) -> np.ndarray:
    oh, ow = old_hw
    nh, nw = new_hw
    sx, sy = nw / float(ow), nh / float(oh)
    out = intrinsics.copy()
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def _traj_to_verts(
    traj: np.ndarray,
    beta: np.ndarray,
    hand_mask: np.ndarray,
    mano: MANO,
    *,
    is_left: bool,
) -> np.ndarray:
    t = len(hand_mask)
    labels = {
        "transl_worldspace": traj[:, 0:3],
        "global_orient_worldspace": R.from_euler("xyz", traj[:, 3:6]).as_matrix(),
        "hand_pose": euler_traj_to_rotmat_traj(traj[:, 6:51], t),
        "beta": beta,
    }
    verts, _ = process_single_hand_labels(labels, hand_mask, mano, is_left=is_left)
    return verts


def _overlay_mesh_rgb(
    base_rgb: np.ndarray,
    renderer: Renderer,
    verts_list,
    faces_list,
    colors_list,
    alpha: float = 0.85,
) -> np.ndarray:
    if not verts_list:
        return base_rgb
    img = base_rgb.astype(np.float32) / 255.0
    rend, mask = renderer.render(verts_list, faces_list, colors_list)
    color_mesh = rend.astype(np.float32) / 255.0
    valid = mask[..., None].astype(np.float32)
    out = img * (1.0 - valid) + color_mesh * valid * alpha + img * valid * (1.0 - alpha)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _overlay_future_stack(
    base_rgb: np.ndarray,
    renderer: Renderer,
    left_v: np.ndarray,
    right_v: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    faces_left,
    faces_right,
    color: np.ndarray,
    *,
    include_t0: bool = False,
    alpha: float = 0.82,
) -> np.ndarray:
    """Overlay all future hand meshes (steps 1..K, optionally t0) on one frame."""
    rgb = base_rgb.copy()
    t_total = left_v.shape[0]
    start = 0 if include_t0 else 1
    for step in range(start, t_total):
        for verts, mask, faces in (
            (left_v, left_mask, faces_left),
            (right_v, right_mask, faces_right),
        ):
            if step >= len(mask) or mask[step] == 0:
                continue
            v = torch.from_numpy(verts[step]).float().cuda()
            c = torch.from_numpy(color).float().unsqueeze(0).repeat(778, 1).cuda()
            rgb = _overlay_mesh_rgb(rgb, renderer, [v], [faces], [c], alpha=alpha)
    return rgb


def _pad_frames_for_video(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return frames
    h = frames[0].shape[0]
    pad_h = (16 - h % 16) % 16
    if pad_h == 0:
        return frames
    return [np.pad(f, ((0, pad_h), (0, 0), (0, 0)), mode="edge") for f in frames]


def _draw_banner(
    rgb: np.ndarray,
    text: str,
    *,
    bar_color: Tuple[int, int, int],
) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    bar_h = 28
    overlay = out.copy()
    overlay[:bar_h] = (overlay[:bar_h].astype(np.float32) * 0.35).astype(np.uint8)
    out[:bar_h] = overlay[:bar_h]
    cv2.rectangle(out, (0, 0), (w - 1, bar_h - 1), bar_color, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _draw_frame_border(rgb: np.ndarray, color: Tuple[int, int, int], thickness: int = 4) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness, cv2.LINE_AA)
    return out


def _mask_future_frame(rgb: np.ndarray, alpha: float = 0.72) -> np.ndarray:
    out = rgb.astype(np.float32)
    gray = np.full_like(out, 32.0)
    masked = out * (1.0 - alpha) + gray * alpha
    h, w = masked.shape[:2]
    cv2.line(masked, (0, 0), (w - 1, h - 1), (90, 90, 90), 2, cv2.LINE_AA)
    cv2.line(masked, (w - 1, 0), (0, h - 1), (90, 90, 90), 2, cv2.LINE_AA)
    return masked.astype(np.uint8)


class LlavaOV2MeshVisualizer:
    def __init__(self, mano_model_path: str):
        _patch_numpy_aliases_for_chumpy()
        self.mano = MANO(model_path=mano_model_path).cuda()
        faces_right = torch.from_numpy(self.mano.faces).float().cuda()
        self.faces_left = faces_right[:, [0, 2, 1]]
        self.faces_right = faces_right

    def render_anchor_future_stack(
        self,
        video_path: str,
        anchor: int,
        norm_state: np.ndarray,
        norm_gt_actions: np.ndarray,
        action_masks: np.ndarray,
        intrinsics: np.ndarray,
        normalizer: GaussianNormalizer,
        out_path: str,
        *,
        beta_left: np.ndarray,
        beta_right: np.ndarray,
        hand_state_mask: np.ndarray,
        norm_pred_actions: Optional[np.ndarray] = None,
        fps: int = 8,
        context: int = 8,
        gt_only: bool = False,
    ) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        all_frames = read_video_frames(cap)
        cap.release()
        if not all_frames:
            raise RuntimeError(f"No frames read from {video_path}")

        anchor = int(np.clip(anchor, 0, len(all_frames) - 1))
        orig_h, orig_w = all_frames[0].shape[:2]
        frames = resize_frames_to_long_side(all_frames, RENDER_LONG_SIDE)
        h, w = frames[0].shape[:2]
        k = _scale_intrinsics(np.asarray(intrinsics, dtype=np.float32), (orig_h, orig_w), (h, w))

        gt_left, gt_right = trajectories_from_normalized(norm_state, norm_gt_actions, normalizer)
        left_mask, right_mask = build_step_hand_masks(hand_state_mask, action_masks)

        gt_left_v = _traj_to_verts(gt_left, beta_left, left_mask, self.mano, is_left=True)
        gt_right_v = _traj_to_verts(gt_right, beta_right, right_mask, self.mano, is_left=False)

        pred_left_v = pred_right_v = None
        if norm_pred_actions is not None and not gt_only:
            pred_left, pred_right = trajectories_from_normalized(norm_state, norm_pred_actions, normalizer)
            pred_left_v = _traj_to_verts(pred_left, beta_left, left_mask, self.mano, is_left=True)
            pred_right_v = _traj_to_verts(pred_right, beta_right, right_mask, self.mano, is_left=False)

        renderer = Renderer(w, h, (k[0, 0], k[1, 1]), "cuda")
        future_k = norm_gt_actions.shape[0]
        start = max(0, anchor - context)
        end = min(len(all_frames), anchor + context + 1)

        anchor_rgb = cv2.cvtColor(frames[anchor], cv2.COLOR_BGR2RGB)
        anchor_rgb = _overlay_future_stack(
            anchor_rgb, renderer, gt_left_v, gt_right_v, left_mask, right_mask,
            self.faces_left, self.faces_right, GT_COLOR, include_t0=False, alpha=0.85,
        )
        if pred_left_v is not None and pred_right_v is not None:
            anchor_rgb = _overlay_future_stack(
                anchor_rgb, renderer, pred_left_v, pred_right_v, left_mask, right_mask,
                self.faces_left, self.faces_right, PRED_COLOR, include_t0=False, alpha=0.85,
            )

        title1 = f"anchor={anchor}  future steps={future_k} (all overlaid)"
        title2 = "GT=green  Pred=red" if pred_left_v is not None else "GT=green"
        cv2.putText(anchor_rgb, title1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(anchor_rgb, title2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)

        out_frames = []
        for fi in range(start, end):
            if fi == anchor:
                out_frames.append(anchor_rgb)
            else:
                bgr = frames[fi].copy()
                cv2.putText(
                    bgr, f"frame {fi} (anchor={anchor})", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA,
                )
                out_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_to_video(_pad_frames_for_video(out_frames), out_path, fps=fps)

        import imageio.v2 as imageio
        png_path = os.path.splitext(out_path)[0] + "_anchor.png"
        imageio.imwrite(png_path, anchor_rgb)

    def render_anchor_dataset_diagnostic(
        self,
        video_path: str,
        anchor: int,
        total_frames: int,
        norm_state: np.ndarray,
        norm_gt_actions: np.ndarray,
        action_masks: np.ndarray,
        intrinsics: np.ndarray,
        normalizer: GaussianNormalizer,
        out_path: str,
        *,
        beta_left: np.ndarray,
        beta_right: np.ndarray,
        hand_state_mask: np.ndarray,
        fps: int = 8,
    ) -> dict:
        """One video for a single anchor: label history / anchor / masked future over full episode."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        all_frames = read_video_frames(cap)
        cap.release()
        if not all_frames:
            raise RuntimeError(f"No frames read from {video_path}")

        total_frames = min(int(total_frames), len(all_frames))
        anchor = int(np.clip(anchor, 0, total_frames - 1))
        orig_h, orig_w = all_frames[0].shape[:2]
        frames = resize_frames_to_long_side(all_frames[:total_frames], RENDER_LONG_SIDE)
        h, w = frames[0].shape[:2]
        k = _scale_intrinsics(np.asarray(intrinsics, dtype=np.float32), (orig_h, orig_w), (h, w))

        gt_left, gt_right = trajectories_from_normalized(norm_state, norm_gt_actions, normalizer)
        left_mask, right_mask = build_step_hand_masks(hand_state_mask, action_masks)
        gt_left_v = _traj_to_verts(gt_left, beta_left, left_mask, self.mano, is_left=True)
        gt_right_v = _traj_to_verts(gt_right, beta_right, right_mask, self.mano, is_left=False)

        renderer = Renderer(w, h, (k[0, 0], k[1, 1]), "cuda")
        history_len = anchor + 1
        padded_history_len = max(history_len, DEFAULT_CODEC_HISTORY_FRAMES)
        per_hand_valid = action_masks[:, :102].any(axis=1)
        valid_future_steps = int(per_hand_valid.sum())

        out_frames: list[np.ndarray] = []
        for fi in range(total_frames):
            bgr = frames[fi].copy()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if fi < anchor:
                rgb = _draw_frame_border(rgb, HISTORY_COLOR, thickness=3)
                label = f"HISTORY  f{fi}"
                bar_color = HISTORY_COLOR
            elif fi == anchor:
                rgb = _overlay_future_stack(
                    rgb,
                    renderer,
                    gt_left_v,
                    gt_right_v,
                    left_mask,
                    right_mask,
                    self.faces_left,
                    self.faces_right,
                    GT_COLOR,
                    include_t0=False,
                    alpha=0.85,
                )
                rgb = _draw_frame_border(rgb, ANCHOR_COLOR, thickness=5)
                label = f"ANCHOR  f{fi}  GTx{valid_future_steps}"
                bar_color = ANCHOR_COLOR
            else:
                rgb = _mask_future_frame(rgb, alpha=0.74)
                rgb = _draw_frame_border(rgb, MASKED_COLOR, thickness=3)
                label = f"MASKED  f{fi}"
                bar_color = MASKED_COLOR

            rgb = _draw_banner(rgb, label, bar_color=bar_color)
            out_frames.append(rgb)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_to_video(_pad_frames_for_video(out_frames), out_path, fps=fps)

        return {
            "anchor": anchor,
            "total_frames": total_frames,
            "history_len": history_len,
            "padded_history_len": padded_history_len,
            "valid_future_steps": valid_future_steps,
            "future_k": int(norm_gt_actions.shape[0]),
            "out_path": out_path,
        }


def render_episode_dataset_diagnostics(
    dataset,
    episode_id: str,
    out_dir: str,
    normalizer: GaussianNormalizer,
    *,
    mano_model_path: Optional[str] = None,
    fps: int = 8,
    max_anchors: int = 0,
) -> dict:
    """Render one diagnostic video per anchor (all indexed frames except the last)."""
    if mano_model_path is None:
        mano_model_path = str(VITRA_ROOT / "weights/mano")

    anchors = dataset.get_episode_anchor_frames(episode_id)
    if max_anchors > 0:
        anchors = anchors[:max_anchors]
    total_frames = dataset.get_episode_length(episode_id)
    ep_out = os.path.join(out_dir, episode_id)
    os.makedirs(ep_out, exist_ok=True)

    viz = LlavaOV2MeshVisualizer(mano_model_path)
    results = []
    for anchor in anchors:
        sample = dataset.get_norm_viz_sample_by_frame(episode_id, anchor)
        out_path = os.path.join(ep_out, f"anchor_{anchor:04d}.mp4")
        meta = viz.render_anchor_dataset_diagnostic(
            video_path=sample["video_path"],
            anchor=anchor,
            total_frames=total_frames,
            norm_state=sample["norm_state"],
            norm_gt_actions=sample["norm_actions"],
            action_masks=sample["action_masks"],
            intrinsics=sample["intrinsics"],
            normalizer=normalizer,
            out_path=out_path,
            beta_left=sample["beta_left"],
            beta_right=sample["beta_right"],
            hand_state_mask=sample["hand_state_mask"],
            fps=fps,
        )
        meta["episode_id"] = episode_id
        meta["sample_idx"] = sample["sample_idx"]
        results.append(meta)

    summary = {
        "episode_id": episode_id,
        "total_frames": total_frames,
        "num_anchor_videos": len(results),
        "expected_num_videos": max(total_frames - 1, 0),
        "anchors": anchors,
        "results": results,
    }
    return summary


def render_sample_future_video(
    video_path: str,
    anchor: int,
    norm_state: np.ndarray,
    norm_gt_actions: np.ndarray,
    action_masks: np.ndarray,
    intrinsics: np.ndarray,
    normalizer: GaussianNormalizer,
    out_path: str,
    *,
    beta_left: np.ndarray,
    beta_right: np.ndarray,
    hand_state_mask: np.ndarray,
    norm_pred_actions: Optional[np.ndarray] = None,
    mano_model_path: Optional[str] = None,
    gt_only: bool = False,
    fps: int = 8,
    context: int = 8,
) -> None:
    if mano_model_path is None:
        mano_model_path = str(VITRA_ROOT / "weights/mano")
    viz = LlavaOV2MeshVisualizer(mano_model_path)
    viz.render_anchor_future_stack(
        video_path=video_path,
        anchor=anchor,
        norm_state=norm_state,
        norm_gt_actions=norm_gt_actions,
        action_masks=action_masks,
        intrinsics=intrinsics,
        normalizer=normalizer,
        out_path=out_path,
        beta_left=beta_left,
        beta_right=beta_right,
        hand_state_mask=hand_state_mask,
        norm_pred_actions=norm_pred_actions,
        fps=fps,
        context=context,
        gt_only=gt_only,
    )
