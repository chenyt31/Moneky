#!/usr/bin/env bash
# Wrist SFT with LLaVA-OneVision 0.5B (HF) — video modality (pixel_values_videos).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/yangtao/VITRA/.vitra/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

MODEL="${MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/wrist_llava_ov_train}"
FRAMES="${FRAMES_UPBOUND:-8}"

mkdir -p "$OUT_DIR"

echo "=== Wrist xyz normalization stats ==="
python -m llava.wrist.normalize \
  --data_root "${DATA_ROOT}" \
  --output "${REPO_ROOT}/outputs/wrist_norm_stats.json"

echo "=== LLaVA-OneVision wrist SFT (full finetune, video) ==="
echo "MODEL=$MODEL  FRAMES_UPBOUND=$FRAMES"

# Default: train entire LLaVA backbone + wrist head. Pass --freeze_llava to train head only.
python llava/train/train_wrist.py \
  --model_name_or_path "$MODEL" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_DIR" \
  --frames_upbound "$FRAMES" \
  --future_k 16 \
  --batch_size 1 \
  --epochs 2 \
  --lr 1e-4 \
  --llava_lr 1e-5 \
  --num_workers 0 \
  --val_ratio 0 \
  --norm_stats "${REPO_ROOT}/outputs/wrist_norm_stats.json" \
  --log_steps 5 \
  --eval_steps 15 \
  "$@"

CKPT="${OUT_DIR}/wrist_llava_ov_best.pt"
[[ -f "$CKPT" ]] || CKPT="${OUT_DIR}/wrist_llava_ov_last.pt"

EVAL_SPLIT="${EVAL_SPLIT:-train}"
echo "=== Inference eval (${EVAL_SPLIT}, all training data when val_ratio=0) ==="
python llava/eval/eval_wrist.py \
  --checkpoint "$CKPT" \
  --model_name_or_path "$MODEL" \
  --data_root "$DATA_ROOT" \
  --output_dir "${REPO_ROOT}/outputs/wrist_llava_ov_eval" \
  --split "$EVAL_SPLIT" \
  --val_ratio 0 \
  --frames_upbound "$FRAMES" \
  --batch_size 1

echo "=== Visualize inference (GT green / pred red) ==="
python llava/eval/visualize_wrist_infer.py \
  --checkpoint "$CKPT" \
  --model_name_or_path "$MODEL" \
  --data_root "$DATA_ROOT" \
  --output_dir "${REPO_ROOT}/outputs/wrist_infer_viz" \
  --frames_upbound "$FRAMES" \
  --all

echo "Done. Viz (future16 GT green / pred red): ${REPO_ROOT}/outputs/wrist_infer_viz"
