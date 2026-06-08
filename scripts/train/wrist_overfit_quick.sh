#!/usr/bin/env bash
# Quick pipeline: short overfit train + eval on LLaVA-OneVision-2 codec features.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "${REPO_ROOT}/scripts/activate_env.sh"

OUT="${OUT:-${REPO_ROOT}/outputs/wrist_overfit}"
EVAL_OUT="${EVAL_OUT:-${REPO_ROOT}/outputs/wrist_overfit_eval}"
MODE="${MODE:-llava}"
MODEL="${MODEL:-/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/yangtao/monkey_asset/LLaVA-OneVision-2-8B-Instruct}"
EPOCHS="${EPOCHS:-10}"
NORM="${REPO_ROOT}/outputs/wrist_norm_stats.json"

echo "=== Wrist norm stats ==="
python -m llava.wrist.normalize --data_root "${REPO_ROOT}/data" --output "$NORM"

echo "=== Short overfit train (${MODE}, ${EPOCHS} epochs) ==="
python llava/train/train_wrist_overfit.py \
  --mode "$MODE" \
  --model_name_or_path "$MODEL" \
  --data_root "${REPO_ROOT}/data" \
  --output_dir "$OUT" \
  --epochs "$EPOCHS" \
  --eval_every 5 \
  --batch_size 16 \
  --lr 1e-3 \
  --hidden 2048 \
  --depth 4 \
  --target_mae 0.002 \
  --norm_stats "$NORM" \
  "$@"

CKPT="${OUT}/wrist_overfit_best.pt"
[[ -f "$CKPT" ]] || CKPT="${OUT}/wrist_overfit_last.pt"

echo "=== Eval checkpoint: $CKPT ==="
python llava/eval/eval_wrist_overfit.py \
  --checkpoint "$CKPT" \
  --data_root "${REPO_ROOT}/data" \
  --output_dir "$EVAL_OUT" \
  --model_name_or_path "$MODEL" \
  --norm_stats "$NORM"

echo "Done. ckpt=$CKPT  eval=${EVAL_OUT}/metrics_train.json  viz=${EVAL_OUT}/*_codec.mp4"
