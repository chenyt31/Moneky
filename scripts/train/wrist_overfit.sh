#!/usr/bin/env bash
# Overfit wrist trajectories on data/ (2 episodes). Expect train MAE -> ~0.
# Uses LLaVA-OneVision-2 codec backbone when MODE=llava|llava_full.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "${REPO_ROOT}/scripts/activate_env.sh"

OUT="${OUT:-${REPO_ROOT}/outputs/wrist_overfit}"
MODE="${MODE:-llava}"  # mlp | llava | llava_full
MODEL="${MODEL:-/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/yangtao/monkey_asset/LLaVA-OneVision-2-8B-Instruct}"

echo "=== Compute wrist normalization stats ==="
python -m llava.wrist.normalize \
  --data_root "${REPO_ROOT}/data" \
  --output "${REPO_ROOT}/outputs/wrist_norm_stats.json"

echo "=== Overfit wrist (${MODE}) with LLaVA-OneVision-2 codec ==="
python llava/train/train_wrist_overfit.py \
  --mode "$MODE" \
  --model_name_or_path "$MODEL" \
  --data_root "${REPO_ROOT}/data" \
  --output_dir "$OUT" \
  --epochs 800 \
  --batch_size 16 \
  --lr 1e-3 \
  --hidden 2048 \
  --depth 4 \
  --target_mae 0.002 \
  --norm_stats "${REPO_ROOT}/outputs/wrist_norm_stats.json" \
  "$@"

echo "Checkpoint: ${OUT}/wrist_overfit_best.pt"
