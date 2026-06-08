#!/usr/bin/env bash
# Mesh visualization for an existing memorize checkpoint (gt vs pred overlay videos).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VITRA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "${REPO_ROOT}/scripts/activate_env.sh"
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export HF_HUB_DISABLE_PROMPT_FOR_TRUST_REMOTE_CODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT="${OUT:-${VITRA_ROOT}/outputs/monkey_cyt_memorize}"
EVAL_OUT="${EVAL_OUT:-${OUT}/eval}"
VIZ_OUT="${VIZ_OUT:-${EVAL_OUT}/viz}"
CFG="${CFG:-${VITRA_ROOT}/vitra/configs/human_llava_ov2_memorize.json}"
CKPT="${CKPT:-${OUT}/vitra_llava_ov2_memorize_best.pt}"
[[ -f "$CKPT" ]] || CKPT="${OUT}/vitra_llava_ov2_memorize_last.pt"
MAX_PRED_VIZ="${MAX_PRED_VIZ:-8}"
PRED_INDICES="${PRED_INDICES:-0,1,2,3,4,5,6,7}"
LOG="${VIZ_OUT}/viz.log"

mkdir -p "$VIZ_OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== VITRA memorize mesh viz ==="
echo "host=$(hostname) time=$(date -Is)"
echo "ckpt=$CKPT viz_out=$VIZ_OUT pred_indices=$PRED_INDICES max_pred_viz=$MAX_PRED_VIZ"

python "${VITRA_ROOT}/scripts/eval_llava_ov2_overfit.py" \
  --checkpoint "$CKPT" \
  --config "$CFG" \
  --output_dir "$VIZ_OUT" \
  --num_eval 0 \
  --pred_indices "$PRED_INDICES" \
  --no_gt_viz_all \
  --max_pred_viz "$MAX_PRED_VIZ"

echo "=== Done ==="
echo "log=$LOG"
echo "videos=${VIZ_OUT}/pred_viz/*.mp4"
