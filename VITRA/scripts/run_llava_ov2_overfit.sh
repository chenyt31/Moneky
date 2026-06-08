#!/usr/bin/env bash
# VITRA + LLaVA-OV2 history-video overfit (frozen backbone, train action head).
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

OUT="${OUT:-${VITRA_ROOT}/outputs/vitra_llava_ov2_overfit}"
EVAL_OUT="${EVAL_OUT:-${VITRA_ROOT}/outputs/vitra_llava_ov2_eval}"
CFG="${CFG:-${VITRA_ROOT}/vitra/configs/human_llava_ov2_overfit.json}"
EPOCHS="${EPOCHS:-5}"

echo "=== Train VITRA-LLaVA-OV2 overfit (${EPOCHS} epochs) ==="
python "${VITRA_ROOT}/scripts/train_llava_ov2_overfit.py" \
  --config "$CFG" \
  --output_dir "$OUT" \
  --epochs "$EPOCHS" \
  --batch_size 1 \
  "$@"

CKPT="${OUT}/vitra_llava_ov2_best.pt"
[[ -f "$CKPT" ]] || CKPT="${OUT}/vitra_llava_ov2_last.pt"

echo "=== Eval + visualize: $CKPT ==="
python "${VITRA_ROOT}/scripts/eval_llava_ov2_overfit.py" \
  --checkpoint "$CKPT" \
  --config "$CFG" \
  --output_dir "$EVAL_OUT"

echo "Done. ckpt=$CKPT  metrics=${EVAL_OUT}/metrics.json  viz=${EVAL_OUT}/*.mp4"
