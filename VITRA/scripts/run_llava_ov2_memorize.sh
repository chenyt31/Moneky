#!/usr/bin/env bash
# Memorize pipeline: head-only aggressive train until training-set sampled action MAE -> ~0.
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

OUT="${OUT:-${VITRA_ROOT}/outputs/vitra_llava_ov2_memorize}"
EVAL_OUT="${EVAL_OUT:-${OUT}/eval}"
CFG="${CFG:-${VITRA_ROOT}/vitra/configs/human_llava_ov2_memorize.json}"
EPOCHS="${EPOCHS:-80}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
LOG="${OUT}/run.log"

mkdir -p "$OUT" "$EVAL_OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== VITRA LLaVA-OV2 memorize (head-only, cfg=1 eval) ==="
echo "host=$(hostname) time=$(date -Is)"
echo "out=$OUT eval_out=$EVAL_OUT epochs=$EPOCHS cfg=$CFG max_samples=${MAX_SAMPLES:-all}"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo "=== [1/2] Memorize train ==="
TRAIN_ARGS=(
  --config "$CFG"
  --output_dir "$OUT"
  --epochs "$EPOCHS"
)
if [[ -n "$MAX_SAMPLES" ]]; then
  TRAIN_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [[ "${SKIP_PREWARM:-}" == "1" ]]; then
  TRAIN_ARGS+=(--skip_prewarm)
fi
python "${VITRA_ROOT}/scripts/train_llava_ov2_memorize.py" "${TRAIN_ARGS[@]}"

CKPT="${OUT}/vitra_llava_ov2_memorize_best.pt"
[[ -f "$CKPT" ]] || CKPT="${OUT}/vitra_llava_ov2_memorize_last.pt"

echo "=== [2/2] Memorize eval: $CKPT ==="
EVAL_ARGS=(
  --checkpoint "$CKPT"
  --config "$CFG"
  --output_dir "$EVAL_OUT"
)
if [[ -n "$MAX_SAMPLES" ]]; then
  EVAL_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
python "${VITRA_ROOT}/scripts/eval_llava_ov2_memorize.py" "${EVAL_ARGS[@]}"

VIZ_OUT="${EVAL_OUT}/viz"
MAX_PRED_VIZ="${MAX_PRED_VIZ:-8}"
PRED_INDICES="${PRED_INDICES:-0,1,2,3,4,5,6,7}"
echo "=== [3/3] Mesh viz (gt vs pred): $CKPT ==="
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
echo "ckpt=$CKPT"
echo "metrics=${EVAL_OUT}/memorize_metrics.json"
echo "viz=${VIZ_OUT}/pred_viz/*.mp4"
