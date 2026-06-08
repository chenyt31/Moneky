#!/usr/bin/env bash
# Train + eval VITRA LLaVA-OV2 overfit on a single GPU; tee logs into output dir.
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

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-${VITRA_ROOT}/outputs/vitra_llava_ov2_overfit_h100_${TS}}"
EVAL_OUT="${EVAL_OUT:-${OUT}/eval}"
CFG="${CFG:-${VITRA_ROOT}/vitra/configs/human_llava_ov2_overfit.json}"
EPOCHS="${EPOCHS:-5}"
LOG="${OUT}/run.log"

mkdir -p "$OUT" "$EVAL_OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== VITRA LLaVA-OV2 overfit (single GPU) ==="
echo "host=$(hostname) time=$(date -Is)"
echo "repo=$REPO_ROOT"
echo "out=$OUT eval_out=$EVAL_OUT epochs=$EPOCHS cfg=$CFG"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo "=== [1/2] Train ==="
python "${VITRA_ROOT}/scripts/train_llava_ov2_overfit.py" \
  --config "$CFG" \
  --output_dir "$OUT" \
  --epochs "$EPOCHS" \
  --batch_size 1 \
  --num_workers 0

CKPT="${OUT}/vitra_llava_ov2_best.pt"
[[ -f "$CKPT" ]] || CKPT="${OUT}/vitra_llava_ov2_last.pt"

echo "=== [2/2] Eval + visualize: $CKPT ==="
python "${VITRA_ROOT}/scripts/eval_llava_ov2_overfit.py" \
  --checkpoint "$CKPT" \
  --config "$CFG" \
  --output_dir "$EVAL_OUT"

echo "=== Done ==="
echo "log=$LOG"
echo "ckpt=$CKPT"
echo "metrics=${EVAL_OUT}/metrics.json"
