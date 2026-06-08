#!/usr/bin/env bash
# Launch full-dataset + full-parameter overfit on monkey-cyt.
set -euo pipefail

REPO="/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/yangtao/Moneky"
PYTHON="${PYTHON:-/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/conda/envs/qzcli/bin/python}"
OUT="${OUT:-${REPO}/VITRA/outputs/monkey_cyt_overfit_full}"
CFG="${CFG:-${REPO}/VITRA/vitra/configs/human_llava_ov2_overfit_full.json}"
EPOCHS="${EPOCHS:-10}"
RUNNER="${REPO}/VITRA/scripts/run_llava_ov2_overfit_full_qz.sh"

exec env OUT="$OUT" EVAL_OUT="${OUT}/eval" CFG="$CFG" EPOCHS="$EPOCHS" NUM_EVAL=0 MAX_PRED_VIZ=0 RUNNER="$RUNNER" \
  "$PYTHON" "${REPO}/VITRA/scripts/launch_monkey_cyt_overfit.py" \
  --out "$OUT" --runner "$RUNNER"
