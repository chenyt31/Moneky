#!/usr/bin/env bash
# Launch memorize overfit on monkey-cyt (head-only, action MAE target).
set -euo pipefail

REPO="/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/yangtao/Moneky"
PYTHON="${PYTHON:-/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/conda/envs/qzcli/bin/python}"
OUT="${OUT:-${REPO}/VITRA/outputs/monkey_cyt_memorize}"
CFG="${CFG:-${REPO}/VITRA/vitra/configs/human_llava_ov2_memorize.json}"
EPOCHS="${EPOCHS:-80}"
RUNNER="${REPO}/VITRA/scripts/run_llava_ov2_memorize.sh"

exec env OUT="$OUT" EVAL_OUT="${OUT}/eval" CFG="$CFG" EPOCHS="$EPOCHS" RUNNER="$RUNNER" \
  "$PYTHON" "${REPO}/VITRA/scripts/launch_monkey_cyt_overfit.py" \
  --out "$OUT" --runner "$RUNNER"
