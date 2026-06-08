#!/usr/bin/env bash
# Launch memorize mesh viz on monkey-cyt (gt vs pred overlay videos).
set -euo pipefail

REPO="/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/yangtao/Moneky"
PYTHON="${PYTHON:-/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/conda/envs/qzcli/bin/python}"
OUT="${OUT:-${REPO}/VITRA/outputs/monkey_cyt_memorize}"
EVAL_OUT="${EVAL_OUT:-${OUT}/eval}"
VIZ_OUT="${VIZ_OUT:-${EVAL_OUT}/viz}"
CFG="${CFG:-${REPO}/VITRA/vitra/configs/human_llava_ov2_memorize.json}"
CKPT="${CKPT:-${OUT}/vitra_llava_ov2_memorize_best.pt}"
MAX_PRED_VIZ="${MAX_PRED_VIZ:-8}"
PRED_INDICES="${PRED_INDICES:-0,1,2,3,4,5,6,7}"
RUNNER="${REPO}/VITRA/scripts/run_llava_ov2_memorize_viz.sh"
MONITOR_LOG="${VIZ_OUT}/viz.log"
LAUNCH_LOG="${VIZ_OUT}/launch.log"

exec env \
  OUT="$OUT" \
  EVAL_OUT="$EVAL_OUT" \
  VIZ_OUT="$VIZ_OUT" \
  CFG="$CFG" \
  CKPT="$CKPT" \
  MAX_PRED_VIZ="$MAX_PRED_VIZ" \
  PRED_INDICES="$PRED_INDICES" \
  RUNNER="$RUNNER" \
  MONITOR_LOG="$MONITOR_LOG" \
  LAUNCH_LOG="$LAUNCH_LOG" \
  "$PYTHON" "${REPO}/VITRA/scripts/launch_monkey_cyt_overfit.py" \
  --out "$OUT" \
  --runner "$RUNNER" \
  --log "$MONITOR_LOG"
