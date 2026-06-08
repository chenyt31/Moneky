#!/usr/bin/env bash
# History-video overfit pipeline: eval before -> train -> eval after + mesh viz.
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

OUT="${OUT:-${VITRA_ROOT}/outputs/vitra_llava_ov2_overfit_hist}"
EVAL_BEFORE="${EVAL_BEFORE:-${VITRA_ROOT}/outputs/vitra_llava_ov2_eval_before}"
EVAL_AFTER="${EVAL_AFTER:-${VITRA_ROOT}/outputs/vitra_llava_ov2_eval_after}"
CFG="${CFG:-${VITRA_ROOT}/vitra/configs/human_llava_ov2_overfit.json}"
EPOCHS="${EPOCHS:-10}"
NUM_EVAL="${NUM_EVAL:-32}"
MAX_VIZ="${MAX_VIZ:-3}"

echo "=== [1/3] Eval BEFORE training (fresh init head) ==="
python "${VITRA_ROOT}/scripts/eval_llava_ov2_overfit.py" \
  --fresh_init \
  --config "$CFG" \
  --output_dir "$EVAL_BEFORE" \
  --num_eval "$NUM_EVAL" \
  --max_pred_viz "$MAX_VIZ" \
  --tag before_training

echo "=== [2/3] Train overfit (${EPOCHS} epochs) -> ${OUT} ==="
python "${VITRA_ROOT}/scripts/train_llava_ov2_overfit.py" \
  --config "$CFG" \
  --output_dir "$OUT" \
  --epochs "$EPOCHS" \
  --batch_size 1

CKPT="${OUT}/vitra_llava_ov2_best.pt"
[[ -f "$CKPT" ]] || CKPT="${OUT}/vitra_llava_ov2_last.pt"

echo "=== [3/3] Eval AFTER training + mesh viz: $CKPT ==="
python "${VITRA_ROOT}/scripts/eval_llava_ov2_overfit.py" \
  --checkpoint "$CKPT" \
  --config "$CFG" \
  --output_dir "$EVAL_AFTER" \
  --num_eval "$NUM_EVAL" \
  --max_pred_viz "$MAX_VIZ" \
  --tag after_training

python - <<PY
import json
from pathlib import Path

before = json.loads(Path("${EVAL_BEFORE}/metrics.json").read_text())
after = json.loads(Path("${EVAL_AFTER}/metrics.json").read_text())
train_hist = json.loads(Path("${OUT}/train_history.json").read_text()) if Path("${OUT}/train_history.json").exists() else {}

summary = {
    "before_training": before,
    "after_training": after,
    "train_history": {
        "baseline_eval_loss": train_hist.get("baseline", {}).get("eval_loss"),
        "best_eval_loss": train_hist.get("best_eval_loss"),
        "final_eval_loss": train_hist.get("final", {}).get("eval_loss"),
    },
    "delta": {
        "action_mae_norm_mean": (before.get("action_mae_norm_mean") or 0) - (after.get("action_mae_norm_mean") or 0),
        "action_mae_unnorm_mean": (before.get("action_mae_unnorm_mean") or 0) - (after.get("action_mae_unnorm_mean") or 0),
    },
}
out = Path("${OUT}/overfit_compare.json")
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"Wrote comparison -> {out}")
PY

echo "Done."
echo "  before metrics: ${EVAL_BEFORE}/metrics.json"
echo "  after metrics:  ${EVAL_AFTER}/metrics.json"
echo "  mesh viz:       ${EVAL_AFTER}/*.mp4"
echo "  compare:        ${OUT}/overfit_compare.json"
