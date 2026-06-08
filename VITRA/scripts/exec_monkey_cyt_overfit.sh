#!/usr/bin/env bash
# Launch VITRA overfit on monkey-cyt (分布式训练空间) via qzcli exec.
set -euo pipefail

REPO="/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/yangtao/Moneky"
PYTHON="${PYTHON:-/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/conda/envs/qzcli/bin/python}"
exec "$PYTHON" "${REPO}/VITRA/scripts/launch_monkey_cyt_overfit.py" "$@"
