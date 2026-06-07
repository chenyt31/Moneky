#!/usr/bin/env bash
# Activate the project-local uv venv (Moneky/.venv).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_DISABLE_PROMPT_FOR_TRUST_REMOTE_CODE=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
