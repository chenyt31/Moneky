"""Hugging Face offline + headless settings for compute nodes without internet."""

from __future__ import annotations

import os

_PATCHED = False


def enable_hf_offline(*, trust_remote_code: bool = True) -> None:
    """Apply offline env vars and avoid interactive trust_remote_code prompts."""
    if trust_remote_code:
        os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"
        os.environ["HF_HUB_DISABLE_PROMPT_FOR_TRUST_REMOTE_CODE"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    if trust_remote_code:
        _patch_resolve_trust_remote_code()


def _patch_resolve_trust_remote_code() -> None:
    global _PATCHED
    if _PATCHED:
        return
    try:
        import transformers.dynamic_module_utils as dynamic_module_utils
    except ImportError:
        return

    original = dynamic_module_utils.resolve_trust_remote_code

    def resolve_trust_remote_code(trust_remote_code, *args, **kwargs):
        if trust_remote_code is None and os.environ.get("TRANSFORMERS_TRUST_REMOTE_CODE", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            trust_remote_code = True
        return original(trust_remote_code, *args, **kwargs)

    dynamic_module_utils.resolve_trust_remote_code = resolve_trust_remote_code
    _PATCHED = True
