"""Hugging Face authentication helpers for gated datasets."""

from __future__ import annotations

import os
from typing import Optional


def resolve_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    """Prefer explicit token, then HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars."""
    if explicit:
        return explicit.strip()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    return None


def ensure_hf_login(token: Optional[str] = None, *, soft: bool = False) -> Optional[str]:
    """Log into Hugging Face Hub if a token is available.

    Returns the token used (or None). Raises if login is required and missing
    when soft=False and no cached credentials exist.
    """
    tok = resolve_hf_token(token)
    try:
        from huggingface_hub import login, whoami
        try:
            from huggingface_hub import get_token
        except ImportError:
            get_token = None  # type: ignore
        try:
            from huggingface_hub import HfFolder
        except ImportError:
            HfFolder = None  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required. Run: pip install huggingface_hub datasets"
        ) from e

    if tok:
        login(token=tok, add_to_git_credential=False)
        try:
            info = whoami(token=tok)
            name = info.get("name") or info.get("fullname") or "unknown"
            print(f"Hugging Face login OK as: {name}")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: token set but whoami failed: {e}")
        return tok

    # Fall back to cached token from `huggingface-cli login`
    cached = None
    if get_token is not None:
        try:
            cached = get_token()
        except Exception:
            cached = None
    if not cached and HfFolder is not None:
        try:
            cached = HfFolder.get_token()
        except Exception:
            cached = None

    if cached:
        try:
            info = whoami()
            name = info.get("name") or info.get("fullname") or "unknown"
            print(f"Using cached Hugging Face credentials as: {name}")
        except Exception:
            print("Using cached Hugging Face credentials.")
        return cached

    msg = (
        "No Hugging Face token found.\n"
        "  1) Create a token: https://huggingface.co/settings/tokens\n"
        "  2) Accept dataset terms: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations\n"
        "  3) Then either:\n"
        "       huggingface-cli login\n"
        "     or in PowerShell:\n"
        "       $env:HF_TOKEN = 'hf_...'\n"
        "       python scripts/01_prepare_data.py\n"
    )
    if soft:
        print(msg)
        return None
    raise RuntimeError(msg)
