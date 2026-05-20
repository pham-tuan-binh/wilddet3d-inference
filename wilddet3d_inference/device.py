"""Device autodetection and Mac-friendly torch env setup.

Preference order: ``cuda`` (best raw throughput) > ``mps`` (Apple Silicon
GPU) > ``cpu``. ``mps`` is preferred over ``cpu`` on any Apple Silicon
Mac, but only when the user hasn't explicitly opted out.

We also flip ``PYTORCH_ENABLE_MPS_FALLBACK=1`` at import time. Several
ops in WildDet3D (and SAM3) don't yet have MPS kernels — without the
fallback flag they crash; with it they silently fall back to CPU for
that single op, which is the right default for "make it work".
"""

from __future__ import annotations

import os


def setup_environment() -> None:
    """Set torch env vars that improve Mac inference. Idempotent."""
    # MPS doesn't have every op; let unsupported ops fall back to CPU.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Avoid Tokenizers fork warnings spamming stderr during demos.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Activation checkpointing is a training memory-saving trick that
    # adds wrapper overhead in inference. The SAM3 backbone reads this
    # at import time (third_party/sam3/sam3/model_builder.py:74).
    os.environ.setdefault("SAM3_DISABLE_ACT_CKPT", "1")


def pick_device(prefer: str | None = None) -> str:
    """Choose the best available device.

    Args:
        prefer: "cuda", "mps", "cpu", or None for autodetect. If a
            preference is given but unavailable, fall back to the next
            best option (with a printed note).

    Returns:
        One of "cuda", "mps", "cpu".
    """
    import torch

    have_cuda = torch.cuda.is_available()
    have_mps = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )

    if prefer:
        prefer = prefer.lower()
        if prefer == "cuda" and have_cuda:
            return "cuda"
        if prefer == "mps" and have_mps:
            return "mps"
        if prefer == "cpu":
            return "cpu"
        # Unsupported preference: fall through to autodetect with a note.
        print(
            f"[wilddet3d-inference] requested device={prefer!r} not "
            f"available; falling back to autodetect."
        )

    if have_cuda:
        return "cuda"
    if have_mps:
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    """Return a human-readable label for the chosen device."""
    import torch

    if device == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"CUDA: {name} ({mem:.1f} GB)"
        except Exception:
            return "CUDA"
    if device == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"
