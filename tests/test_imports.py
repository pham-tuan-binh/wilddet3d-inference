"""Smoke tests that don't need the model weights.

Verifies:

* ``wilddet3d_inference`` imports cleanly without ``vis4d_cuda_ops``
  installed (the whole point of the wrapper).
* The CUDA-op shim raises ``NotImplementedError`` if invoked.
* ``pick_device`` returns a string in the expected set.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def test_import_installs_shim():
    # Make sure no real vis4d_cuda_ops is left from a previous test.
    sys.modules.pop("vis4d_cuda_ops", None)

    import wilddet3d_inference  # noqa: F401

    assert "vis4d_cuda_ops" in sys.modules


def test_shim_raises_on_call():
    importlib.import_module("wilddet3d_inference")
    mod = sys.modules["vis4d_cuda_ops"]
    with pytest.raises(NotImplementedError):
        mod.iou_box3d(None, None)


def test_pick_device_returns_known():
    from wilddet3d_inference.device import pick_device

    d = pick_device()
    assert d in {"cuda", "mps", "cpu"}
