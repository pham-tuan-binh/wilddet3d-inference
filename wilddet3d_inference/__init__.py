"""wilddet3d-inference: device-agnostic local inference for WildDet3D.

Import order matters: this module installs a sys.modules shim for
``vis4d_cuda_ops`` and configures torch device flags BEFORE importing
the upstream ``wilddet3d`` package. Always do::

    import wilddet3d_inference  # before any `import wilddet3d`

The shim only raises if the (eval-only) 3D-IoU kernel is actually called,
so monocular inference imports succeed on CPU/MPS machines without CUDA.
"""

from .patches import (
    ensure_wilddet3d_on_path,
    install_torch_cuda_stream_noop,
    install_triton_shim,
)
from .device import pick_device, setup_environment

# Side effects (idempotent): run before upstream imports.
setup_environment()
# NOTE: we do NOT shim vis4d_cuda_ops. vis4d's `package_available()`
# uses importlib.util.find_spec, which crashes on a sys.modules entry
# whose __spec__ is None. Letting the genuine ImportError surface
# (the package isn't installed on Mac) makes vis4d use its non-CUDA
# fallbacks. wilddet3d's own usage of vis4d_cuda_ops is confined to
# `wilddet3d/ops/box3d.py` and `iou_box3d.py`, which are only loaded
# by the eval pipeline — never by the inference forward pass.
install_triton_shim()
install_torch_cuda_stream_noop()
ensure_wilddet3d_on_path()

from .inference import (  # noqa: E402
    Detection,
    DetectionResult,
    Detector,
    build_detector,
)

__all__ = [
    "Detector",
    "Detection",
    "DetectionResult",
    "build_detector",
    "pick_device",
    "setup_environment",
]
__version__ = "0.1.0"
