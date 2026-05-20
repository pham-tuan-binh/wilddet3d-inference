"""Load camera intrinsics from YAML.

**Default format** — the one this repo's CLI flag assumes — matches
the rebot-pnp OpenCV-calibration shape:

    image_size:
    - 1280
    - 720
    camera_matrix:
    - - fx
      - 0.0
      - cx
    - - 0.0
      - fy
      - cy
    - - 0.0
      - 0.0
      - 1.0
    # dist_coeffs / reprojection_error_px / notes are ignored — the
    # model assumes pinhole optics and we don't undistort.

We also accept ``cv2.FileStorage`` dumps with
``camera_matrix.data: [9 floats]``, Kalibr ``cam0: { intrinsics:
[fx, fy, cx, cy] }``, plain ``{fx, fy, cx, cy}``, or a top-level
``K:`` 3×3. Whichever the file matches first wins.

If ``image_size`` / ``image_width``-``image_height`` is present in
the file and doesn't match the runtime image, K is rescaled so the
rays still correspond.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _coerce_3x3(data) -> Optional[np.ndarray]:
    """Try to turn ``data`` into a (3, 3) float array."""
    if data is None:
        return None
    arr = np.array(data, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (9,):
        return arr.reshape(3, 3)
    return None


def _from_fxfycxcy(d: dict) -> Optional[np.ndarray]:
    keys = {"fx", "fy", "cx", "cy"}
    if keys.issubset(d):
        return np.array(
            [
                [float(d["fx"]), 0.0, float(d["cx"])],
                [0.0, float(d["fy"]), float(d["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return None


def _from_intrinsics_list(d: dict) -> Optional[np.ndarray]:
    """Kalibr-style ``intrinsics: [fx, fy, cx, cy]``."""
    intr = d.get("intrinsics")
    if isinstance(intr, (list, tuple)) and len(intr) == 4:
        fx, fy, cx, cy = (float(v) for v in intr)
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    if isinstance(intr, (list, tuple)) and len(intr) == 9:
        return np.array(intr, dtype=np.float64).reshape(3, 3)
    return None


def _resolution(d: dict) -> Optional[tuple[int, int]]:
    for key in ("resolution", "image_size"):
        res = d.get(key)
        if isinstance(res, (list, tuple)) and len(res) == 2:
            return int(res[0]), int(res[1])
    if "image_width" in d and "image_height" in d:
        return int(d["image_width"]), int(d["image_height"])
    if "width" in d and "height" in d:
        return int(d["width"]), int(d["height"])
    return None


def _parse(d: dict) -> tuple[np.ndarray, Optional[tuple[int, int]]]:
    """Return ``(K, (calib_w, calib_h) or None)``. Raises ValueError."""
    def _try_dict(sub: dict) -> Optional[np.ndarray]:
        # Order matters: more specific keys first.
        K = _coerce_3x3(sub.get("K"))
        if K is not None:
            return K
        cm = sub.get("camera_matrix")
        if isinstance(cm, dict):
            K = _coerce_3x3(cm.get("data"))
        else:
            K = _coerce_3x3(cm)
        if K is not None:
            return K
        K = _from_intrinsics_list(sub)
        if K is not None:
            return K
        return _from_fxfycxcy(sub)

    # Drill into nested camera entries if present (Kalibr ``cam0:``).
    for nested_key in ("cam0", "cam1", "camera", "camera_0"):
        if isinstance(d.get(nested_key), dict):
            sub = d[nested_key]
            K = _try_dict(sub)
            if K is not None:
                return K, _resolution(sub) or _resolution(d)

    # Top-level layouts.
    K = _try_dict(d)
    if K is None:
        raise ValueError(
            "Could not find camera intrinsics in YAML. Expected one "
            "of: camera_matrix.data (OpenCV), intrinsics=[fx,fy,cx,cy] "
            "(Kalibr), {fx,fy,cx,cy}, or K as a 3×3 matrix."
        )
    return K, _resolution(d)


def load_intrinsics(
    path: str | Path,
    runtime_hw: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Load a camera K matrix from a YAML file.

    Args:
        path: Path to the YAML file.
        runtime_hw: Optional ``(H, W)`` of the image you'll actually
            run inference on. If the YAML reports a different
            resolution, ``K`` is rescaled (multiplied by H_runtime /
            H_calib in y and W_runtime / W_calib in x) so the rays
            still correspond. ``None`` skips this rescaling — use
            this when you know the file already matches.

    Returns:
        ``(3, 3)`` float32 K matrix ready to pass to ``preprocess`` /
        ``Detector.detect_text(intrinsics=...)``.
    """
    import yaml

    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    K, calib_wh = _parse(data)

    if runtime_hw is not None and calib_wh is not None:
        h_r, w_r = runtime_hw
        w_c, h_c = calib_wh  # (width, height) per most conventions
        if (w_c, h_c) != (w_r, h_r):
            sx = w_r / w_c
            sy = h_r / h_c
            K = K.copy()
            K[0, 0] *= sx  # fx
            K[1, 1] *= sy  # fy
            K[0, 2] *= sx  # cx
            K[1, 2] *= sy  # cy
            print(
                f"[intrinsics] rescaled K from "
                f"{w_c}×{h_c} -> {w_r}×{h_r}"
            )

    return K.astype(np.float32)
