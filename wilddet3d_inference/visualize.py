"""Thin wrapper around ``wilddet3d.vis.visualize.draw_3d_boxes``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .inference import DetectionResult


def render_result(
    result: DetectionResult,
    *,
    save_path: Optional[str | Path] = None,
    class_names: Optional[list[str]] = None,
    score_2d_threshold: float = 0.3,
    score_3d_threshold: float = 0.1,
) -> np.ndarray:
    """Render 3D boxes on the original image.

    Returns the rendered RGB image as a uint8 numpy array; also saves it
    to ``save_path`` if provided.
    """
    from wilddet3d.vis.visualize import draw_3d_boxes

    if not result.detections:
        out = result.original_image.copy()
        if save_path:
            from PIL import Image as _Image

            _Image.fromarray(out).save(save_path)
        return out

    boxes3d = np.stack([d.box3d for d in result.detections], axis=0)
    boxes2d = np.stack([d.box2d for d in result.detections], axis=0)
    s2 = np.array([d.score_2d for d in result.detections], dtype=np.float32)
    s3 = np.array([d.score_3d for d in result.detections], dtype=np.float32)
    cids = np.array(
        [d.class_id for d in result.detections], dtype=np.int64
    )

    if class_names is None:
        # Try to infer from detection names; else fall back to numeric.
        named = [d.class_name for d in result.detections]
        if all(n is not None for n in named):
            uniq: list[str] = []
            for n in named:
                if n not in uniq:
                    uniq.append(n)  # type: ignore[arg-type]
            class_names = uniq
            cids = np.array(
                [class_names.index(n) for n in named], dtype=np.int64
            )

    rendered = draw_3d_boxes(
        image=result.original_image,
        boxes3d=boxes3d,
        intrinsics=result.intrinsics_original,
        scores_2d=s2,
        scores_3d=s3,
        class_ids=cids,
        class_names=class_names,
        score_2d_threshold=score_2d_threshold,
        score_3d_threshold=score_3d_threshold,
        save_path=str(save_path) if save_path else None,
        boxes_2d=boxes2d,
        draw_predicted_2d_boxes=False,
    )
    return rendered
