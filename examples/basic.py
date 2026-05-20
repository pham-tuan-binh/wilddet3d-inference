"""Smallest possible usage example. Run after `pip install -e .[demo]`.

    python examples/basic.py path/to/image.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import wilddet3d_inference  # noqa: F401  (env + path patches before wilddet3d)
from wilddet3d_inference import build_detector
from wilddet3d_inference.visualize import render_result


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <image>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    detector = build_detector()  # autodetect device + dtype
    result = detector.detect_text(
        image_path, ["car", "person", "bicycle", "chair", "table"]
    )

    out = image_path.with_suffix(".det.png")
    render_result(result, save_path=out, class_names=None)
    print(f"saved {out}  ({len(result.detections)} detections)")


if __name__ == "__main__":
    main()
