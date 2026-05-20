"""Command-line interface: detect 3D boxes in an image.

Usage examples::

    wilddet3d-detect image.jpg --text "car,person,bicycle"
    wilddet3d-detect image.jpg --text car,person --out out.png
    wilddet3d-detect image.jpg --box 100,200,300,400 --label chair
    wilddet3d-detect image.jpg --point 150,250 --label cup

By default outputs a rendered PNG next to the input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _parse_box(s: str) -> list[float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--box expects 4 comma-separated numbers (xyxy), got {s!r}"
        )
    return parts


def _parse_point(s: str) -> tuple[float, float, int]:
    parts = s.split(",")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"--point expects 'x,y' or 'x,y,label' (label 0/1), got {s!r}"
        )
    x = float(parts[0])
    y = float(parts[1])
    label = int(parts[2]) if len(parts) == 3 else 1
    return (x, y, label)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wilddet3d-detect",
        description=(
            "Run WildDet3D on a local image. Output is a PNG with 3D "
            "wireframe boxes drawn on the input image."
        ),
    )
    parser.add_argument("image", type=Path, help="Path to an RGB image.")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help='Comma-separated class names (e.g. "car,person").',
    )
    parser.add_argument(
        "--box",
        type=_parse_box,
        default=None,
        help='2D box prompt in pixel xyxy (e.g. "100,200,300,400").',
    )
    parser.add_argument(
        "--point",
        type=_parse_point,
        action="append",
        default=None,
        help=(
            'Point prompt as "x,y" or "x,y,label" (label=1 positive, '
            "0 negative). May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("multi", "single"),
        default="single",
        help='For --box: "single" (one-to-one geometric, default) or '
        '"multi" (visual exemplar, one-to-many).',
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional category label for --box / --point prompts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a WildDet3D .pt checkpoint. Default: download from HF.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default=None,
        help="Force a device. Default: autodetect (cuda > mps > cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp16", "fp32"),
        default="auto",
        help='Weight dtype. Default "auto" = fp16 on GPU, fp32 on CPU.',
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "Override model input size (default 1008). Must be a "
            "multiple of 14. Try 672 on Mac MPS for ~3x speedup."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Default: <image>.det.png alongside input.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="2D score floor (text/visual prompts).",
    )
    parser.add_argument(
        "--score-3d-threshold",
        type=float,
        default=0.1,
        help="3D score floor (text/visual prompts).",
    )
    args = parser.parse_args(argv)

    prompts = sum(p is not None for p in (args.text, args.box, args.point))
    if prompts != 1:
        parser.error("Pick exactly one of --text / --box / --point.")

    # Import here so --help doesn't pay the torch import cost.
    import wilddet3d_inference  # noqa: F401  (apply env + path patches)
    from wilddet3d_inference import build_detector
    from wilddet3d_inference.visualize import render_result

    detector = build_detector(
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        device=args.device,
        dtype=args.dtype,
        score_threshold=args.score_threshold,
        score_3d_threshold=args.score_3d_threshold,
        input_size=args.resolution,
    )

    if args.text is not None:
        classes = [c.strip() for c in args.text.split(",") if c.strip()]
        result = detector.detect_text(args.image, classes)
    elif args.box is not None:
        if args.mode == "multi":
            result = detector.detect_box_multi(
                args.image, args.box, label=args.label
            )
        else:
            result = detector.detect_box_single(
                args.image, args.box, label=args.label
            )
    else:  # args.point
        result = detector.detect_point(
            args.image, args.point, label=args.label
        )

    out_path = args.out or args.image.with_suffix(".det.png")
    render_result(
        result,
        save_path=out_path,
        score_2d_threshold=args.score_threshold,
        score_3d_threshold=args.score_3d_threshold,
    )

    print(
        f"detections: {len(result.detections)}  ->  saved to {out_path}"
    )
    for d in result.detections[:20]:
        cx, cy, cz, w, h, l = d.box3d[:6]
        print(
            f"  cls={d.class_name or d.class_id}  "
            f"score={d.score:.2f}  "
            f"xyz=({cx:+.2f},{cy:+.2f},{cz:+.2f}) m  "
            f"whl=({w:.2f},{h:.2f},{l:.2f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
