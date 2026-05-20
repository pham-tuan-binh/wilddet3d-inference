"""Micro-benchmark for inference latency on the current device.

Usage::

    wilddet3d-benchmark image.jpg --text car,person --runs 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wilddet3d-benchmark",
        description="Time WildDet3D inference on the current device.",
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--text", default="car,person,chair,table")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--device", choices=("cuda", "mps", "cpu"), default=None
    )
    parser.add_argument(
        "--dtype", choices=("auto", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override SAM3 input size (multiple of 14, default 1008).",
    )
    parser.add_argument(
        "--depth-resolution",
        type=int,
        default=None,
        help=(
            "Override LingBot depth backbone input size (multiple of "
            "14). LingBot's ViT-L was the runtime bottleneck — at 336 "
            "it's ~3x faster than at 1008 with negligible depth loss."
        ),
    )
    args = parser.parse_args(argv)

    import wilddet3d_inference  # noqa: F401  (env + patches)
    import torch
    from wilddet3d_inference import build_detector

    detector = build_detector(
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        device=args.device,
        dtype=args.dtype,
        input_size=args.resolution,
        depth_input_size=args.depth_resolution,
    )
    classes = [c.strip() for c in args.text.split(",") if c.strip()]

    def _one() -> float:
        t0 = time.perf_counter()
        detector.detect_text(args.image, classes)
        if detector.device == "cuda":
            torch.cuda.synchronize()
        elif detector.device == "mps":
            torch.mps.synchronize()
        return time.perf_counter() - t0

    for _ in range(args.warmup):
        _one()

    timings = [_one() for _ in range(args.runs)]
    timings.sort()
    median = timings[len(timings) // 2]
    mean = sum(timings) / len(timings)

    print()
    print(f"device  : {detector.device}")
    print(f"dtype   : {detector.dtype}")
    print(f"runs    : {args.runs} (warmup {args.warmup})")
    print(
        f"latency : median {median*1000:.0f} ms  "
        f"mean {mean*1000:.0f} ms  "
        f"min {min(timings)*1000:.0f} ms  "
        f"max {max(timings)*1000:.0f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
