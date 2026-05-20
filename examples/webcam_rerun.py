"""Webcam → WildDet3D → rerun viewer.

Live text-prompt 3D detection visualised in three synchronised panels:

  * 2D camera view with predicted 2D boxes overlaid.
  * 2D depth heatmap (predicted metric depth in metres).
  * 3D scene: predicted 3D bounding box wireframes (built from
    vis4d's own corner extractor) + an RGB-colored point cloud
    unprojected from the predicted depth. The 3D view explicitly
    *excludes* the rerun-auto-generated depth point cloud so the
    only points in the scene are RGB-colored.

Usage::

    # spawn rerun viewer with default classes
    uv run python examples/webcam_rerun.py

    # comma-separated classes, optional intrinsics, save offline
    uv run python examples/webcam_rerun.py \\
        --text "laptop,keyboard,scissors,robot arm" \\
        --intrinsics-yaml /path/cam.yaml \\
        --save /tmp/session.rrd

    uv run rerun /tmp/session.rrd       # open later

Press Ctrl-C in the terminal to quit.

Latency: a 1.2 B-param model running locally is not real-time. The
rerun viewer keeps the last result on screen between inferences so
the experience is closer to "slideshow with depth" than "AR overlay".
On Apple Silicon expect ~10-15 s/frame at the default trained
resolution (1008²) and ~3-5 s/frame at ``--resolution 672``.
"""

from __future__ import annotations

# Set MPS-friendly env vars *before* ``import torch`` so the fallback
# path is active even if torch gets pulled in before
# ``wilddet3d_inference.setup_environment`` runs.
import os as _os

_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_os.environ.setdefault("SAM3_DISABLE_ACT_CKPT", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import signal
import sys
import time
from typing import Optional

import cv2
import numpy as np
import torch

from wilddet3d_inference import Detector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PALETTE = [
    (231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166),
    (52, 73, 94), (211, 84, 0),
]


def _color_for_class(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


def _unproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    rgb: np.ndarray,
    stride: int = 6,
    max_depth_m: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift an (H, W) metric depth map to a colored point cloud.

    ``depth`` is assumed to already be aligned with ``rgb`` (the
    library does the model-space → original-image crop+resize).
    """
    h, w = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    zs = depth[ys, xs]
    mask = np.isfinite(zs) & (zs > 1e-3) & (zs < max_depth_m)
    zs, xs, ys = zs[mask], xs[mask], ys[mask]

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    pts = np.stack([X, Y, zs], axis=-1).astype(np.float32)
    colors = rgb[ys, xs].astype(np.uint8)
    return pts, colors


def _open_capture(index: int, width: int, height: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}. On macOS check "
            "System Settings → Privacy & Security → Camera and "
            "grant your terminal access."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _grab_rgb(cap) -> Optional[np.ndarray]:
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    import rerun as rr
    import rerun.blueprint as rrb
    from vis4d.data.const import AxisMode
    from vis4d.op.box.box3d import boxes3d_to_corners

    # ---- rerun setup ----
    rr.init("wilddet3d-rerun", spawn=args.save is None)
    if args.save:
        rr.save(args.save)

    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "world/camera",
        rr.Transform3D(translation=[0, 0, 0]),
        static=True,
    )
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(
                        name="Camera (RGB + 2D boxes)",
                        origin="/world/camera/rgb",
                    ),
                    rrb.Spatial2DView(
                        name="Predicted depth (m)",
                        origin="/world/camera/depth",
                    ),
                    row_shares=[2, 1],
                ),
                rrb.Spatial3DView(
                    name="3D scene (boxes + RGB-colored point cloud)",
                    origin="/world",
                    # Exclude the DepthImage entity so rerun doesn't
                    # also render it as a heat-colored point cloud.
                    contents=["+ /world/**", "- /world/camera/depth/**"],
                ),
                column_shares=[1, 1.6],
            ),
            collapse_panels=True,
        )
    )

    # ---- detector ----
    det = Detector(
        device=args.device,
        dtype=args.dtype,
        input_size=args.resolution,
        intrinsics=args.intrinsics_yaml,
        warmup_steps=1,
        warmup_hw=(args.height, args.width),
    )
    print(
        f"[rerun] detector ready: device={det.device} "
        f"dtype={det.dtype}"
    )

    classes = [c.strip() for c in args.text.split(",") if c.strip()]
    print(f"[rerun] classes: {classes}")

    # ---- capture ----
    cap = _open_capture(args.camera_index, args.width, args.height)
    print(
        f"[rerun] camera opened "
        f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})"
    )

    # ---- loop ----
    stop = {"flag": False}

    def _sigint(_s, _f):
        stop["flag"] = True
        print("\n[rerun] interrupted; shutting down…")

    signal.signal(signal.SIGINT, _sigint)

    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    frame_idx = -1
    infer_idx = 0
    try:
        while not stop["flag"]:
            frame_idx += 1
            rgb = _grab_rgb(cap)
            if rgb is None:
                time.sleep(0.05)
                continue

            rr.set_time("frame", sequence=frame_idx)
            rr.log("world/camera/rgb", rr.Image(rgb))

            if frame_idx % args.every_n != 0:
                continue

            t0 = time.perf_counter()
            try:
                result = det.detect(rgb, classes)
            except Exception as e:  # noqa: BLE001
                rr.log("errors", rr.TextLog(f"{type(e).__name__}: {e}"))
                print(f"[rerun] inference error: {e}")
                continue
            dt = time.perf_counter() - t0
            infer_idx += 1

            # Pinhole at the runtime K (whatever the library resolved).
            K = result.intrinsics_original
            h, w = rgb.shape[:2]
            pinhole = rr.Pinhole(image_from_camera=K, width=w, height=h,
                                  camera_xyz=rr.ViewCoordinates.RDF)
            rr.log("world/camera/rgb", pinhole)
            rr.log("world/camera/depth", pinhole)

            # 2D + 3D boxes
            if result.detections:
                # 3D wireframes via vis4d's own corner extractor.
                boxes_t = torch.tensor(
                    np.stack([d.box3d for d in result.detections]),
                    dtype=torch.float32,
                )
                corners = boxes3d_to_corners(boxes_t, AxisMode.OPENCV).numpy()

                strips = []
                strip_colors = []
                labels = []
                for i, d in enumerate(result.detections):
                    col = _color_for_class(d.class_id)
                    labels.append(
                        f"{d.class_name or d.class_id}  {d.score:.2f}"
                    )
                    for a, b in edges:
                        strips.append(
                            np.stack([corners[i, a], corners[i, b]],
                                     axis=0).astype(np.float32)
                        )
                        strip_colors.append(col)

                rr.log(
                    "world/camera/boxes_3d",
                    rr.LineStrips3D(
                        strips,
                        colors=np.array(strip_colors, dtype=np.uint8),
                        radii=0.003,
                    ),
                )

                # 2D overlay.
                boxes2d = np.stack(
                    [d.box2d for d in result.detections], axis=0
                ).astype(np.float32)
                colors2d = np.array(
                    [_color_for_class(d.class_id)
                     for d in result.detections], dtype=np.uint8
                )
                rr.log(
                    "world/camera/rgb/boxes_2d",
                    rr.Boxes2D(
                        array=boxes2d,
                        array_format=rr.Box2DFormat.XYXY,
                        labels=labels,
                        colors=colors2d,
                    ),
                )
            else:
                rr.log("world/camera/boxes_3d", rr.Clear(recursive=False))
                rr.log(
                    "world/camera/rgb/boxes_2d", rr.Clear(recursive=False)
                )

            # Depth heatmap (2D panel) + RGB-colored 3D point cloud.
            if result.depth_map is not None:
                rr.log(
                    "world/camera/depth",
                    rr.DepthImage(result.depth_map, meter=1.0),
                )
                pts, colors = _unproject_depth(
                    result.depth_map,
                    K,
                    rgb,
                    stride=args.point_cloud_stride,
                    max_depth_m=args.max_depth_m,
                )
                if len(pts) > 0:
                    rr.log(
                        "world/points",
                        rr.Points3D(pts, colors=colors, radii=0.002),
                    )

            top = ", ".join(
                f"{d.class_name or d.class_id}={d.score:.2f}"
                for d in result.detections[:5]
            )
            rr.log(
                "stats",
                rr.TextLog(
                    f"frame {frame_idx} | inference #{infer_idx} "
                    f"in {dt * 1000:.0f} ms | "
                    f"{len(result.detections)} det | {top}"
                ),
            )

            if args.max_frames is not None and infer_idx >= args.max_frames:
                print(
                    f"[rerun] reached --max-frames={args.max_frames}, "
                    "exiting."
                )
                break
    finally:
        cap.release()


def main() -> None:
    p = argparse.ArgumentParser(
        prog="examples/rerun.py",
        description=(
            "Webcam → WildDet3D text-prompt detection → rerun viewer."
        ),
    )
    p.add_argument(
        "--text",
        default="laptop,keyboard,monitor,chair",
        help='Comma-separated class names. Default "laptop,keyboard,...".',
    )
    p.add_argument(
        "--camera-index", type=int, default=0,
        help="OpenCV camera device index. macOS will prompt for access.",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--device", choices=("cuda", "mps", "cpu"), default=None,
        help="Force a device. Default: autodetect.",
    )
    p.add_argument(
        "--dtype", choices=("auto", "fp16", "fp32"), default="auto",
    )
    p.add_argument(
        "--resolution", type=int, default=None,
        help=(
            "Override SAM3 input size (multiple of 14, default 1008). "
            "Lower trades depth/box alignment for speed."
        ),
    )
    p.add_argument(
        "--every-n", type=int, default=1,
        help="Run inference on every Nth frame.",
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after this many inferences. Default: run forever.",
    )
    p.add_argument(
        "--point-cloud-stride", type=int, default=6,
        help="Subsample stride for the depth point cloud.",
    )
    p.add_argument(
        "--max-depth-m", type=float, default=12.0,
        help="Clip points beyond this distance.",
    )
    p.add_argument(
        "--intrinsics-yaml", type=str, default=None,
        help=(
            "Path to a YAML with camera intrinsics (any common shape). "
            "Default: use placeholder K — recommended for in-the-wild."
        ),
    )
    p.add_argument(
        "--save", type=str, default=None,
        help=(
            "Save an .rrd recording instead of spawning the viewer. "
            "Open later with `uv run rerun <file.rrd>`."
        ),
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
