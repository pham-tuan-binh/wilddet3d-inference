"""Library API for WildDet3D local inference.

Usage from another project::

    from wilddet3d_inference import Detector

    det = Detector()                                  # autodetect + warmup
    result = det.detect("frame.jpg", "laptop, chair")

    for d in result.detections:
        print(d.class_name, d.score, d.box3d)

    depth = result.depth_map      # (H, W) metres, aligned with the image
    K     = result.intrinsics_original  # (3, 3)

The same instance can be reused across many ``.detect`` calls and is
thread-safe for *serial* invocations (no internal queue / parallelism
contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from .device import describe_device, pick_device


_HF_REPO = "allenai/WildDet3D"
_HF_FILENAME = "wilddet3d_alldata_all_prompt_v1.0.pt"

# Type alias: an intrinsics spec can be a 3×3 array, a YAML path, or None.
IntrinsicsLike = Union[np.ndarray, str, Path, None]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """One detection. Coordinates are in the camera's 3D frame, metres."""

    box2d: np.ndarray  # (4,) xyxy pixels in original-image space
    box3d: np.ndarray  # (10,) cx, cy, cz, w, l, h, qw, qx, qy, qz  (Hamilton)
    score: float
    score_2d: float
    score_3d: float
    class_id: int
    class_name: Optional[str] = None


@dataclass
class DetectionResult:
    """Per-image result bundle. Everything is in original-image space.

    Attributes:
        detections: list of :class:`Detection`.
        depth_map: ``(H, W)`` float32 metres, pixel-aligned with
            ``original_image`` (model-space padding already removed).
            ``None`` if the model didn't emit depth.
        original_image: ``(H, W, 3)`` uint8 RGB — your input frame.
        intrinsics_original: ``(3, 3)`` float K, valid for both the
            RGB and the depth.
    """

    detections: list[Detection]
    depth_map: Optional[np.ndarray]
    original_image: np.ndarray
    intrinsics_original: np.ndarray


# ---------------------------------------------------------------------------
# Detector — the public API
# ---------------------------------------------------------------------------


class Detector:
    """Build a WildDet3D model once, run :meth:`detect` many times.

    Constructor builds the model (downloads the ~5 GB checkpoint from
    HF Hub on first use), applies all the runtime-compatibility
    patches, and runs ``warmup_steps`` dummy inferences to make later
    calls predictable. Pass ``warmup_steps=0`` to skip.

    Args:
        checkpoint: Path to a ``.pt`` checkpoint. ``None`` (default)
            downloads ``allenai/WildDet3D`` from HuggingFace.
        intrinsics: Default camera intrinsics applied when
            :meth:`detect` isn't given a per-call ``intrinsics=``.
            Accepts:

              * ``None`` (default) — use a placeholder K
                (focal = ``max(H, W)``, principal point centered).
                **Recommended** for in-the-wild use; the model was
                trained on the placeholder and learned its joint
                depth/box scale around it.
              * ``np.ndarray`` of shape ``(3, 3)`` — used as-is.
              * path-like — YAML file in any of the formats supported
                by :mod:`wilddet3d_inference.intrinsics`. Auto-rescaled
                to the runtime image resolution if needed.

        device: ``"cuda"`` / ``"mps"`` / ``"cpu"`` / ``None`` (auto).
        dtype: ``"fp32"`` / ``"fp16"`` / ``"auto"``. ``"auto"`` picks
            fp16 on CUDA, fp32 elsewhere.
        score_threshold: 2D combined-score floor for text prompts.
        score_3d_threshold: 3D score floor.
        canonical_rotation: Use the canonical-rotation head (matches
            the official checkpoint).
        use_depth_input: ``True`` only if you'll pass a measured
            ``depth=`` to :meth:`detect`.
        input_size: Override SAM3 input resolution (multiple of 14).
            Default 1008 = trained size, best alignment. 672 is
            ~3-5× faster on MPS but visibly shifts box positions.
        depth_input_size: Independent LingBot input size (multiple
            of 14). Off by default; experimental.
        warmup_steps: Run this many dummy forwards on a synthetic
            image at construction time, so the first real call isn't
            mysteriously slow. Default 1.
        warmup_hw: ``(H, W)`` of the dummy image used for warmup.
            Default 720×1280.

    Example::

        det = Detector(intrinsics="cam.yaml", input_size=672)
        result = det.detect("frame.png", ["laptop", "chair"])
        for d in result.detections:
            print(d.class_name, d.box3d[:3])
    """

    # ----- construction -----

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        *,
        intrinsics: IntrinsicsLike = None,
        device: Optional[str] = None,
        dtype: str = "auto",
        score_threshold: float = 0.3,
        score_3d_threshold: float = 0.1,
        canonical_rotation: bool = True,
        use_depth_input: bool = False,
        input_size: Optional[int] = None,
        depth_input_size: Optional[int] = None,
        warmup_steps: int = 1,
        warmup_hw: tuple[int, int] = (720, 1280),
    ):
        from wilddet3d import build_model

        # Resolve device + dtype strategy.
        device = pick_device(device)
        print(f"[wilddet3d-inference] device = {describe_device(device)}")
        if dtype == "auto":
            dtype_real = "fp16" if device == "cuda" else "fp32"
        else:
            dtype_real = dtype.lower()

        # Resolve checkpoint (HF download on demand).
        if checkpoint is None:
            print("[wilddet3d-inference] downloading checkpoint from HF…")
            checkpoint = _download_default_checkpoint()

        # Apply runtime patches AFTER the upstream module is importable.
        from .patches import apply_runtime_patches

        apply_runtime_patches()

        # Optional preprocess resolution override (multiple of 14).
        if input_size is not None and input_size != 1008:
            if input_size % 14 != 0:
                raise ValueError(
                    f"input_size={input_size} must be a multiple of 14 "
                    "(SAM3 ViT patch size)."
                )
            from wilddet3d import preprocessing as _pp

            _pp.IMAGE_SIZE = (input_size, input_size)
            print(
                f"[wilddet3d-inference] preprocess target = "
                f"{input_size}×{input_size}"
            )

        # Build the actual model.
        model = build_model(
            checkpoint=checkpoint,
            score_threshold=score_threshold,
            score_3d_threshold=score_3d_threshold,
            device=device,
            skip_pretrained=True,
            canonical_rotation=canonical_rotation,
            use_depth_input_test=use_depth_input,
        )

        if input_size is not None and input_size != 1008:
            from .patches import resize_sam3_vit

            resize_sam3_vit(model, input_size)

        if depth_input_size is not None:
            from .patches import downsample_lingbot_input

            downsample_lingbot_input(model, depth_input_size)

        # Cast / autocast strategy.
        autocast_dtype = None
        if dtype_real == "fp16":
            if device == "cuda":
                model = model.half()
                print("[wilddet3d-inference] casted model to fp16 (CUDA)")
            elif device == "mps":
                autocast_dtype = torch.float16
                print(
                    "[wilddet3d-inference] fp16 strategy on MPS: "
                    "autocast (weights stay fp32, ops cast per-call)"
                )
            else:
                print(
                    "[wilddet3d-inference] fp16 ignored on CPU; "
                    "using fp32."
                )
                dtype_real = "fp32"

        # Stash everything on the instance.
        self.model = model
        self.device = device
        self.dtype = dtype_real
        self.use_depth_input = use_depth_input
        self.autocast_dtype = autocast_dtype
        # Default intrinsics: kept as-given so we can re-rescale per
        # frame if the user provided a YAML path.
        self._intrinsics_default = intrinsics

        if warmup_steps > 0:
            self.warmup(steps=warmup_steps, hw=warmup_hw)

    # ----- warmup -----

    def warmup(
        self,
        steps: int = 1,
        hw: tuple[int, int] = (720, 1280),
        prompt: str = "object",
    ) -> None:
        """Run ``steps`` dummy forwards on a synthetic frame.

        Useful for taking the first-call latency hit off the critical
        path. Idempotent; safe to call multiple times.
        """
        import time as _time

        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 255, (hw[0], hw[1], 3), dtype=np.uint8)
        print(
            f"[wilddet3d-inference] warming up "
            f"({steps} steps, {hw[1]}×{hw[0]} dummy frame)…"
        )
        for i in range(steps):
            t0 = _time.perf_counter()
            self.detect(rgb, prompt)
            print(
                f"  warmup {i + 1}/{steps}  "
                f"{(_time.perf_counter() - t0) * 1000:.0f} ms"
            )

    # ----- inference -----

    def detect(
        self,
        image: Union[np.ndarray, str, Path, Image.Image],
        prompt: Union[str, Iterable[str]],
        *,
        intrinsics: IntrinsicsLike = None,
        depth: Optional[np.ndarray] = None,
    ) -> DetectionResult:
        """Detect objects matching ``prompt`` in ``image``.

        Args:
            image: a numpy ``(H, W, 3)`` uint8 RGB array, a path to an
                image file, or a ``PIL.Image``.
            prompt: a comma-separated string like ``"laptop, chair"``
                or an iterable of class names.
            intrinsics: override the default intrinsics for *this*
                call only. Accepts the same shapes as the constructor
                arg (``np.ndarray`` / path / ``None``).
            depth: optional measured depth map ``(H, W)`` metres,
                same resolution as ``image``. Only used if the
                detector was built with ``use_depth_input=True``.

        Returns:
            :class:`DetectionResult`.
        """
        classes = _parse_prompt(prompt)
        return self._run(
            image=image,
            intrinsics=intrinsics,
            depth=depth,
            class_names=classes,
            kwargs={"input_texts": classes},
        )

    # The four prompt-mode helpers are kept for users who need them.

    def detect_text(
        self,
        image,
        classes: Sequence[str],
        *,
        intrinsics: IntrinsicsLike = None,
        depth: Optional[np.ndarray] = None,
    ) -> DetectionResult:
        return self.detect(image, list(classes), intrinsics=intrinsics, depth=depth)

    def detect_box_multi(
        self,
        image,
        box: Sequence[float],
        *,
        intrinsics: IntrinsicsLike = None,
        depth: Optional[np.ndarray] = None,
        label: Optional[str] = None,
    ) -> DetectionResult:
        prompt = f"visual: {label}" if label else "visual"
        return self._run(
            image=image,
            intrinsics=intrinsics,
            depth=depth,
            class_names=[label] if label else ["exemplar"],
            kwargs={"input_boxes": [list(box)], "prompt_text": prompt},
        )

    def detect_box_single(
        self,
        image,
        box: Sequence[float],
        *,
        intrinsics: IntrinsicsLike = None,
        depth: Optional[np.ndarray] = None,
        label: Optional[str] = None,
    ) -> DetectionResult:
        prompt = f"geometric: {label}" if label else "geometric"
        return self._run(
            image=image,
            intrinsics=intrinsics,
            depth=depth,
            class_names=[label] if label else ["box"],
            kwargs={"input_boxes": [list(box)], "prompt_text": prompt},
        )

    def detect_point(
        self,
        image,
        points: Sequence[tuple[float, float, int]],
        *,
        intrinsics: IntrinsicsLike = None,
        depth: Optional[np.ndarray] = None,
        label: Optional[str] = None,
    ) -> DetectionResult:
        prompt = f"geometric: {label}" if label else "geometric"
        return self._run(
            image=image,
            intrinsics=intrinsics,
            depth=depth,
            class_names=[label] if label else ["point"],
            kwargs={"input_points": [list(points)], "prompt_text": prompt},
        )

    # ---------- internals ----------

    def _resolve_intrinsics(
        self,
        spec: IntrinsicsLike,
        runtime_hw: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Turn whatever intrinsics-spec the caller gave us into a K
        matrix matching ``runtime_hw``, or ``None`` for placeholder K.
        """
        if spec is None:
            spec = self._intrinsics_default
        if spec is None:
            return None
        if isinstance(spec, np.ndarray):
            return spec.astype(np.float32)
        # Treat as path-like — defer the import for low-priority deps.
        from .intrinsics import load_intrinsics

        return load_intrinsics(spec, runtime_hw=runtime_hw)

    def _run(
        self,
        *,
        image,
        intrinsics: IntrinsicsLike,
        depth: Optional[np.ndarray],
        class_names: list[str],
        kwargs: dict,
    ) -> DetectionResult:
        from wilddet3d import preprocess

        img_np = _load_image(image)
        K = self._resolve_intrinsics(intrinsics, runtime_hw=img_np.shape[:2])

        data = preprocess(
            img_np.astype(np.float32),
            K,
            depth=depth,
        )

        device = self.device
        if self.autocast_dtype is not None:
            dtype_t = torch.float32
        else:
            dtype_t = torch.float16 if self.dtype == "fp16" else torch.float32

        images_t = data["images"].to(device=device, dtype=dtype_t)
        intr_t = data["intrinsics"].to(device=device, dtype=torch.float32)[
            None
        ]
        call_kwargs = dict(
            images=images_t,
            intrinsics=intr_t,
            input_hw=[data["input_hw"]],
            original_hw=[data["original_hw"]],
            padding=[data["padding"]],
            **kwargs,
        )
        if depth is not None and self.use_depth_input:
            call_kwargs["depth_gt"] = data["depth_gt"].to(
                device=device, dtype=dtype_t
            )

        with torch.inference_mode():
            if self.autocast_dtype is not None:
                with torch.autocast(
                    device_type=self.device, dtype=self.autocast_dtype
                ):
                    results = self.model(**call_kwargs)
            else:
                results = self.model(**call_kwargs)

        (
            boxes,
            boxes3d,
            scores,
            scores_2d,
            scores_3d,
            class_ids,
            depth_maps,
        ) = results

        detections: list[Detection] = []
        b2d = boxes[0].detach().cpu().float().numpy()
        b3d = boxes3d[0].detach().cpu().float().numpy()
        s = scores[0].detach().cpu().float().numpy()
        s2 = scores_2d[0].detach().cpu().float().numpy()
        s3 = scores_3d[0].detach().cpu().float().numpy()
        cid = class_ids[0].detach().cpu().long().numpy()

        for i in range(len(b2d)):
            class_idx = int(cid[i])
            name = (
                class_names[class_idx]
                if 0 <= class_idx < len(class_names)
                else None
            )
            detections.append(
                Detection(
                    box2d=b2d[i],
                    box3d=b3d[i],
                    score=float(s[i]),
                    score_2d=float(s2[i]),
                    score_3d=float(s3[i]),
                    class_id=class_idx,
                    class_name=name,
                )
            )

        depth_out = None
        if depth_maps is not None and depth_maps[0] is not None:
            raw = depth_maps[0].detach().cpu().float().numpy().squeeze()
            depth_out = _align_depth_to_original(
                raw,
                padding=tuple(int(p) for p in data["padding"]),
                original_hw=img_np.shape[:2],
            )

        return DetectionResult(
            detections=detections,
            depth_map=depth_out,
            original_image=img_np.astype(np.uint8),
            intrinsics_original=data["original_intrinsics"]
            .detach()
            .cpu()
            .numpy(),
        )


# ---------------------------------------------------------------------------
# Backward-compat factory (the CLI scripts still call this)
# ---------------------------------------------------------------------------


def build_detector(
    checkpoint: Optional[str] = None,
    *,
    device: Optional[str] = None,
    dtype: str = "auto",
    score_threshold: float = 0.3,
    score_3d_threshold: float = 0.1,
    canonical_rotation: bool = True,
    use_depth_input: bool = False,
    input_size: Optional[int] = None,
    depth_input_size: Optional[int] = None,
    intrinsics: IntrinsicsLike = None,
    warmup_steps: int = 0,  # CLI scripts don't need it; library callers do
) -> Detector:
    """Thin backwards-compatible wrapper around :class:`Detector`.

    Library callers should prefer ``Detector(...)`` directly.
    """
    return Detector(
        checkpoint=checkpoint,
        intrinsics=intrinsics,
        device=device,
        dtype=dtype,
        score_threshold=score_threshold,
        score_3d_threshold=score_3d_threshold,
        canonical_rotation=canonical_rotation,
        use_depth_input=use_depth_input,
        input_size=input_size,
        depth_input_size=depth_input_size,
        warmup_steps=warmup_steps,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_prompt(prompt) -> list[str]:
    if isinstance(prompt, str):
        out = [p.strip() for p in prompt.split(",") if p.strip()]
    else:
        out = [str(p).strip() for p in prompt if str(p).strip()]
    if not out:
        raise ValueError("prompt is empty")
    return out


def _download_default_checkpoint() -> str:
    from huggingface_hub import hf_hub_download

    cache_dir = Path.home() / ".cache" / "wilddet3d-inference"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=_HF_REPO,
        filename=_HF_FILENAME,
        local_dir=str(cache_dir),
    )


def _align_depth_to_original(
    depth_model: np.ndarray,
    padding: tuple[int, int, int, int],
    original_hw: tuple[int, int],
) -> np.ndarray:
    """Crop letterbox padding, then resize to the original image."""
    import cv2

    left, right, top, bottom = padding
    h_pad, w_pad = depth_model.shape
    valid = depth_model[top : h_pad - bottom, left : w_pad - right]
    if valid.size == 0:
        valid = depth_model
    h_orig, w_orig = original_hw
    if valid.shape != (h_orig, w_orig):
        valid = cv2.resize(
            valid.astype(np.float32),
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST,
        )
    return valid.astype(np.float32)


def _load_image(image) -> np.ndarray:
    """Accept a numpy array, path-like, or PIL.Image; return RGB uint8."""
    if isinstance(image, np.ndarray):
        arr = image
    elif isinstance(image, (str, Path)):
        arr = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        raise TypeError(
            f"Unsupported image type: {type(image).__name__}. "
            f"Pass numpy.ndarray, path, or PIL.Image."
        )
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB(A) image, got shape {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr
