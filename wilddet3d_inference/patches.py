"""Make upstream ``wilddet3d`` importable on non-CUDA machines.

The upstream repo has two files that do a top-level
``from vis4d_cuda_ops import iou_box3d``:

  - ``wilddet3d/ops/box3d.py``
  - ``wilddet3d/ops/iou_box3d.py``

These are only reached by the eval pipeline
(``wilddet3d/eval/detect3d.py``), never by the inference forward pass.
But because ``vis4d_cuda_ops`` is a CUDA-only C++ extension that won't
install on macOS, even *importing* those modules from elsewhere would
raise ``ModuleNotFoundError`` at module-load time.

We sidestep the install requirement by inserting a stub module into
``sys.modules['vis4d_cuda_ops']``. The stub exposes an ``iou_box3d``
callable that raises a clear error only if someone actually invokes it.
Pure inference never does.

We also help locate the upstream ``wilddet3d`` package: this wrapper is
designed to live next to the WildDet3D repo (or have it on PYTHONPATH /
pip-installed). We resolve via, in order:

  1. ``$WILDDET3D_ROOT`` environment variable
  2. ``~/workspace/WildDet3D``
  3. ``../WildDet3D`` relative to this wrapper's repo root
  4. Assume the user pip-installed it (no path injection needed)
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


_CUDA_OP_ERROR = (
    "vis4d_cuda_ops.iou_box3d was called, but vis4d_cuda_ops is not "
    "installed. This op is only required for the WildDet3D eval pipeline "
    "(box3d_overlap), not inference. Install vis4d_cuda_ops on a CUDA "
    "machine if you need 3D-IoU eval:\n"
    "  pip install git+https://github.com/SysCV/vis4d_cuda_ops.git "
    "--no-build-isolation --no-cache-dir"
)


def install_cuda_op_shim() -> None:
    """Insert a dummy ``vis4d_cuda_ops`` module so imports succeed.

    Idempotent. Only installs the shim if the real package isn't present.
    """
    if "vis4d_cuda_ops" in sys.modules:
        return
    try:
        import vis4d_cuda_ops  # type: ignore[import-not-found]  # noqa: F401
        return
    except ImportError:
        pass

    shim = types.ModuleType("vis4d_cuda_ops")

    def _missing(*_args, **_kwargs):
        raise NotImplementedError(_CUDA_OP_ERROR)

    shim.iou_box3d = _missing  # type: ignore[attr-defined]
    shim.__doc__ = "Stub installed by wilddet3d_inference; CUDA-only ops."
    sys.modules["vis4d_cuda_ops"] = shim


class _AnythingModule(types.ModuleType):
    """A module-like object where every attribute lookup and every call
    yields *itself*. Used to stub deep import chains we don't want to
    install (e.g. CUDA-only ``triton``). Decorators like ``@triton.jit``
    receive the decorated function and return it unchanged; attribute
    annotations like ``x: tl.constexpr`` resolve to this object, which
    Python is happy to treat as a type for annotation purposes.

    The catch: nothing here is callable as a *real* kernel. If anyone
    actually invokes the stubbed API, you'll just get something that
    silently returns ``_AnythingModule``-shaped placeholders. WildDet3D
    inference does not, so this is fine.
    """

    def __getattr__(self, name: str):  # noqa: D401
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return self

    def __call__(self, *args, **_kwargs):
        if args and callable(args[0]):
            return args[0]  # decorator usage: @triton.jit
        return self


def install_torch_cuda_stream_noop() -> None:
    """Patch ``torch.cuda.Stream`` / ``torch.cuda.stream`` to no-ops when
    CUDA is absent.

    Upstream ``wilddet3d/model.py`` uses two CUDA streams to overlap the
    SAM3 backbone and the geometry backend. On Mac/CPU PyTorch ships a
    *dummy* ``Stream`` base class that raises
    ``RuntimeError("Tried to instantiate dummy base class Stream")``,
    which blows up the forward pass before any actual op runs. We
    replace it with a tiny stub that's both a context manager and has a
    no-op ``.synchronize()``, so the backbone + geom branches just run
    sequentially.
    """
    import torch  # local to keep top-level import light

    if torch.cuda.is_available():
        return

    # Torch's `_dynamo.device_interface` asserts at import time that
    # ``torch.cuda.Stream`` inherits from ``torch._streambase._StreamBase``.
    # Make our dummy a real subclass — but override ``__new__`` so we
    # don't go through the C++ constructor (which raises on Mac).
    try:
        from torch._streambase import _StreamBase as StreamBase
    except ImportError:  # very old torch fallback
        StreamBase = object  # type: ignore[assignment, misc]

    class _DummyStream(StreamBase):  # type: ignore[misc, valid-type]
        def __new__(cls, *_args, **_kwargs):
            return object.__new__(cls)

        def __init__(self, *_args, **_kwargs):  # noqa: D401
            pass

        def synchronize(self) -> None:
            return None

        def wait_event(self, _event=None) -> None:
            return None

        def wait_stream(self, _stream=None) -> None:
            return None

        def record_event(self, _event=None) -> None:
            return None

        def query(self) -> bool:
            return True

        def __eq__(self, _other) -> bool:
            return self is _other

        def __hash__(self) -> int:
            return id(self)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    from contextlib import contextmanager

    @contextmanager
    def _stream_ctx(_stream):
        yield

    torch.cuda.Stream = _DummyStream  # type: ignore[assignment]
    torch.cuda.stream = _stream_ctx  # type: ignore[assignment]

    # `Tensor.pin_memory()` is a CUDA host-memory trick. On MPS it
    # returns a CPU tensor whose storage can't then be reassigned to
    # MPS via ``.to(non_blocking=True)`` — SAM3's
    # ``_encode_boxes`` does exactly that. Make it a no-op so the
    # subsequent ``.to(mps)`` is a plain copy.
    _orig_pin_memory = torch.Tensor.pin_memory

    def _pin_memory_noop(self, device=None):  # noqa: D401
        return self

    torch.Tensor.pin_memory = _pin_memory_noop  # type: ignore[assignment]
    # Stash original on the module so callers can still find it if
    # they really want CUDA pinning later.
    torch.Tensor._wd3d_orig_pin_memory = _orig_pin_memory  # type: ignore[attr-defined]


def apply_runtime_patches() -> None:
    """Hot-patch upstream behaviours that break off-CUDA inference.

    Must be called AFTER ``wilddet3d``/``sam3`` are imported. Idempotent.
    Current patches:

    * SAM3's ``Sam3Image._get_img_feats`` indexes ``vision_pos_enc``
      with ``img_ids``. On non-CUDA devices the pos-enc tensors come
      back from the vision backbone on CPU while everything else is on
      device — causing ``RuntimeError("indices should be either on cpu
      or on the same device as the indexed tensor")``. We wrap the
      method to coerce ``vision_pos_enc`` (and ``vision_features`` for
      good measure) to ``img_ids.device`` first.
    """
    try:
        from sam3.model.sam3_image import Sam3Image
    except ImportError:
        return  # sam3 not on path yet; caller will re-run

    if getattr(Sam3Image._get_img_feats, "_wd3d_inference_patched", False):
        return

    _orig = Sam3Image._get_img_feats

    def _patched(self, backbone_out, img_ids):
        device = img_ids.device
        for k in ("vision_pos_enc", "backbone_fpn"):
            v = backbone_out.get(k)
            if v is None:
                continue
            backbone_out[k] = [
                t.to(device) if t.device != device else t for t in v
            ]
        return _orig(self, backbone_out, img_ids)

    _patched._wd3d_inference_patched = True  # type: ignore[attr-defined]
    Sam3Image._get_img_feats = _patched

    _patch_grid_sample_for_empty()
    _patch_decoder_coord_cache()
    _patch_roi_align_dtype_coerce()
    _patch_interpolate_antialias_half()
    _patch_addmm_act_unfuse()


def _patch_grid_sample_for_empty() -> None:
    """Make ``F.grid_sample`` safe for empty grids on MPS.

    SAM3's geometry encoder unconditionally calls ``grid_sample`` even
    when there are no point/box prompts (the case for text-only
    inference). On CUDA an empty grid yields an empty result; on MPS
    PyTorch 2.5 asserts ``"[srcBuf length] > 0 ... Placeholder tensor
    is empty!"``. We short-circuit empty grids to a same-shape zero
    tensor before MPS ever sees them.
    """
    import torch
    import torch.nn.functional as F

    if getattr(F.grid_sample, "_wd3d_empty_safe", False):
        return

    _orig = F.grid_sample

    def _safe_grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=None,
    ):
        if grid.numel() == 0:
            n, c = input.shape[:2]
            h_out, w_out = int(grid.shape[1]), int(grid.shape[2])
            return torch.zeros(
                (n, c, h_out, w_out),
                dtype=input.dtype,
                device=input.device,
            )
        return _orig(
            input, grid, mode=mode, padding_mode=padding_mode,
            align_corners=align_corners,
        )

    _safe_grid_sample._wd3d_empty_safe = True  # type: ignore[attr-defined]
    F.grid_sample = _safe_grid_sample  # type: ignore[assignment]


def _patch_decoder_coord_cache() -> None:
    """Coerce SAM3 decoder's pre-baked coord cache to the runtime device.

    ``TransformerDecoder.__init__`` populates ``compilable_cord_cache``
    with ``"cuda" if torch.cuda.is_available() else "cpu"`` at module
    construction time (sam3/model/decoder.py:285). On a Mac the cache
    lives on CPU forever — but the rest of the model runs on MPS, and
    ``_get_rpb_matrix`` does arithmetic between the two, raising
    ``"Expected all tensors to be on the same device"``. We wrap the
    method to move the cache to ``reference_boxes.device`` first.
    """
    from sam3.model.decoder import TransformerDecoder

    if getattr(TransformerDecoder._get_rpb_matrix, "_wd3d_patched", False):
        return

    _orig = TransformerDecoder._get_rpb_matrix

    def _patched(self, reference_boxes, feat_size):
        device = reference_boxes.device
        cache = self.compilable_cord_cache
        if cache is not None and cache[0].device != device:
            self.compilable_cord_cache = (
                cache[0].to(device),
                cache[1].to(device),
            )
        for k, v in list(self.coord_cache.items()):
            if v[0].device != device:
                self.coord_cache[k] = (v[0].to(device), v[1].to(device))
        return _orig(self, reference_boxes, feat_size)

    _patched._wd3d_patched = True  # type: ignore[attr-defined]
    TransformerDecoder._get_rpb_matrix = _patched


def _patch_roi_align_dtype_coerce() -> None:
    """Make ``torchvision.ops.roi_align`` tolerate mixed-dtype inputs.

    Inside ``torch.autocast(fp16)`` only ops on the autocast allow-list
    are auto-cast; ``roi_align`` is not, so when SAM3's geometry
    encoder passes an fp16 feature map + fp32 boxes you get
    ``MPSHalfType does not equal MPSFloatType``. We coerce the boxes
    (and the feature map, if it differs from the boxes) to a single
    dtype before calling the original op.
    """
    import torchvision.ops as tvops

    if getattr(tvops.roi_align, "_wd3d_patched", False):
        return

    _orig = tvops.roi_align

    import torch as _torch

    def _patched(input, boxes, *args, **kwargs):
        target = input.dtype
        if isinstance(boxes, _torch.Tensor):
            if boxes.dtype != target:
                boxes = boxes.to(target)
        elif isinstance(boxes, (list, tuple)):
            boxes = type(boxes)(
                b.to(target) if b.dtype != target else b for b in boxes
            )
        return _orig(input, boxes, *args, **kwargs)

    _patched._wd3d_patched = True  # type: ignore[attr-defined]
    tvops.roi_align = _patched  # type: ignore[assignment]


def _patch_interpolate_antialias_half() -> None:
    """Promote antialiased bilinear ``F.interpolate`` to fp32 on Half.

    LingBot's encoder and WildDet3D's 3D head both call
    ``F.interpolate(..., mode='bilinear', antialias=True)``. PyTorch
    has no Half kernel for the antialias variant on MPS or CPU
    (``RuntimeError: "compute_index_ranges_weights" not implemented
    for 'Half'``). With ``torch.autocast(fp16)`` active those calls
    receive Half tensors and crash. We wrap the function: when the
    inputs are Half and antialias is requested, upcast to fp32, run
    the op, then cast back to Half.
    """
    import torch
    import torch.nn.functional as F

    if getattr(F.interpolate, "_wd3d_antialias_safe", False):
        return

    _orig = F.interpolate

    def _patched(
        input,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        # MPS has no upsample_bicubic2d kernel; it falls back to CPU
        # every call (DINOv2 hits this once per forward for its pos
        # embed interpolation). Substitute bilinear, which has an
        # MPS kernel and is visually indistinguishable for pos-embed
        # interpolation. align_corners cleared when needed.
        if mode == "bicubic" and input.device.type == "mps":
            mode = "bilinear"
        if antialias:
            # On MPS the antialias bilinear kernel doesn't exist (it
            # falls back to CPU per call — kills LingBot perf). On
            # fp16 it errors with "not implemented for Half". In both
            # cases drop antialiasing: visually it's a tiny quality
            # difference, but it keeps the op on-device.
            if input.device.type == "mps":
                antialias = False
            elif input.dtype == torch.float16:
                out = _orig(
                    input.float(),
                    size=size,
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corners,
                    recompute_scale_factor=recompute_scale_factor,
                    antialias=True,
                )
                return out.to(torch.float16)
        return _orig(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )

    _patched._wd3d_antialias_safe = True  # type: ignore[attr-defined]
    F.interpolate = _patched  # type: ignore[assignment]


def _patch_addmm_act_unfuse() -> None:
    """Replace SAM3's fused ``addmm_act`` with the unfused equivalent
    on non-CUDA devices.

    ``sam3/perflib/fused.py:addmm_act`` calls
    ``torch.ops.aten._addmm_activation`` — a CUDA-only fused
    Linear+activation kernel. On MPS the op falls back to CPU per
    call, and it fires inside every ViT block's MLP. With 32 backbone
    blocks that means 32 GPU→CPU→GPU round-trips per forward pass,
    which dominated the profile (~7 s of "unaccounted" wall time).

    The unfused path (``F.linear`` then activation) stays on whatever
    device the weights are on, and respects ``torch.autocast``. We
    also drop the original's bf16 cast on non-CUDA (MPS bf16 support
    in torch 2.5 is partial and ``F.linear`` will use the autocast
    dtype anyway).
    """
    import torch
    import torch.nn.functional as F
    from sam3.perflib import fused as _fused

    if getattr(_fused.addmm_act, "_wd3d_unfused", False):
        return
    if torch.cuda.is_available():
        # Real CUDA path is faster fused; don't touch it.
        return

    _ReLU = torch.nn.ReLU
    _GELU = torch.nn.GELU

    def _addmm_act_unfused(activation, linear, mat1):
        # Stay on whatever device the linear weights are on; let
        # autocast pick the dtype.
        y = F.linear(mat1, linear.weight, linear.bias)
        if activation in (F.relu, _ReLU):
            return F.relu(y)
        if activation in (F.gelu, _GELU):
            return F.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")

    _addmm_act_unfused._wd3d_unfused = True  # type: ignore[attr-defined]
    _fused.addmm_act = _addmm_act_unfused

    # vitdet.py imported the symbol at module load: ``from
    # sam3.perflib.fused import addmm_act``. Rebind there too.
    try:
        from sam3.model import vitdet as _vitdet

        _vitdet.addmm_act = _addmm_act_unfused
    except ImportError:
        pass


def downsample_lingbot_input(model, target_size: int) -> None:
    """Run LingBot-Depth at a lower input resolution than SAM3.

    The geometry backend (DINOv2 ViT-L, ~300 M params) dominated the
    profile at ~80% of wall time, even after lowering SAM3's input
    size. Most depth pipelines run at 224-336 input; WildDet3D only
    uses 1008 because it shares an input tensor with SAM3. We
    independently downsample LingBot's input via a forward-hook
    wrapper, then let its DINOv2 backbone interpolate positional
    embeddings (it already does this for variable sizes).

    The downstream pipeline reads ``depth_latents_hw`` from the
    geometry-backend output, so the smaller spatial extent propagates
    correctly into early-depth-fusion and the 3D head without further
    changes.

    Args:
        model: ``WildDet3DPredictor`` or ``WildDet3D``.
        target_size: LingBot input H == W in pixels (multiple of 14).
    """
    import torch
    import torch.nn.functional as F

    if target_size % 14 != 0:
        raise ValueError(
            f"target_size={target_size} must be a multiple of 14"
        )

    wd = getattr(model, "wilddet3d", model)
    backend = wd.geometry_backend
    if backend is None:
        return

    if getattr(backend, "_wd3d_input_size_override", None) == target_size:
        return  # already applied

    _orig = backend.forward

    def _patched(*args, **kwargs):
        # Resample ``images`` to ``target_size`` before LingBot runs.
        images = kwargs.get("images")
        if images is None and args:
            images = args[0]
        if images is None or images.shape[-1] == target_size:
            return _orig(*args, **kwargs)

        small = F.interpolate(
            images.float(),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).to(images.dtype)
        if "images" in kwargs:
            kwargs["images"] = small
        else:
            args = (small,) + args[1:]
        return _orig(*args, **kwargs)

    backend.forward = _patched
    backend._wd3d_input_size_override = target_size  # type: ignore[attr-defined]
    print(
        f"[wilddet3d-inference] LingBot input downsampled to "
        f"{target_size}×{target_size} (SAM3 input is independent)"
    )


def resize_sam3_vit(model, new_img_size: int) -> None:
    """Rebuild SAM3 ViT-H's global-attention RoPE buffers for a new
    image size, so the backbone can run at lower resolution.

    SAM3 ViT-H (32 blocks) uses window attention for 28 blocks and
    *global* attention only at blocks 7, 15, 23, 31. The window
    blocks operate on 24×24 patches regardless of image size, so they
    need no change. The global blocks have ``input_size = (H/14,
    W/14)`` baked in, and their ``freqs_cis`` (RoPE) was precomputed
    for that size; we re-run ``_setup_rope_freqs()`` after updating
    ``input_size`` so it produces freqs matching the new patch grid.

    There are no ``rel_pos_h/w`` learned parameters to interpolate
    (the WildDet3D config sets ``rel_pos_blocks=()``). The absolute
    positional embedding is already size-agnostic via the ``get_abs_pos``
    helper (which calls ``F.interpolate`` on demand).

    Args:
        model: a ``WildDet3DPredictor`` or the underlying ``WildDet3D``.
        new_img_size: new input image size in pixels (must be multiple
            of 14).
    """
    import torch  # noqa: F401

    if new_img_size % 14 != 0:
        raise ValueError(
            f"new_img_size={new_img_size} must be a multiple of 14"
        )

    wd = getattr(model, "wilddet3d", model)
    vit = wd.sam3.backbone.vision_backbone.trunk
    new_grid = new_img_size // 14

    # Update top-level attributes (the patch grid).
    vit.img_size = new_img_size

    rebuilt = 0
    global_ids = set(getattr(vit, "full_attn_ids", []))
    for block_idx, block in enumerate(vit.blocks):
        if block_idx not in global_ids:
            continue  # window block; input_size is window-relative
        attn = getattr(block, "attn", None)
        if attn is None or getattr(attn, "input_size", None) is None:
            continue
        h, w = attn.input_size
        if h == w == new_grid:
            continue  # already correct
        # Recompute relative_coords (just a buffer; not learned).
        if hasattr(attn, "relative_coords"):
            H, W = new_grid, new_grid
            q = torch.arange(H)[:, None]
            k = torch.arange(W)[None, :]
            rc = (q - k) + (H - 1)
            attn.register_buffer("relative_coords", rc.long())
        # Update size + rebuild RoPE freqs on the same device/dtype as
        # the existing freqs.
        old_freqs = getattr(attn, "freqs_cis", None)
        attn.input_size = (new_grid, new_grid)
        attn._setup_rope_freqs()
        if old_freqs is not None and hasattr(attn, "freqs_cis"):
            attn.freqs_cis = attn.freqs_cis.to(
                device=old_freqs.device, dtype=old_freqs.dtype
            )
        rebuilt += 1

    print(
        f"[wilddet3d-inference] resized SAM3 ViT to "
        f"{new_img_size}×{new_img_size} "
        f"({new_grid}×{new_grid} patches), rebuilt RoPE on {rebuilt} "
        f"global-attention blocks"
    )


def install_triton_shim() -> None:
    """Insert a dummy ``triton`` / ``triton.language`` so SAM3's tracker
    module (``sam3/model/edt.py``) imports cleanly on machines without
    CUDA. WildDet3D's image-only inference never exercises this path.
    """
    if "triton" in sys.modules:
        return
    try:
        import triton  # type: ignore[import-not-found]  # noqa: F401
        return
    except ImportError:
        pass

    triton = _AnythingModule("triton")
    triton_language = _AnythingModule("triton.language")
    triton.language = triton_language  # type: ignore[attr-defined]
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = triton_language


def _candidate_wilddet3d_roots() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    out: list[Path] = []
    env = os.environ.get("WILDDET3D_ROOT")
    if env:
        out.append(Path(env).expanduser().resolve())
    out.append(Path.home() / "workspace" / "WildDet3D")
    out.append((repo_root / ".." / "WildDet3D").resolve())
    return out


def ensure_wilddet3d_on_path() -> Path | None:
    """Add the WildDet3D repo to ``sys.path`` if we can find it locally.

    Returns the chosen root (or ``None`` if we punt to pip-installed).
    Also adds ``third_party/{sam3,lingbot_depth,moge}`` since the upstream
    ``__init__`` does the same.
    """
    if "wilddet3d" in sys.modules:
        return None
    for root in _candidate_wilddet3d_roots():
        if not (root / "wilddet3d" / "__init__.py").is_file():
            continue
        for p in (
            str(root),
            str(root / "third_party" / "sam3"),
            str(root / "third_party" / "lingbot_depth"),
            str(root / "third_party" / "moge"),
        ):
            if p not in sys.path:
                sys.path.insert(0, p)
        return root
    return None  # rely on pip-installed wilddet3d
