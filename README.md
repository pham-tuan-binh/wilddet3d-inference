# WildDet3D Inference

Local, device-agnostic inference for [WildDet3D](https://github.com/allenai/WildDet3D).
Runs on Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU — without modifying upstream
sources. Exposes a single `Detector` class.

```python
from wilddet3d_inference import Detector

det = Detector()
result = det.detect("frame.jpg", "laptop, chair, scissors")

for d in result.detections:
    print(d.class_name, d.score, d.box3d)   # (cx,cy,cz,w,l,h,qw,qx,qy,qz)
```

> **Status: inference only.** Eval still needs CUDA (`vis4d_cuda_ops`).

## Install

This package is git-only for now. You also need the upstream WildDet3D source on disk.

**1. Clone WildDet3D** (used at runtime, not pip-installable):

```bash
git clone --recurse-submodules https://github.com/allenai/WildDet3D.git ~/workspace/WildDet3D
```

**2. Install this package from git.** Recommended via `uv` (handles the pydantic
override required by `vis4d==1.0.0`):

```toml
# pyproject.toml of the consuming project
[project]
dependencies = [
  "wilddet3d-inference @ git+https://github.com/pham-tuan-binh/wilddet3d-inference.git",
]

[tool.uv]
override-dependencies = ["pydantic>=2.0"]
```

Then `uv sync`.

Or with pip:

```bash
pip install "pydantic>=2.0"
pip install git+https://github.com/pham-tuan-binh/wilddet3d-inference.git
```

**3. Point the wrapper at WildDet3D.** It checks, in order:

1. `$WILDDET3D_ROOT`
2. `~/workspace/WildDet3D`
3. `../WildDet3D` relative to this repo

Set `WILDDET3D_ROOT=/path/to/WildDet3D` if upstream lives elsewhere.

## Usage

```python
from wilddet3d_inference import Detector

det = Detector(
    intrinsics=None,    # None is recommended; or YAML path / (3,3) numpy K
    device=None,        # autodetect (cuda > mps > cpu)
    dtype="auto",       # fp16 on CUDA, fp32 elsewhere
    input_size=1008,    # multiple of 14. Smaller = faster, worse alignment
)

result = det.detect("frame.jpg", "laptop, chair, scissors")
# also accepts a numpy RGB array or PIL.Image, and a list[str] prompt.
```

`result` is a `DetectionResult` with:

- `detections: list[Detection]` — `class_name`, `score`, `box2d (xyxy)`, `box3d
  (cx,cy,cz,w,l,h,qw,qx,qy,qz)`.
- `depth_map: (H, W) float32` — metric depth, pixel-aligned with `original_image`.
- `original_image: (H, W, 3) uint8` RGB.
- `intrinsics_original: (3, 3)`.

Explicit prompt modes are also available: `detect_text`, `detect_box_multi`,
`detect_box_single`, `detect_point`.

## CLI

```bash
wilddet3d-detect    path/to/image.jpg --text "car,person,chair"
wilddet3d-benchmark path/to/image.jpg --runs 5
wilddet3d-quantize  fp16 ckpt/model.pt --out ckpt/model_fp16.pt
```

## Examples

`examples/basic.py` — minimal one-shot inference + visualisation.
`examples/webcam_rerun.py` — webcam → detection → live rerun viewer (RGB,
depth, 3D boxes in one synchronised scene).

```bash
uv run python examples/basic.py path/to/image.jpg
uv run python examples/webcam_rerun.py --text "laptop,keyboard,scissors"
```

## Device support

| Device | Status | Latency (1008² fp32) |
|---|---|---|
| CUDA (Linux) | known-good upstream | ~0.3–0.8 s |
| MPS (Apple Silicon) | this wrapper's target | ~13 s |
| MPS @ `input_size=672` | works | ~3.5 s (shifts boxes ~25 cm) |
| CPU | works, slow | minutes |

## Caveats

- **`intrinsics=None` is usually best.** The model was trained largely without
  ground-truth K and learned its joint depth/box metric scale around a
  placeholder K. Real intrinsics shift `cz` away from the depth map.
- **Depth is ~10–20% shallower than the box centers.** For absolute object
  position, trust the box. Treat the point cloud as scene context.
- **Lower `input_size` increases that offset.** 1008 → ~8 cm laptop offset,
  672 → ~25 cm.
- **Eval is CUDA-only.** Anything that needs `vis4d_cuda_ops` won't run
  through this wrapper.

## How it works

Upstream `wilddet3d` hard-requires CUDA at install and forward time (CUDA
streams, fused ops, `device="cuda"` defaults). This wrapper patches those at
import time — defensive only, no upstream edits — so the model runs anywhere.
See `wilddet3d_inference/patches.py` for the full list.

## License

Inherits the upstream [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
via WildDet3D. Research/educational use per
[Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).
