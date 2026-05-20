"""Checkpoint conversion + quantization.

Two practical paths for Mac/CPU/CUDA:

* **fp16** — half-precision weights. Works on CUDA and MPS. ~2x smaller
  on disk, ~2x faster forward on Apple Silicon GPU. Recommended Mac
  default. No accuracy loss in practice for this model class.

* **int8 (CPU only)** — ``torch.ao.quantization.quantize_dynamic`` on
  Linear layers. Useful when you don't have an MPS-capable Mac. Not
  supported on MPS as of PyTorch 2.5.

Static / per-channel int8 (needing calibration) and CoreML conversion
are intentionally out of scope for v0 — they fail on SAM3's dynamic
prompt shapes and would need a sizable rewrite. See docs/NOTES.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch


def cast_state_dict_to_fp16(checkpoint: str | Path, out_path: str | Path) -> Path:
    """Reload a fp32 WildDet3D checkpoint, downcast its tensors, save.

    This is a *weight-only* fp16 conversion: shape and key names are
    preserved, only the dtype changes. The result loads with the same
    ``build_model`` call.

    Skipping: keys whose tensors are integer dtypes (e.g. counters) or
    whose shape is 0-d (e.g. step counters). Buffers are also cast.
    """
    checkpoint = Path(checkpoint)
    out_path = Path(out_path)

    print(f"[fp16] loading {checkpoint}…")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt)
    new_state = {}
    n_cast = 0
    n_skip = 0
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            new_state[k] = v
            n_skip += 1
            continue
        if v.is_floating_point():
            new_state[k] = v.to(torch.float16)
            n_cast += 1
        else:
            new_state[k] = v
            n_skip += 1

    if "state_dict" in ckpt:
        ckpt["state_dict"] = new_state
        out_obj = ckpt
    else:
        out_obj = new_state

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, out_path)
    in_mb = checkpoint.stat().st_size / 1e6
    out_mb = out_path.stat().st_size / 1e6
    print(
        f"[fp16] cast {n_cast} tensors, kept {n_skip}. "
        f"{in_mb:.1f} MB -> {out_mb:.1f} MB. saved: {out_path}"
    )
    return out_path


def quantize_int8_cpu(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic int8 quantization to all Linear layers (CPU-only)."""
    from torch.ao.quantization import quantize_dynamic

    model = model.to("cpu").eval()
    print(
        "[int8] applying dynamic quantization to Linear layers (CPU-only)…"
    )
    qmodel = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return qmodel


def cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wilddet3d-quantize",
        description=(
            "Convert a WildDet3D fp32 checkpoint to fp16 (CUDA/MPS-ready) "
            "or apply dynamic int8 quantization (CPU-only)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fp16 = sub.add_parser("fp16", help="Save a fp16 weight-only copy")
    p_fp16.add_argument("checkpoint", type=Path)
    p_fp16.add_argument("--out", type=Path, required=True)

    p_int8 = sub.add_parser(
        "int8",
        help="Build model, dynamic-quantize Linear layers, save state dict",
    )
    p_int8.add_argument("checkpoint", type=Path)
    p_int8.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.cmd == "fp16":
        cast_state_dict_to_fp16(args.checkpoint, args.out)
        return 0

    if args.cmd == "int8":
        # Import inside to keep the CLI usable for fp16 without torch
        # pulling all of wilddet3d's transitive deps if user only wants
        # the weight cast.
        import wilddet3d_inference  # noqa: F401  (apply patches)
        from wilddet3d import build_model

        model = build_model(
            checkpoint=str(args.checkpoint),
            device="cpu",
            skip_pretrained=True,
        )
        qmodel = quantize_int8_cpu(model)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(qmodel.state_dict(), args.out)
        in_mb = args.checkpoint.stat().st_size / 1e6
        out_mb = args.out.stat().st_size / 1e6
        print(
            f"[int8] saved {args.out}  ({in_mb:.1f} MB -> {out_mb:.1f} MB)"
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(cli())
