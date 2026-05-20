"""Download the official WildDet3D checkpoint into ./ckpt/.

Equivalent to::

    huggingface-cli download allenai/WildDet3D \\
        wilddet3d_alldata_all_prompt_v1.0.pt --local-dir ckpt/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("ckpt"))
    parser.add_argument(
        "--filename",
        default="wilddet3d_alldata_all_prompt_v1.0.pt",
    )
    parser.add_argument("--repo-id", default="allenai/WildDet3D")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.repo_id}/{args.filename} -> {args.out}/")
    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=str(args.out),
    )
    print(f"done: {path}")


if __name__ == "__main__":
    main()
