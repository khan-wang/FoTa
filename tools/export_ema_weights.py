"""Export an EMA generator state dict from a FoTa-Net training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the EMA generator from a trusted FoTa-Net checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Full FoTa-Net training checkpoint.")
    parser.add_argument("output", type=Path, help="Destination *.weights.pth file.")
    parser.add_argument(
        "--key",
        default="emaG",
        help="Checkpoint key containing the EMA state dict (default: emaG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a mapping-based training checkpoint.")

    state_dict = checkpoint.get(args.key)
    if not isinstance(state_dict, Mapping) or not state_dict:
        available = ", ".join(sorted(str(key) for key in checkpoint.keys()))
        raise KeyError(f"State dict key '{args.key}' not found. Available keys: {available}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict), args.output)
    print(f"Exported {len(state_dict)} tensors to {args.output}")


if __name__ == "__main__":
    main()
