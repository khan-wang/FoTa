"""Create a reproducible file manifest from an image directory."""

import argparse
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate a shuffled image file list.")
    parser.add_argument("--root", required=True, help="Directory to scan recursively")
    parser.add_argument("--out", required=True, help="Output .flist path")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of paths")
    parser.add_argument("--seed", type=int, default=3407, help="Shuffle seed")
    args = parser.parse_args()

    p = Path(args.root)
    if not p.is_dir():
        raise FileNotFoundError(f"Input directory not found: {p}")
    out_path = Path(args.out)
    limit = args.limit

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = sorted([str(f.absolute()) for f in p.rglob('*') if f.suffix.lower() in exts])

    if not paths:
        print(f"Warning: No files with extensions {exts} found in {p}")
        return

    random.seed(args.seed)
    random.shuffle(paths)

    if limit and limit < len(paths):
        paths = paths[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(paths))

    print(f"Wrote {len(paths)} lines to {out_path}")

if __name__ == "__main__":
    main()
