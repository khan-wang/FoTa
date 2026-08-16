"""Create a file manifest from area-binned mask directories."""

import pathlib
import argparse


def write_flist(root_dir, out_flist, include_bins):
    """
    Scans subdirectories for mask images and writes their absolute paths to a file list.
    Only includes paths from subdirectories specified in `include_bins`.

    Args:
        root_dir (pathlib.Path): The root directory containing the mask bin subdirectories.
        out_flist (pathlib.Path): The file path for the output list.
        include_bins (list): A list of subdirectory names to scan for masks.
    """
    root = pathlib.Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    paths = []
    print(f"Scanning in '{root}' for bins: {include_bins}...")
    for b in include_bins:
        bin_dir = root / b
        if not bin_dir.exists():
            print(f"  - Warning: Bin directory not found and skipped: {bin_dir}")
            continue

        found_in_bin = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            found_in_bin.extend(bin_dir.rglob(ext))

        if found_in_bin:
            print(f"  - Found {len(found_in_bin)} masks in bin '{b}'")
            # Convert to absolute paths and add to the main list
            paths.extend([str(p.resolve()) for p in found_in_bin])
        else:
            print(f"  - No masks found in bin '{b}'")

    paths = sorted(paths)
    out_flist = pathlib.Path(out_flist)
    out_flist.parent.mkdir(parents=True, exist_ok=True)

    with open(out_flist, "w", encoding="utf-8") as f:
        f.write("\n".join(paths))

    print(f"\nWrote {len(paths)} mask paths to {out_flist}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a file list (.flist) for mask datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help="Root directory of the mask dataset, containing bin subdirectories."
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help="Output file path for the generated .flist file."
    )
    parser.add_argument(
        '--bins',
        nargs='+',
        default=["mask_rates_10_20", "mask_rates_20_30", "mask_rates_30_40", "mask_rates_40_50"],
        help="A space-separated list of bin subdirectory names to include in the scan."
    )

    args = parser.parse_args()

    print("Generating mask file list...")
    write_flist(args.root, args.out, args.bins)
    print("\nFile list generation complete.")
