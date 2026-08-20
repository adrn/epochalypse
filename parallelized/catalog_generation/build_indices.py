#!/usr/bin/env python
"""Build the per-source lookup indices. Run once, after stage 1.

Every worker needs to fetch one star's row and one star's scan law by Gaia
source id. At ~4 million stars the scan law is far too large to load per
process, so this builds two small index files that make the lookup O(1) against
a memory-mapped Arrow table:

    outputs/data/index/stars_index.parquet     gaia_source_id -> row
    outputs/data/index/scanlaw_index.parquet   gaia_source_id -> (offset, length)

The scan-law index assumes each source occupies one contiguous block; that is
verified here and reported as an error naming the offending ids rather than
silently returning the wrong epochs.

Usage
-----
    python catalog_generation/build_indices.py
    python catalog_generation/build_indices.py --overwrite --output-root /scratch/run7
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_catalog import build_config      # noqa: E402
from pipeline.sources import build_indices     # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args.output_root.resolve() if args.output_root else None)
    if not config.paths.stars_csv.exists():
        raise SystemExit(f"{config.paths.stars_csv} not found; run the stars stage first:\n"
                         "  python catalog_generation/generate_catalog.py --stages stars")

    print(f"indices -> {config.paths.index_dir}")
    started = time.time()
    build_indices(config, overwrite=args.overwrite)
    print(f"done in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
