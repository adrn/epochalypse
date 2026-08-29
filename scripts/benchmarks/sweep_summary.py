#!/usr/bin/env python
"""Put the arms of a prior sweep side by side, from their parquet output.

    python scripts/benchmarks/sweep_summary.py $HARV_ROOT-sweeps/a0-*

Reads each arm's per-system table directly rather than parsing its log, so the
numbers are exact and the comparison cannot drift from what the census computes.
Every definition comes from `epochalypse.harv.census`, the same one the census
and the figures use.

**Read the rows against each other, never against the production run.** A sweep
runs at a smaller library size to stay cheap, which lowers recovery across every
arm; only the relative comparison at fixed M is meaningful, which is why the
sweep must always include the current setting as a control arm.

The SNR bins are fixed rather than derived from the data, so arms with different
recovery rates still line up column for column.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from epochalypse.harv import census
from epochalypse.harv import config as C
from epochalypse.periodogram import config as PG

# Fixed so arms are comparable. The first two bracket the measured cliff.
SNR_BINS = np.array([5.0, 10.0, 20.0, 40.0, 80.0, np.inf])


def arm(root, population="1_companion"):
    """One arm's numbers: the prior it used and how it did, by SNR."""
    C.set_output_root(root)
    manifest = C.manifest_path()
    settings = json.loads(manifest.read_text())["library"] if manifest.exists() else {}

    table = census.read_systems(population, census.system_columns(population))
    high_snr = census.high_snr_mask(table, population)
    searchable = census.in_search_range(table, population)
    keep = high_snr & searchable
    recovered = census.recovered(table, population)[keep]
    railed = census.railed(table)[keep]
    snr = census.best_truth(table, population, "snr_total")[keep]

    rows = []
    for lo, hi in pairwise(SNR_BINS):
        sel = (snr >= lo) & (snr < hi)
        rows.append(
            (
                int(sel.sum()),
                float(railed[sel].mean()) if sel.any() else np.nan,
                float(recovered[sel].mean()) if sel.any() else np.nan,
            )
        )
    return {
        "sigma_a0": settings.get("sigma_a0_au"),
        "n_prior": settings.get("n_prior_samples"),
        "period": (settings.get("period_min_yr"), settings.get("period_max_yr")),
        "n": int(keep.sum()),
        "recovered": float(recovered.mean()) if keep.any() else np.nan,
        "railed": float(railed.mean()) if keep.any() else np.nan,
        "bins": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("roots", nargs="+", type=Path, help="one --output-root per arm")
    parser.add_argument(
        "--population", default="1_companion", choices=list(C.POPULATIONS)
    )
    args = parser.parse_args(argv)

    arms = []
    for root in args.roots:
        try:
            arms.append((root, arm(root, args.population)))
        except FileNotFoundError as error:
            print(f"  skipping {root}: {error}")
    if not arms:
        raise SystemExit("no arms readable")

    libs = {a["n_prior"] for _, a in arms}
    periods = {a["period"] for _, a in arms}
    print(f"{args.population}, high-SNR and within the searched period range")
    print(
        f"library size : {', '.join(f'{v:,}' for v in sorted(libs) if v)}"
        + ("   *** ARMS DIFFER -- not comparable ***" if len(libs) > 1 else "")
    )
    print(
        f"period prior : {periods}"
        + ("   *** ARMS DIFFER -- not comparable ***" if len(periods) > 1 else "")
    )

    head = f"\n{'sigma_a0':>9}{'n':>8}{'recovered':>11}{'railed':>9}  |"
    for lo, hi in pairwise(SNR_BINS):
        head += f"{f'SNR {lo:g}-{hi:g}' if np.isfinite(hi) else f'SNR >{lo:g}':>16}"
    print(head)
    print(f"{'':>37}  |" + "".join(f"{'rail / rec':>16}" for _ in SNR_BINS[:-1]))
    print("-" * len(head))
    for root, a in sorted(arms, key=lambda x: -(x[1]["sigma_a0"] or 0)):
        label = f"{a['sigma_a0']:g}" if a["sigma_a0"] is not None else root.name
        line = f"{label:>9}{a['n']:>8,}{a['recovered']:>10.1%}{a['railed']:>9.1%}  |"
        for n, rail, rec in a["bins"]:
            line += f"{'--':>16}" if not n else f"{f'{rail:.0%} / {rec:.0%}':>16}"
        print(line)

    print(
        "\nrail = collapsed to the prior floor (no detection); rec = period within "
        f"{PG.PERIOD_RECOVER_TOL:g}x."
        "\nWANT: the two leftmost bins improving. WATCH: the rightmost bin degrading --"
        "\na prior tight enough to fix the faint end can bias the brightest orbits, and"
        "\nthat is the failure this sweep exists to catch."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
