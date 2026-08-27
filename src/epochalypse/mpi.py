"""The MPI launcher plumbing both parallel stages share.

`scripts/simulate_mpi.py` and `scripts/characterize_mpi.py` are SPMD: every rank
runs the same code, asks `COMM_WORLD` which rank it is, takes its own contiguous
slice of the work list, and writes its own files. Ranks talk once, in a `gather`
at the end, to print a summary. MPI is a launcher, not a message bus -- nothing
here needs MPI-IO or parallel HDF5.

That much is identical between the two stages, so it lives here rather than
being copied. What differs -- what a unit of work is, and what to do with it --
stays in the scripts.
"""

from __future__ import annotations

import os


def mpi_context():
    """(comm, rank, size). Falls back to a single rank when mpi4py is absent.

    The fallback is what makes both stages runnable on a laptop.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def slice_for_rank(n_items, rank, size):
    """This rank's contiguous [start, stop) of the work list.

    Contiguous rather than round-robin: the scan law and the epoch shards are
    memory-mapped, so a rank reading a contiguous block streams a contiguous
    region of the file. Per-item cost varies little, so I/O locality is worth
    more than load balancing.

    The remainder is spread over the first few ranks, so the largest and
    smallest slices differ by at most one item.
    """
    base, extra = divmod(n_items, size)
    start = rank * base + min(rank, extra)
    stop = start + base + (1 if rank < extra else 0)
    return start, stop


def banner(comm, size, n_items, item="sources", **extra):
    """Rank 0's header: fleet size, work per rank, and the threading warning.

    With tens of ranks per node the per-rank BLAS thread pools would
    oversubscribe the cores, and that is invisible until the job is slow.
    """
    print(
        f"ranks       : {size}"
        + ("" if comm else "  (mpi4py not found -- running as a single rank)")
    )
    print(f"{item:<12}: {n_items:,}  ->  ~{n_items // max(size, 1):,} per rank")
    for key, value in extra.items():
        print(f"{key:<12}: {value}")
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    print(
        f"threads/rank: OMP_NUM_THREADS={threads}"
        + ("" if threads == "1" else "   <- set this to 1 to avoid oversubscription"),
        flush=True,
    )


def gather(comm, summary):
    """Every rank's summary on rank 0, or just this one without mpi4py."""
    return comm.gather(summary, root=0) if comm else [summary]
