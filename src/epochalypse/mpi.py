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


def stride_for_rank(items, rank, size):
    """This rank's interleaved share -- `items[rank::size]`.

    The alternative to `slice_for_rank`, and the choice between them is about
    whether per-item cost is flat.

    Contiguous slicing is right when it is: the periodogram's cost varies ~15%
    across the catalog because the frequency loop dominates, so I/O locality is
    worth more than balance and a rank streams a contiguous region of one
    population's directory.

    It is wrong when cost varies with the same thing the ordering follows. The
    harv stage costs time linear in a system's padded epoch count, work units
    are ordered by shard, and shard order is sky order -- so a contiguous slice
    is a contiguous patch of sky with correlated transit counts. Measured on the
    first production run: per-unit cost varied 2.4x and 44% of the allocation
    sat idle waiting for the ranks that drew the expensive shards. Striding
    decorrelates cost from rank at the price of I/O locality, which that stage
    can afford -- it spends ~25 s of compute per system against reading one row
    group.

    Returns the items themselves rather than bounds, because a stride is not
    expressible as a `start, stop` pair.
    """
    return items[rank::size]


def balance(items, costs, rank, size):
    """This rank's share, longest-processing-time-first.

    Sort by cost descending and give each item to the currently least-loaded
    rank. That is the classic LPT heuristic -- provably within 4/3 of optimal
    for this problem, and in practice near-perfect once no single item dominates
    the total.

    Use it instead of `slice_for_rank` or `stride_for_rank` when per-item cost
    has a long tail AND each rank gets only a few items. Those two differ in
    whether cost is *correlated* with position; neither reduces the variance of
    a rank's total, so with ~2 items per rank the slowest rank is set by whoever
    drew the most expensive one. Measured on the harv stage: contiguous gave 57%
    of the allocation used, striding 52%, with unit cost spanning 2.7x.

    Deterministic given the same `costs`, so every rank computes the same
    assignment with no communication -- but the costs themselves have to agree,
    which is what `broadcast` is for.
    """
    import heapq

    order = sorted(range(len(items)), key=lambda i: -costs[i])
    heap = [(0.0, r) for r in range(size)]
    heapq.heapify(heap)
    mine = []
    for i in order:
        load, owner = heapq.heappop(heap)
        if owner == rank:
            mine.append(items[i])
        heapq.heappush(heap, (load + float(costs[i]), owner))
    return mine


def broadcast(comm, value, root=0):
    """`value` from `root` on every rank, or itself without mpi4py."""
    return comm.bcast(value, root=root) if comm else value


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
