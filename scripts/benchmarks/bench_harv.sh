#!/bin/zsh -l
#SBATCH -o scripts/benchmarks/logs/%x-%j.o
#SBATCH -e scripts/benchmarks/logs/%x-%j.e
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 0:30:00
#SBATCH -p cca

# A benchmark, not a pipeline stage -- which is why it lives here and not in
# scripts/mpi/ beside the numbered production scripts, and why its logs land in
# scripts/benchmarks/logs/. It writes no catalog output at all; it only reads.
#
# It does source scripts/mpi/env.sh, because $OUT_ROOT and $MPLCONFIGDIR should
# have exactly one definition in the repo.
#
# Time the harv fit under controlled conditions. Deliberately carries NO
# --constraint and only one task: both are meant to be set on the command line,
# because varying them is the entire point. Extra arguments are forwarded to
# scripts/bench_harv.py.
#
# Which CPU is fastest? One exclusive node each, submitted together so you wait
# in one queue rather than three:
#
#   for arch in genoa icelake rome; do
#       sbatch -C $arch -J bench-$arch scripts/benchmarks/bench_harv.sh
#   done
#
# What does packing a node cost? Same architecture, more ranks. Every rank fits
# the SAME systems, so per-rank time against the 1-rank run is the contention
# factor, and the aggregate line says whether more ranks still buys throughput:
#
#   for n in 1 16 32 64 96; do
#       sbatch -C genoa -J bench-genoa-r$n --ntasks-per-node=$n scripts/benchmarks/bench_harv.sh
#   done
#
# Does a cheaper library pay for itself?
#
#   sbatch -C genoa -J bench-M1e5 scripts/benchmarks/bench_harv.sh --n-prior-samples 100000
#   sbatch -C genoa -J bench-B1e4 scripts/benchmarks/bench_harv.sh --batch-size 10000
#
# --exclusive on every one of these, always: a shared node makes the numbers
# meaningless, and a contention measurement on a node someone else is using is
# measuring their job as much as yours.

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

if [[ ! -d $MPLCONFIGDIR ]]; then
    echo "WARNING: $MPLCONFIGDIR does not exist -- every rank will rebuild the"
    echo "         matplotlib font cache. Build it once on the login node:"
    echo "           mkdir -p \$MPLCONFIGDIR && python -c 'import matplotlib.font_manager'"
fi

date
mpirun python scripts/benchmarks/bench_harv.py --catalog-root $OUT_ROOT "$@"
date
