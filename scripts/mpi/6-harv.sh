#!/bin/zsh -l
#SBATCH -J epochalypse-harv
#SBATCH -o scripts/mpi/logs/epochalypse-harv.o
#SBATCH -e scripts/mpi/logs/epochalypse-harv.e
#SBATCH -N 16
# --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

# 2048 ranks, 960 shards -> --n-parts 3 gives 2880 units of ~5,970 systems, so
# every rank gets one unit and a bit. Parts matter far more here than in the
# periodogram stage: at ~5 s/system a whole 17,890-system shard is ~25 h on one
# rank, so without splitting the walltime is set by the shard, not the cluster.
#
# THE BUDGET IS A LAPTOP EXTRAPOLATION. 2.5 s/system measured on an M-series
# core, doubled for Rome, is ~24,000 core-hours for the catalog. Run the
# subsample below FIRST and read the real rate out of its log before committing
# this job -- if Rome is 3x rather than 2x, this allocation does not finish.
#
#   mpirun python scripts/harv_mpi.py --subsample 20000 --max-units 96 \
#       --catalog-root $OUT_ROOT --output-root $HARV_ROOT
#
# Per-rank memory is ~1 GB (32 MB library, one row group of epochs, one shard's
# truths) plus the JAX compile cache for the nine bucket shapes, so 64 ranks per
# node is conservative on a 1 TB node. Raise it if the subsample run shows the
# cache is small.

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

# one BLAS thread per rank: the per-system design matrix is ~300 x 9, so BLAS
# threads buy nothing and with tens of ranks per node they oversubscribe
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

date
mpirun python scripts/harv_mpi.py --skip-existing --n-parts 3 \
    --catalog-root $OUT_ROOT --output-root $HARV_ROOT
date
