#!/bin/zsh -l
#SBATCH -J epochalypse-harv
#SBATCH -o scripts/mpi/logs/epochalypse-harv.o
#SBATCH -e scripts/mpi/logs/epochalypse-harv.e
#SBATCH -N 16
# --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -p cca
#SBATCH --constraint=genoa

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

# one BLAS thread per rank: the per-system design matrix is ~300 x 9, so BLAS
# threads buy nothing and with tens of ranks per node they oversubscribe
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

date
#mpirun python scripts/harv_mpi.py --skip-existing --n-parts 3 \
    #    --catalog-root $OUT_ROOT --output-root $HARV_ROOT
# --n-parts 6, not 3. Work is assigned cost-first (units are predicted from the
# truth tables' epoch counts), but that only pays when no single unit costs more
# than a rank's fair share. At 16 nodes x 96 ranks the measured/simulated
# allocation used is ~52% at 1.9 units per rank, ~77% at 1.9 with cost-aware
# assignment, and ~95% at 3.8. Raise it further if you add ranks.
mpirun python scripts/harv_mpi.py --subsample 10000 --n-parts 6 \
      --catalog-root $OUT_ROOT --output-root $HARV_ROOT
date
