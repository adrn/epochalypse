#!/bin/zsh -l
#SBATCH -J epochalypse-pgram
#SBATCH -o logs/epochalypse-pgram.o
#SBATCH -e logs/epochalypse-pgram.e
#SBATCH -N 10
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

# 320 ranks = one per shard of one population, 960 work units in all, so each
# rank does three shards (~53,700 systems). At the measured ~0.6 s/system on a
# rome core that is ~9 h; the 12 h walltime is the margin. Raising
# --ntasks-per-node to 64 halves it, and unlike the generator, per-rank memory
# is not the constraint -- a rank holds one row group of epochs and one shard's
# truth table, ~0.3 GB, so 128 ranks/node fits comfortably. Cores are the
# constraint, so take as many as the partition will give.
#
# Do NOT pass -n to mpirun: the allocation already fixes the rank count, and
# run_mpi.py is SPMD (every rank runs the same code and asks COMM_WORLD which
# one it is), so there is no -m mpi4py.futures and no --mpi flag either.

cd /mnt/home/apricewhelan/work/epochalypse/periodograms
source .venv/bin/activate

# one BLAS thread per rank: the per-system arrays are ~100 x 9, so BLAS threads
# buy nothing and with tens of ranks per node they oversubscribe the cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# No --catalog-root or --output-root: the defaults are <repo>/outputs (where
# generate_catalog.py left the shards) and <repo>/periodograms/outputs. The
# characterization tables are ~6.3 GB and the default --power subsample adds
# ~1.6 GB of curves, which is small enough to sit in the repo tree. Switching
# to --power all is ~915 GB and belongs on ceph:
#
#   --output-root /mnt/ceph/users/apricewhelan/projects/epochalypse/periodograms
#
date
mpirun python scripts/run_mpi.py --skip-existing
date
