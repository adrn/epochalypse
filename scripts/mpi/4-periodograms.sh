#!/bin/zsh -l
#SBATCH -J epochalypse-pgram
#SBATCH -o scripts/mpi/logs/epochalypse-pgram.o
#SBATCH -e scripts/mpi/logs/epochalypse-pgram.e
#SBATCH -N 10
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

# 320 ranks = one per shard of one population, 960 work units in all, so each
# rank does three shards. Unlike the generator, per-rank memory is not the
# constraint here (~0.3 GB: one row group of epochs plus one shard's truths),
# so take as many cores as the partition will give -- see PERIODOGRAMS.md.

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

# one BLAS thread per rank: the per-system arrays are ~100 x 9, so BLAS threads
# buy nothing and with tens of ranks per node they oversubscribe the cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

date
mpirun python scripts/characterize_mpi.py --skip-existing \
    --catalog-root $OUT_ROOT --output-root $PGRAM_ROOT
date
