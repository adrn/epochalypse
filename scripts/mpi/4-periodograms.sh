#!/bin/zsh -l
#SBATCH -J epochalypse-pgram
#SBATCH -o scripts/mpi/logs/epochalypse-pgram.o
#SBATCH -e scripts/mpi/logs/epochalypse-pgram.e
#SBATCH -N 8
#SBATCH --ntasks-per-node=120  # only need 960 ranks total
#SBATCH --exclusive
#SBATCH -t 16:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

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
