#!/bin/zsh -l
#SBATCH -J epochalypse-sim
#SBATCH -o logs/epochalypse-sim.o
#SBATCH -e logs/epochalypse-sim.e
#SBATCH -N 10
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH -t 6:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate

# one BLAS thread per rank: with tens of ranks per node the per-rank thread
# pools would otherwise oversubscribe the cores
export OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu          # skip the GPU probe on CPU nodes

# Inputs and outputs both live on ceph: the delivered dataset is ~12 GB and the
# catalog is ~50 GB, neither of which belongs in a home directory.
DATA_ROOT=/mnt/ceph/users/apricewhelan/project-data/epochalypse
OUT_ROOT=/mnt/ceph/users/apricewhelan/project-outputs/epochalypse

date
mpirun python scripts/simulate_mpi.py --skip-existing \
    --data-root $DATA_ROOT --output-root $OUT_ROOT
date
