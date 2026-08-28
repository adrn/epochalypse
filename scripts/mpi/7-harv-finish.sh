#!/bin/zsh -l
#SBATCH -J epochalypse-harv-finish
#SBATCH -o scripts/mpi/logs/epochalypse-harv-finish.o
#SBATCH -e scripts/mpi/logs/epochalypse-harv-finish.e
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

# Serial and cheap: the census reads five columns out of the parquet dataset,
# and the merge only touches the per-system rows (~2 GB), never the ~850 GB of
# samples. One node because the merge holds one population's table in memory.

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

date
python scripts/harv_finish.py --stages census merge --output-root $HARV_ROOT
date
