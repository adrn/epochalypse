#!/bin/zsh -l
#SBATCH -J epochalypse-finish
#SBATCH -o scripts/mpi/logs/epochalypse-finish.o
#SBATCH -e scripts/mpi/logs/epochalypse-finish.e
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

date
python scripts/generate_catalog.py --stages merge select figures \
    --data-root $DATA_ROOT --output-root $OUT_ROOT
date
