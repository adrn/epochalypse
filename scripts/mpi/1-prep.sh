#!/bin/zsh -l
#SBATCH -J epochalypse-prep
#SBATCH -o scripts/mpi/logs/epochalypse-prep.o
#SBATCH -e scripts/mpi/logs/epochalypse-prep.e
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 4:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

date
python scripts/generate_catalog.py --stages stars index \
    --data-root $DATA_ROOT --output-root $OUT_ROOT
date
