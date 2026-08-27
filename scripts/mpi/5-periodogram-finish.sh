#!/bin/zsh -l
#SBATCH -J epochalypse-pgram-finish
#SBATCH -o scripts/mpi/logs/epochalypse-pgram-finish.o
#SBATCH -e scripts/mpi/logs/epochalypse-pgram-finish.e
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

# Serial and cheap. `calibrate` and `census` read two columns out of the parquet
# dataset and would run on a login node; `merge` builds a 5.7 M-row frame in
# pandas, which is what the 120 GB is for. Drop `merge` and this fits in 8 GB.

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

date

python scripts/characterize_finish.py --stages calibrate census merge \
    --catalog-root $OUT_ROOT --output-root $PGRAM_ROOT
date
