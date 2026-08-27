#!/bin/zsh -l
#SBATCH -J epochalypse-finish
#SBATCH -o logs/epochalypse-finish.o
#SBATCH -e logs/epochalypse-finish.e
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate

# Inputs and outputs both live on ceph: the delivered dataset is ~12 GB and the
# catalog is ~50 GB, neither of which belongs in a home directory.
DATA_ROOT=/mnt/ceph/users/apricewhelan/project-data/epochalypse
OUT_ROOT=/mnt/ceph/users/apricewhelan/project-outputs/epochalypse

date
python scripts/generate_catalog.py --stages merge select figures \
    --data-root $DATA_ROOT --output-root $OUT_ROOT
date
