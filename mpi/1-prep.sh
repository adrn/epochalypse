#!/bin/zsh -l
#SBATCH -J epochalypse-prep
#SBATCH -o logs/epochalypse-prep.o
#SBATCH -e logs/epochalypse-prep.e
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=300G
#SBATCH -t 4:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate

date
python scripts/generate_catalog.py --stages stars index
date
