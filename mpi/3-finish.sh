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

date
python scripts/generate_catalog.py --stages merge select figures
date
