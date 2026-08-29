#!/bin/zsh -l
#SBATCH -J epochalypse-harv-finish
#SBATCH -o scripts/mpi/logs/epochalypse-harv-finish.o
#SBATCH -e scripts/mpi/logs/epochalypse-harv-finish.e
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH -p cca
#SBATCH --constraint=genoa

# Serial and cheap: nothing here touches the ~850 GB of samples. The census
# reads five columns out of the parquet dataset, the diagnostics read about
# fifteen, and the merge only concatenates the per-system rows (~2 GB). One node
# because the merge holds one population's table in memory at a time.
#
# Stage order is deliberate:
#
#   census    the headline table -- ESS, weight_captured, period recovery
#   recovery  recovery binned by injected period and eccentricity, plus a
#             histogram of where the MISSES landed. This is the stage that says
#             whether a low recovery number is the 5.5 yr baseline's fault
#             (unrecoverable by construction), the prior's, or the library's.
#   figures   the same information as four PNGs, in $HARV_ROOT/figures/
#   gallery   per-system diagnostics: the data, the reconstructed model and the
#             posterior samples, for GALLERY_PER_BIN systems from each cell of a
#             (SNR, injected period) grid. Needs --catalog-root, because it is
#             the only stage that reads epochs. Look at the 0.79-1.26 yr cells
#             first -- a one-year orbit is degenerate with parallax
#   merge     last, because it is the only stage that writes much
#
# Read the output in that order too. A single recovery percentage is close to
# uninterpretable on its own -- see HARV.md, "Reading the diagnostics".

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

# The figures import matplotlib. env.sh points MPLCONFIGDIR at a pre-built font
# cache; without it this stage still works, it just rebuilds the cache once.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu

date
python scripts/harv_finish.py --output-root $HARV_ROOT --catalog-root $OUT_ROOT \
    --stages census recovery figures gallery merge
date

echo
echo "diagnostics written to $HARV_ROOT/figures:"
ls -1 $HARV_ROOT/figures 2>/dev/null
echo
echo "per-system gallery (start with the logP-0.1to+0.1 cells):"
ls -1 $HARV_ROOT/figures/gallery/*/ 2>/dev/null | head -20
