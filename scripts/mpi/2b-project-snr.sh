#!/bin/zsh -l
#SBATCH -J epochalypse-project-snr
#SBATCH -o scripts/mpi/logs/epochalypse-project-snr.o
#SBATCH -e scripts/mpi/logs/epochalypse-project-snr.e
#SBATCH -N 4
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -p cca
#SBATCH --constraint=genoa

# Detectable SNR for every system: how much of each injected orbit survives the
# five-parameter astrometric solution, and therefore how much of it any fit
# could ever have seen. See SNR.md.
#
# NUMBERED 2b BECAUSE THAT IS WHERE THE DEPENDENCY IS. It needs the epochs from
# 2-sim.sh and nothing else -- not the periodograms, not the harv fits -- so it
# can run any time after the simulation, including alongside 4-periodograms.sh
# or 6-harv.sh. What it must precede is 7-harv-finish.sh, whose figures bin on
# the columns this writes; without them they fall back to snr_total and say so
# in the axis label.
#
# WHY IT EXISTS. snr_total is not a detectability measure. Position, proper
# motion and parallax are FREE parameters, so whatever part of an orbit they can
# reproduce is subtracted along with them. Measured over 3,000 high-SNR systems
# of the real catalog: the median orbit keeps ~60% of its amplitude, 18.6% keep
# under 25%, and the worst case measured keeps 5.4% -- a recorded snr_total of
# 21.5 against a detectable 1.85. Every recovery figure was binning on that.
#
# COST. ~50 systems/s per core, so ~95 core-hours over 17.2 M systems: ~15 min
# on 4 genoa nodes, and a few percent of what the harv fits cost. Per-system
# cost is flat, so ranks get contiguous slices and no cost-aware balancing is
# needed -- unlike 6-harv.sh, where the epoch-count spread is 2.7x.
#
# --ntasks-per-node IS NOT OPTIONAL. The work list is 960 whole shards (one unit
# per shard, no --n-parts), so a rank count of 2 rather than 384 gives each rank
# ~480 shards instead of ~3. The first attempt at this stage left it unset and
# died on the 4-hour limit having barely started; every other script in this
# directory sets it, and this one now does too.
#
# FOR A FUTURE CATALOG THIS BELONGS IN 2-sim.sh. The reflex is already in hand
# there -- it is what gets added to the astrometric model -- so the marginal
# cost would be one least-squares per system and nothing read back off disk.
# This stage backfills a catalog that already exists.

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu

date

# Rank 0 builds one (log10 P/T, e) table of E[retained] and broadcasts it, so
# snr_expected is an interpolation rather than 20 orientation draws per system
# -- the difference between ~95 and ~1,900 core-hours. That is only legitimate
# because E[retained] turns out to be a property of the ORBIT, not the star:
#
#     python scripts/diagnostics/check_snr.py --catalog-root $OUT_ROOT --across-stars
#
# measures the star-to-star spread at 0.8-6.5%, no larger than its own Monte
# Carlo noise. RERUN THAT ON ANY NEW CATALOG before trusting the table: it
# stands or falls on the scan law, and the measurement behind it was taken on
# uniformly distributed scan angles rather than Gaia's.
mpirun python scripts/project_snr_mpi.py --catalog-root $OUT_ROOT

date

echo
echo "written to $OUT_ROOT/detectability -- 7-harv-finish.sh will bin on it"
