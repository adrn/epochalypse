#!/bin/zsh -l
#SBATCH -o scripts/benchmarks/logs/%x-%j.o
#SBATCH -e scripts/benchmarks/logs/%x-%j.e
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -p cca
#SBATCH -C genoa

# One arm of the ERROR MODEL sweep. Same discipline as sweep_sigma_a0.sh: a
# calibration that sizes and justifies a production choice, committed rather
# than reconstructed from shell history.
#
#   sbatch -J err-reported scripts/benchmarks/sweep_error_model.sh reported
#   sbatch -J err-injected scripts/benchmarks/sweep_error_model.sh injected
#   sbatch -J err-jitter   scripts/benchmarks/sweep_error_model.sh reported 0.05
#   sbatch -J err-both     scripts/benchmarks/sweep_error_model.sh injected 0.05
#
#   python scripts/benchmarks/compare_runs.py --roots $HARV_ROOT-err-* --figure <png>
#
# Arguments:
#   $1  error mode: "reported" or "injected"   (required)
#   $2  learned jitter scale in mas, or "none"  (default none)
#   $3  HIGH-SNR systems per population         (default 2000)
#   $4  prior library size                      (default 10000000)
#
# ==========================================================================
# WHAT IS BEING SEPARATED
# ==========================================================================
# The generator injects noise at sigma_UEVA,single (AL + calibration) and
# REPORTS sigma_formal (attitude + AL, no calibration term). That is deliberate
# -- equating them would give an artificially self-consistent data set -- but it
# means every fit weights by an uncertainty smaller than the scatter it is
# looking at. Measured over 3,000 high-SNR systems: median ratio 1.276, tail to
# 11.5.
#
# The consequence is not a small bias. Weight is exp(-dchi2/2) and chi-square
# scales as 1/sigma^2, so under-reported errors sharpen the likelihood contrast
# between library draws by r^2 = 1.63 IN THE EXPONENT. The sampler is not merely
# wrong on those systems, it is overconfident by construction.
#
# Four arms, two independent switches:
#
#   reported            (c) the status quo, and the baseline every number in
#                       HARV.md was measured under.
#   injected            (b) weight by the scale the noise was actually drawn
#                       from. Correct calibration at ZERO cost in library
#                       resolution, because nothing new is sampled. Defensible
#                       for real data too: DR3 publishes an excess-noise
#                       estimate that does not depend on the orbit.
#   reported + jitter   (a) learn the excess variance. The honest test of
#                       whether an analyst who does NOT know the calibration
#                       term can recover it -- and the arm that pays for it.
#   injected + jitter   the control on (a). On correctly weighted data a
#                       learned jitter should return ~0. If it does not, the
#                       jitter is absorbing the LIBRARY's inadequacy rather than
#                       the data's noise, which is the failure mode that would
#                       make (a) look good for the wrong reason.
#
# ==========================================================================
# WHY THE JITTER ARMS ARE NOT FREE
# ==========================================================================
# harv's Jitter declares a NONLINEAR parameter, so the shared library goes from
# three sampled nonlinear dimensions to four. At fixed M the effective
# resolution per dimension falls from M^(1/3) ~ 215 to M^(1/4) ~ 56 at M=1e7,
# and library size was measured to saturate near 1e6 for three dimensions.
# Adding a dimension is close to dividing the library, so the jitter arms must
# run at the SAME M as the baseline or the comparison measures resolution rather
# than the error model. That is what $4 is for, and compare_runs.py prints the
# manifests side by side precisely so a drift in it cannot pass unnoticed.
#
# There is also a design tension worth stating before reading any result:
# sigma_reported varies star to star, but the library is ONE library for every
# system. So the jitter prior is ABSOLUTE, in mas, and has to be broad enough to
# cover the whole catalog's noise range -- wasting resolution on every
# individual star. A per-star jitter prior would fix that and would break the
# shared library. sigma_a0 escapes the same tension only because it shapes
# analytically marginalized priors and is never drawn.

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

MODE=${1:?usage: sweep_error_model.sh <reported|injected> [jitter_mas|none] [n] [M]}
JITTER=${2:-none}
NSYS=${3:-2000}
NLIB=${4:-10000000}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

SUFFIX="$MODE"
JITTER_ARG=()
if [[ "$JITTER" != "none" ]]; then
    SUFFIX="$MODE-jitter$JITTER"
    JITTER_ARG=(--jitter-sigma "$JITTER")
fi
ROOT="$HARV_ROOT-err-$SUFFIX"

echo "arm         : error-mode=$MODE jitter=$JITTER"
echo "systems     : $NSYS per population (high-SNR only)"
echo "library     : $NLIB"
echo "output      : $ROOT"
date

mpirun python scripts/harv_mpi.py \
    --catalog-root "$OUT_ROOT" --output-root "$ROOT" \
    --error-mode "$MODE" "${JITTER_ARG[@]}" \
    --subsample "$NSYS" --n-prior-samples "$NLIB" \
    --min-snr 5 --n-parts 6

date

# The census is cheap and reading it beside the run that produced it is what
# makes a log self-contained. The cross-arm comparison is compare_runs.py.
python scripts/harv_finish.py --output-root "$ROOT" --stages census recovery
