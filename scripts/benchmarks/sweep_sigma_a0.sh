#!/bin/zsh -l
#SBATCH -o scripts/benchmarks/logs/%x-%j.o
#SBATCH -e scripts/benchmarks/logs/%x-%j.e
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -p cca
#SBATCH -C genoa

# One arm of the orbit-amplitude prior sweep. This is a CALIBRATION, not a
# benchmark, but it lives here for the same reason the timing runs do: its
# result sizes and justifies the production run, and it must be reproducible
# rather than reconstructed from someone's shell history.
#
#   sbatch -J sweep-a0-0.01 scripts/benchmarks/sweep_sigma_a0.sh 0.01
#   zsh scripts/benchmarks/submit_all.sh sigma-a0      # submits every arm
#   python scripts/benchmarks/sweep_summary.py <roots> # compares them
#
# Arguments (all optional after the first):
#   $1  sigma_a0 in AU at P0            (required)
#   $2  HIGH-SNR systems per population (default 2000)
#   $3  prior library size              (default 10000000)
#
# ==========================================================================
# WHY IT RUNS AT THE PRODUCTION LIBRARY SIZE
# ==========================================================================
# A cheaper library would confound the two effects this is trying to separate.
# At M=1e5 a system can rail because the library never found its orbit, which
# looks identical to railing because the amplitude prior made the null cheaper.
# The default is 1e7 -- ten times the production run -- so that a rail is
# unambiguously a statement about the PRIOR and not about resolution. There is
# no point measuring a detection threshold with a library that imposes one of
# its own.
#
# That is affordable because the sweep spends its budget on the systems that
# carry the signal instead of a random draw:
#
#   * --populations 1_companion. The control cannot inform a rail-vs-SNR curve.
#   * --min-snr. Only 6.9% of the catalog is high-SNR, and the cliff lives in
#     SNR 5-20, so a random subsample wastes ~93% of its fits.
#
# 2,000 high-SNR systems per arm, four arms:
#
#     M=1e7     502 core-h,  ~1.3 h  on one genoa node   <- the default
#     M=1e6      56 core-h,  ~9 min   (pass 1000000 as $3)
#
# MEMORY. M=1e7 is projected at ~7.5 GB/rank -- 722 GB of a 1.5 TB node at 96
# ranks, so ~2x headroom, but that is EXTRAPOLATED from a measurement at M=1e6.
# harv_mpi.py now reports the real peak RSS per rank in its summary, so the
# first arm to finish settles it. If it comes back near 15 GB/rank, resubmit the
# rest with --ntasks-per-node=64.
#
# NOTE: --sigma-a0 PINS a constant, disabling the per-system scaling that
# production uses (config.M_MAX_MJUP / M_star^(2/3), see harv/config.py). That
# is deliberate: a sweep needs one number per arm. To sweep the production
# parameterization instead, vary --m-max-mjup -- 13 MJup at the catalog's median
# host of 0.41 Msun is sigma_a0 = 0.022 AU, between this sweep's two best arms.
#
# ==========================================================================
# WHY THIS SWEEP EXISTS
# ==========================================================================
# `SIGMA_A0_AU` is the width of the Gaussian prior on the orbit's astrometric
# amplitude. It is not a free knob: it sets the DETECTION THRESHOLD, because a
# wider prior means a larger Occam penalty on a real orbit relative to the
# no-orbit solution, and that penalty must be paid out of the orbit's likelihood
# advantage (~SNR^2/2 nats). The penalty grows with period, since the prior
# width scales as (P/P0)^(2/3), so it falls almost entirely on real orbits and
# barely at all on the null.
#
# The 300,000-system run at M=1e6 measured the consequence. Rail fraction -- the
# share whose best sample collapsed to the prior floor, i.e. NO detection --
# against injected SNR, on high-SNR 1_companion systems:
#
#     SNR   5.0-10.0   n=2,671   railed 50.2%   recovered  5.3%
#     SNR  10.0-20.0   n=2,031   railed 20.4%   recovered 34.5%
#     SNR  20.0-40.1   n=1,343   railed  2.4%   recovered 68.9%
#     SNR  40.1-80.4   n=  662   railed  0.3%   recovered 76.0%
#     SNR      >80     n=  240   railed  0.0%   recovered 82-100%
#
# 2_companion reproduces the same cliff independently. So the prior is putting
# the effective threshold near SNR 15-20 while HIGH_SNR_MIN is 5, and the
# 5-10 bin -- 38% of the high-SNR sample -- recovers 5.3%.
#
# SIGMA_A0_AU = 1.0 is ~4,900x the median injected a0 (2.0e-4 AU). The Occam
# arithmetic predicts the crossover at SNR ~7.3 at 1.0 AU and ~4.9 at 0.01 AU.
# That predicts the DIRECTION, not the magnitude, which is what this measures.
#
# ==========================================================================
# WHAT TO LOOK FOR, AND THE RISK
# ==========================================================================
# Compare arms at FIXED library size -- always include sigma_a0 = 1.0 as the
# control, because M=1e5 lowers recovery across the board and only the relative
# comparison is meaningful.
#
#   * do the SNR 5-10 and 10-20 bins improve? that is the whole point.
#   * DOES THE SNR > 80 BIN DEGRADE? this is the risk and the reason for a
#     sweep rather than a single change. 0.01 AU is ~20x too NARROW for the
#     highest-amplitude short-period orbits (injected a0 reaches 2.2 AU), so a
#     prior tight enough to fix the faint end can bias the bright end. If the
#     top bins fall, the sweep has found the floor and the answer is 0.1 or
#     0.03, not 0.01.
#
# Note sigma_a0 does NOT change the library fingerprint -- it only affects the
# analytically marginalized Thiele-Innes priors, which are never drawn -- so
# every arm needs its own --output-root. That is what $ROOT below is for.

set -u
SIGMA_A0=${1:?usage: sbatch scripts/benchmarks/sweep_sigma_a0.sh <sigma_a0 AU> [n_high_snr] [M]}
SUBSAMPLE=${2:-2000}
N_PRIOR=${3:-10000000}
MIN_SNR=${MIN_SNR:-5}

cd /mnt/home/apricewhelan/work/epochalypse
source .venv/bin/activate
source scripts/mpi/env.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

ROOT=${SWEEP_ROOT:-$HARV_ROOT-sweeps}/a0-$SIGMA_A0
echo "sigma_a0 = $SIGMA_A0 AU   high-SNR systems = $SUBSAMPLE   M = $N_PRIOR"
echo "min_snr  = $MIN_SNR   output = $ROOT"

date
mpirun python scripts/harv_mpi.py --sigma-a0 $SIGMA_A0 --subsample $SUBSAMPLE \
    --n-prior-samples $N_PRIOR --n-parts 1 \
    --populations 1_companion --min-snr $MIN_SNR \
    --catalog-root $OUT_ROOT --output-root $ROOT
date

# Each arm carries its own diagnostic, so a log is self-contained even before
# sweep_summary.py puts the arms side by side.
python scripts/harv_finish.py --output-root $ROOT --populations 1_companion \
    --stages census recovery
date
