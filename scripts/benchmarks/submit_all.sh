#!/bin/zsh
# Submit the harv benchmark suite, or summarize what came back.
#
#   zsh scripts/benchmarks/submit_all.sh            # all three experiments (10 jobs)
#   zsh scripts/benchmarks/submit_all.sh arch       # 3 jobs: which CPU?
#   zsh scripts/benchmarks/submit_all.sh ranks      # 5 jobs: what does packing a node cost?
#   zsh scripts/benchmarks/submit_all.sh settings   # 2 jobs: is a cheaper library cheaper?
#   zsh scripts/benchmarks/submit_all.sh summary    # table from the logs, no submitting
#
# Every job is one exclusive node for at most 30 minutes, so the whole suite is
# under 5 node-hours. They are independent and submitted together, so you wait
# in one queue rather than ten.
#
#   ARCH=icelake zsh scripts/benchmarks/submit_all.sh ranks
#
# ARCH sets which CPU the `ranks` and `settings` experiments use. It defaults to
# genoa on the argument that this kernel is memory-bandwidth-bound and genoa has
# ~3x rome's bandwidth per core -- but that is a prediction, so run `arch` first
# and set ARCH from the answer rather than trusting it.

set -u
REPO=${0:A:h:h:h}
cd $REPO

WHICH=${1:-all}
ARCH=${ARCH:-genoa}
SUBMIT=scripts/benchmarks/bench_harv.sh
LOGS=scripts/benchmarks/logs

# ==========================================================================
# summary -- read the logs, no submitting
# ==========================================================================
if [[ $WHICH == summary ]]; then
    setopt local_options null_glob
    logs=($LOGS/*.o)
    if (( ${#logs} == 0 )); then
        print "no logs in $LOGS yet"
        exit 0
    fi
    printf "%-22s %6s %8s %11s %13s %12s\n" \
        job ranks compile "warm s/sys" "sys/s / node" "catalog c-h"
    printf '%s\n' "----------------------------------------------------------------------------------"
    for log in $logs; do
        name=${${log:t}%-*}
        ranks=$(grep -m1 '^ranks' $log 2>/dev/null | sed 's/.*: *\([0-9]*\).*/\1/')
        compile=$(grep -m1 '^compile' $log 2>/dev/null | awk '{print $3}')
        # anchor on '^  min': the 'compile' line above it also says "median"
        warm=$(grep -m1 '^  min ' $log 2>/dev/null | awk '{print $4}')
        node=$(grep -m1 'throughput' $log 2>/dev/null | awk '{print $(NF-2)}')
        ch=$(grep -m1 '^  catalog' $log 2>/dev/null | sed 's/.*-> *//; s/ core-h.*//')
        printf "%-22s %6s %8s %11s %13s %12s\n" \
            ${name:-?} ${ranks:-?} ${compile:-?} ${warm:-?} ${node:-?} ${ch:-?}
    done
    print "\nLower 'warm s/sys' is a faster core; higher 'sys/s / node' is more"
    print "work per node. They can disagree -- the second is what you are buying."
    exit 0
fi

# ==========================================================================
# preflight -- each of these silently invalidates a run
# ==========================================================================
source scripts/mpi/env.sh
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

fail=0
if [[ ! -d $OUT_ROOT/data/simulated_astrometry ]]; then
    print "ERROR: no catalog at \$OUT_ROOT ($OUT_ROOT)"
    fail=1
fi
if [[ ! -d $MPLCONFIGDIR ]]; then
    print "WARNING: \$MPLCONFIGDIR ($MPLCONFIGDIR) missing -- every rank will"
    print "         rebuild matplotlib's font cache. Fix it now with:"
    print "           mkdir -p \$MPLCONFIGDIR && python -c 'import matplotlib.font_manager'"
fi
if ! python -c "import mpi4py" 2>/dev/null; then
    print "WARNING: mpi4py is not importable. Every rank will believe it is rank"
    print "         0 of 1, which makes the multi-rank numbers meaningless. Fix:"
    print "           MPICC=\$(which mpicc) uv pip install --no-binary=mpi4py mpi4py"
    [[ $WHICH == ranks || $WHICH == all ]] && fail=1
fi
(( fail )) && { print "\nrefusing to submit"; exit 1 }
mkdir -p $LOGS

ids=()
submit() {  # submit <job-name> <sbatch args...> -- <script args...>
    local name=$1; shift
    local -a sb; local -a extra
    while [[ $# -gt 0 && $1 != "--" ]]; do sb+=$1; shift; done
    shift 2>/dev/null || true
    extra=($@)
    local id=$(sbatch --parsable -J $name $sb $SUBMIT $extra)
    ids+=$id
    printf "  %-22s %s   %s\n" $name $id "${sb[*]} ${extra[*]}"
}

# ==========================================================================
# the experiments
# ==========================================================================
if [[ $WHICH == arch || $WHICH == all ]]; then
    print "\n=== which CPU? (identical work on one exclusive node each) ==="
    for arch in genoa icelake rome; do
        submit bench-$arch -C $arch --ntasks-per-node=1
    done
fi

if [[ $WHICH == ranks || $WHICH == all ]]; then
    print "\n=== what does packing a $ARCH node cost? ==="
    # 96 is genoa's core count; icelake has 64 and rome 128, so trim the scan to
    # what the node actually has rather than letting sbatch reject it.
    case $ARCH in
        genoa)   scan=(1 16 32 48 96) ;;
        icelake) scan=(1 16 32 48 64) ;;
        rome)    scan=(1 16 32 64 128) ;;
        *)       scan=(1 16 32 64) ;;
    esac
    for n in $scan; do
        submit bench-$ARCH-r$n -C $ARCH --ntasks-per-node=$n
    done
fi

if [[ $WHICH == settings || $WHICH == all ]]; then
    print "\n=== is a cheaper library actually cheaper? (1 rank, $ARCH) ==="
    submit bench-M1e5 -C $ARCH --ntasks-per-node=1 -- --n-prior-samples 100000
    submit bench-B1e4 -C $ARCH --ntasks-per-node=1 -- --batch-size 10000
fi

print "\n${#ids} job(s) submitted."
print "  watch    : squeue -u \$USER -o '%.10i %.24j %.2t %.6M %R'"
print "  results  : zsh scripts/benchmarks/submit_all.sh summary"
print "  cancel   : scancel ${ids[*]}"
