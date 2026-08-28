# Shared roots for every stage. Sourced after `cd`ing to the checkout.
#
# Inputs and outputs both live on ceph: the delivered dataset is ~12 GB, the
# catalog ~50 GB, and the raw periodograms ~915 GB.
export DATA_ROOT=/mnt/ceph/users/apricewhelan/project-data/epochalypse
export OUT_ROOT=/mnt/ceph/users/apricewhelan/project-outputs/epochalypse

# The characterization reads the catalog the generator wrote -- so its
# --catalog-root IS $OUT_ROOT -- and writes its own products beside it.
export PGRAM_ROOT=$OUT_ROOT/periodograms

# harv posterior inference reads the same catalog and writes beside the
# periodograms. ~850 GB of samples at TOP_K=1024, plus ~2 GB of per-system rows.
export HARV_ROOT=$OUT_ROOT/harv

# Matplotlib's font cache. harv imports matplotlib (via its plotting helpers),
# and this site puts caches in node-local /dev/shm, so every rank on every node
# finds an empty cache, races for one lock file, loses, and rebuilds the font
# list by parsing system TTFs. Nothing in this pipeline plots.
#
# Point every rank at one pre-built cache instead. Build it ONCE, on the login
# node, with this same venv:
#
#   mkdir -p $MPLCONFIGDIR && python -c "import matplotlib.font_manager"
#
# The matplotlib version is in the cache filename, so rebuild it after an
# upgrade. MPLCONFIGDIR overrides XDG_CACHE_HOME.
export MPLCONFIGDIR=$HOME/.cache/matplotlib-mpi
