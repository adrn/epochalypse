# Shared roots for every stage. Sourced after `cd`ing to the checkout.
#
# Inputs and outputs both live on ceph: the delivered dataset is ~12 GB, the
# catalog ~50 GB, and the raw periodograms ~915 GB.
export DATA_ROOT=/mnt/ceph/users/apricewhelan/project-data/epochalypse
export OUT_ROOT=/mnt/ceph/users/apricewhelan/project-outputs/epochalypse

# The characterization reads the catalog the generator wrote -- so its
# --catalog-root IS $OUT_ROOT -- and writes its own products beside it.
export PGRAM_ROOT=$OUT_ROOT/periodograms
