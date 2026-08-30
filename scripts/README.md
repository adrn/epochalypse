# scripts/

Three kinds of thing, kept apart because they have different lifetimes. If a new
script does not obviously belong to one of them, it is probably a mode of an
existing script instead — that is how `sweep_summary.py` and `compare_runs.py`
ended up as two implementations of one comparison.

## The pipeline — run in order, produces the catalog and the fits

| script | stage |
| --- | --- |
| `generate_catalog.py` | parent sample, companion draws, selection, figures |
| `simulate_mpi.py` | epoch astrometry for every system (MPI) |
| `characterize_mpi.py` | periodogram search (MPI) |
| `characterize_finish.py` | calibrate, census, merge |
| `harv_mpi.py` | posterior inference (MPI) |
| `harv_finish.py` | census, recovery, figures, gallery, merge |

`mpi/1-prep.sh` … `mpi/7-harv-finish.sh` are the submit scripts, in that order;
they all source `mpi/env.sh`. `simulate_source.py` and `periodogram_source.py`
run one source end to end, for development.

## `diagnostics/` — ask a question of output that already exists

Read-only. None of these re-run the fits, and none are part of the pipeline.

| script | question |
| --- | --- |
| `check_snr.py` | is `snr_total` the signal that is actually in the along-scan data? Also `--calibrate`, which measures `snr_eff`'s absorption penalty against exact geometry, and `--self-test`, which checks the reflex reconstruction against the generator. Both need no catalog. |
| `inspect_system.py` | for one system: could the data ever have distinguished the reported period from the true one? Profile chi-square, plus a both-orbits fit for two-companion systems. |

## `benchmarks/` — sizing and calibration runs whose results justify a production choice

These cost real allocation and their results are the reason production settings
are what they are, so they are committed rather than reconstructed from shell
history — see `RESULTS.md`.

| script | what it decides |
| --- | --- |
| `bench_harv.py` / `bench_harv.sh` | seconds per system on the target CPU, at the target library size, with the node full |
| `sweep_sigma_a0.sh` | the orbit-amplitude prior, which sets the detection threshold |
| `sweep_error_model.sh` | reported vs injected uncertainties, and whether a learned jitter earns its nonlinear dimension |
| `compare_runs.py` | puts any set of run roots side by side — the one comparison tool for every sweep |
| `submit_all.sh` | submits every arm of a named sweep, and summarizes |

**One comparison tool.** Every sweep writes its arms to separate `--output-root`
directories and is read with `compare_runs.py --roots …`. Adding a
sweep-specific summariser is how the duplication happened the first time.
