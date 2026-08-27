"""The kepmodel astrometric periodogram, run over the whole epochalypse catalog.

`src/run_kepmodel_periodograms.py` characterizes 10,000 systems per population
out of a materialized subset. This package characterizes all 5,724,586 x 3 of
them, straight off the generator's parquet shards, on the cluster the catalog
was generated on. The statistic is the same one; what changed is how the work is
divided and what comes out the other end.

See PERIODOGRAMS.md for the run itself. The import surface:

    from epochalypse_periodograms import config as C
    from epochalypse_periodograms.grid import frequency_segments, segment_periods
    from epochalypse_periodograms.periodogram import characterize_system
    from epochalypse_periodograms.shards import ShardReader, work_units
    from epochalypse_periodograms.writers import PowerStore
    from epochalypse_periodograms.calibrate import load_characterization
"""

__version__ = "0.1.0"
