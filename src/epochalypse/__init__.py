"""Simulated Gaia DR4 epoch astrometry.

The library half of the project: the stages the scripts in `scripts/` drive.
Nothing scientific is decided outside `config.py` -- every prior, threshold,
path, and seed is a module constant there.

    stars      parent stellar sample
    sources    per-source lookup (SourceCatalog, ScanLawStore) + high-SNR view
    planets    per-source companion draw (Roche + Hill screens)
    astrometry per-source epoch simulation + ShardWriter
    figures    the catalog figures
    constants  physical and mission constants, derived from astropy
    fitting    periodogram characterization -- targets the serial catalog, not ported
"""
