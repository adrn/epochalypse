"""Gaia scanning law simulation utilities."""

import pathlib

import equinox as eqx
import h5py
import healpy as hp
import numpy as np
import pooch
from unxt import Quantity, ustrip

SCANLAW_DATA_DOI = "10.5281/zenodo.17729248"
SCANLAW_DATA_POOCH = pooch.create(
    path=pooch.os_cache("epochalypse"),
    base_url=f"doi:{SCANLAW_DATA_DOI}/",
    registry={
        "scanlaw_nside64_dr3.h5": "md5:0e41f3c0b3fb98d927b0b62944b3953c",
        "scanlaw_nside64_dr4.h5": "md5:23dcb697eced65db70169deef7ce94a6",
        "scanlaw_nside64_dr5.h5": "md5:bc938363e86100f75966ec38aa697fce",
    },
)


class AbstractGaiaScanLaw(eqx.Module):
    """Abstract base class for Gaia scanning laws."""


class GaiaReducedCommandedScanLaw(AbstractGaiaScanLaw):
    """A simple, reduced version of the Gaia commanded scan law.

    This class provides an interface to query a reduced version of the Gaia commanded
    scanning law. The commanded scan law provides the times, pointings, and parallax
    factors for the two fields of view of the Gaia spacecraft sampled at a ~10 second
    cadence. The "reduced" version of the scan law groups samples by sky location (into
    healpix pixels) and treats each transit of a healpix pixel as a single scan of that
    pixel.

    Parameters
    ----------
    dr
        Data release identifier (e.g., "dr3", "dr4", etc.) specifying which version
        of the scan law data to use. This mainly affects the time span covered by the
        scan law.
    """

    dr: str = eqx.field(converter=lambda x: str(x).lower())
    random_downsample_fraction: float = eqx.field(
        default=0.0, converter=lambda x: float(x)
    )
    random_seed: int | None = eqx.field(
        default=None, converter=lambda x: int(x) if x is not None else None
    )

    # Internal
    _nside: int = eqx.field(init=False, repr=False)

    def __post_init__(self) -> None:
        with h5py.File(self.file_path, "r") as f:
            self._nside = hp.order2nside(f.attrs["healpix_level"])

    @property
    def file_path(self) -> pathlib.Path:
        """Get the path to the cached scan law file, download if necessary."""
        fname = f"scanlaw_nside64_{self.dr}.h5"
        if fname not in SCANLAW_DATA_POOCH.registry:
            raise ValueError(
                f"Data for {self.dr} not available. Available scan laws: "
                f"{list(SCANLAW_DATA_POOCH.registry.keys())}"
            )
        return pathlib.Path(SCANLAW_DATA_POOCH.fetch(fname))

    def read_metadata(self) -> dict:
        """Read and return the metadata from the scan law file."""
        with h5py.File(self.file_path, "r") as f:
            return dict(f.attrs)

    def load_scans_for_healpix(self, healpix_pixel: int) -> np.ndarray:
        """Load the scan data for a specific healpix pixel.

        Parameters
        ----------
        healpix_pixel
            Healpix pixel for which to load the scan data.

        Returns
        -------
        scan_data
            Structured array containing the scan data for the specified healpix pixel.
        """
        with h5py.File(self.file_path, "r") as f:
            start = f["index/starts"][healpix_pixel]
            count = f["index/counts"][healpix_pixel]
            scans = f["scans"][start : start + count]

        if self.random_downsample_fraction > 0:
            rng = np.random.default_rng(self.random_seed)
            frac = 1 - self.random_downsample_fraction
            idx = rng.choice(
                len(scans),
                size=int(len(scans) * frac),
                replace=False,
            )
            scans = scans[idx]

        return scans[np.argsort(scans["time_bjd"])]

    def query(self, ra: Quantity["angle"], dec: Quantity["angle"]) -> np.ndarray:
        """Query the scan data for a specific sky location.

        Parameters
        ----------
        ra
            Right ascension of the sky location.
        dec
            Declination of the sky location.

        Returns
        -------
        scan_data
            Structured array containing the scan data for the specified sky location.
        """
        ra = ustrip("deg", Quantity.from_(ra))
        dec = ustrip("deg", Quantity.from_(dec))
        healpix_pixel = hp.ang2pix(self._nside, ra, dec, lonlat=True)
        return self.load_scans_for_healpix(healpix_pixel)
