# This file is part of meas_extensions_scarlet.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import DTypeLike

from lsst.afw.detection import Footprint
from lsst.afw.image import MultibandExposure
import lsst.scarlet.lite as scl

if TYPE_CHECKING:
    from .io import IsolatedSourceData


__all__ = ["IsolatedSource"]


class IsolatedSource(scl.source.SourceBase):
    """A scarlet source representation for isolated sources.
    """
    def __init__(self, model: scl.Image, peak: tuple[int, int], metadata: dict | None = None):
        """Construct an IsolatedSource.

        Parameters
        ----------
        model :
            The 3D (band, y, x) model of the source.
        center :
            The (y, x) coordinates of the peak pixel within the model.
        metadata :
            Optional metadata to store with the source.
        """
        self.model = model
        self.peak = peak
        self.metadata = metadata

    @staticmethod
    def from_footprint(
        footprint: Footprint,
        mCoadd: MultibandExposure,
        dtype: DTypeLike,
        metadata: dict | None = None,
    ) -> IsolatedSource:
        """Create an IsolatedSource from a footprint in a multiband coadd.

        Parameters
        ----------
        footprint :
            The footprint of the source in the multiband coadd.
        mCoadd :
            The multiband coadd containing the source.
        dtype :
            The desired data type for the source model.
        metadata :
            Optional metadata to store with the source.

        Returns
        -------
        source :
            The isolated source represented as a ScarletSource.
        """
        if len(footprint.peaks) != 1:
            raise ValueError(
                "Footprint must have exactly one peak to create an IsolatedSource, "
                f"found {len(footprint.peaks)}"
            )
        peak = (footprint.peaks[0].getIy(), footprint.peaks[0].getIx())
        bbox = footprint.getBBox()
        x0, y0 = bbox.getMin()
        width, height = bbox.getDimensions()
        # Convert the footprint into a boolean array
        footprint_array = footprint.spans.asArray((height, width), (x0, y0))
        # Create the 3D model array by multiplying the footprint by each band
        # of the multiband coadd.
        model_array = np.ndarray((len(mCoadd.bands), height, width), dtype=dtype)
        for bidx, band in enumerate(mCoadd.bands):
            model_array[bidx] = mCoadd[band, bbox].image.array * footprint_array
        # Create the model
        model = scl.Image(model_array, bands=mCoadd.bands, yx0=(y0, x0))
        return IsolatedSource(model=model, peak=peak, metadata=metadata)

    @property
    def bbox(self) -> scl.Box:
        """The bounding box of the source in the full Blend."""
        return self.model.bbox

    @property
    def bands(self) -> list[str]:
        """The ordered list of bands in the full source model."""
        return self.model.bands

    def get_model(self) -> scl.Image:
        """Get the full 3D (band, y, x) model of the source.

        Returns
        -------
        model :
            The 3D (band, y, x) model of the source.
        """
        return self.model

    def to_data(self) -> IsolatedSourceData:
        """Convert to a ScarletSourceData representation.

        Returns
        -------
        source_data :
            The source represented as a ScarletSourceData.
        """
        from .io import IsolatedSourceData

        span_array = np.any(self.model.data != 0, axis=0)
        return IsolatedSourceData(
            span_array=span_array,
            origin=self.bbox.origin,
            peak=self.peak,
        )
