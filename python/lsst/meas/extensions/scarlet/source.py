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

from copy import deepcopy
from typing import TYPE_CHECKING, Any

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
        self.metadata = metadata
        self.components = [scl.component.CubeComponent(model=model, peak=peak)]

    @property
    def component(self) -> scl.component.CubeComponent:
        """The single component of this isolated source.

        Returns
        -------
        component :
            The CubeComponent representing this isolated source.
        """
        return self.components[0]

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
        return self.component.bbox

    @property
    def bands(self) -> list[str]:
        """The ordered list of bands in the full source model."""
        return self.component.bands

    @property
    def peak(self) -> tuple[int, int]:
        """The (y, x) coordinates of the peak pixel within the model."""
        return self.component.peak

    def get_model(self) -> scl.Image:
        """Get the full 3D (band, y, x) model of the source.

        Returns
        -------
        model :
            The 3D (band, y, x) model of the source.
        """
        return self.component._model

    def to_data(self) -> IsolatedSourceData:
        """Convert to a ScarletSourceData representation.

        Returns
        -------
        source_data :
            The source represented as a ScarletSourceData.
        """
        from .io import IsolatedSourceData

        span_array = np.any(self.component._model.data != 0, axis=0)
        return IsolatedSourceData(
            span_array=span_array,
            origin=self.bbox.origin,
            peak=self.peak,
        )

    def __copy__(self) -> IsolatedSource:
        """Create a copy of this IsolatedSource.

        Returns
        -------
        source_copy :
            A copy of this IsolatedSource.
        """
        return IsolatedSource(
            model=self.component._model,
            peak=self.component.peak,
            metadata=self.metadata,
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> IsolatedSource:
        """Create a deep copy of this IsolatedSource.

        Parameters
        ----------
        memo : dict[int, Any]
            A memoization dictionary used by `copy.deepcopy`.

        Returns
        -------
        source :
            A deep copy of this IsolatedSource.
        """
        if id(self) in memo:
            return memo[id(self)]

        source = IsolatedSource.__new__(IsolatedSource)
        memo[id(self)] = source
        source.__init__(  # type: ignore[misc]
            model=deepcopy(self.component._model, memo),
            peak=deepcopy(self.component.peak, memo),
            metadata=deepcopy(self.metadata, memo),
        )
        return source

    def __getitem__(self, indices: Any) -> IsolatedSource:
        """Get a sub-source corresponding to the given indices.

        Parameters
        ----------
        indices : Any
            The indices to use to slice the source model.

        Returns
        -------
        source :
            A new IsolatedSource that is a sub-source of this one.
        Raises
        ------
        IndexError :
            If the index includes a `Box` or spatial indices.
        """
        component = self.component[indices]
        return IsolatedSource(
            model=component._model,
            peak=component.peak,
            metadata=self.metadata,
        )
