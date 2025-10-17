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

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

import lsst.scarlet.lite as scl
from ..source import IsolatedSource

__all__ = ["IsolatedSourceData"]

CURRENT_SCHEMA = "1.0.0"
SOURCE_TYPE = "isolated"
scl.io.migration.MigrationRegistry.set_current(SOURCE_TYPE, CURRENT_SCHEMA)


@dataclass(kw_only=True)
class IsolatedSourceData(scl.io.blend.ScarletSourceBaseData):
    """A source data instance of an isolated source.

    This is used to represent sources that were not blended with any
    other sources, and therefore do not have any deblending information.
    """

    source_type: str = SOURCE_TYPE
    version: str = CURRENT_SCHEMA
    footprint: np.ndarray
    origin: tuple[int, int]
    peak: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization

        Returns
        -------
        result :
            The object encoded as a JSON-compatible dictionary.
        """
        result: dict[str, Any] = {
            "origin": tuple(int(o) for o in self.origin),
            "shape": tuple(int(s) for s in self.footprint.shape),
            "peak": tuple(float(p) for p in self.peak),
            "footprint": tuple(self.footprint.flatten().astype(float)),
            "version": self.version,
        }
        if self.metadata is not None:
            result["metadata"] = scl.io.utils.encode_metadata(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> IsolatedSourceData:
        """Reconstruct `IsolatedSourceData` from JSON compatible
        dict.

        Parameters
        ----------
        data :
            Dictionary representation of the object
        dtype :
            Datatype of the resulting model.

        Returns
        -------
        result :
            The reconstructed object
        """
        data = scl.io.MigrationRegistry.migrate(SOURCE_TYPE, data)
        shape = tuple(int(s) for s in data["shape"])
        origin = tuple(int(o) for o in data["origin"])
        footprint = np.array(data["footprint"], dtype=dtype).reshape(shape)
        peak = tuple(int(p) for p in data["peak"])
        metadata = scl.io.utils.decode_metadata(data.get("metadata", None))
        return cls(
            footprint=footprint,
            origin=origin,
            peak=peak,
            metadata=metadata,
        )

    def to_source(self, observation: scl.Observation) -> IsolatedSourceData:
        """Convert to a scarlet Source object

        Parameters
        ----------
        observation :
            The observation of the source.

        Returns
        -------
        result :
            The scarlet Source object.
        """
        # Extract the image data that overlaps with the Footprint
        bbox = scl.Box(self.footprint.shape, origin=self.origin)
        image_data = observation.images[:, bbox].data

        # Mask the image data with the footprint
        model_data = image_data * self.footprint[None, :, :]

        # Convert the array and bounding box into a scarlet Image
        model = scl.Image(
            model_data,
            yx0=bbox.origin,
            bands=observation.bands,
        )
        return IsolatedSource(model=model, peak=self.peak, metadata=self.metadata)


IsolatedSourceData.register()
