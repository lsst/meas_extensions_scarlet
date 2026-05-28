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

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

import lsst.scarlet.lite as scl
from ..source import IsolatedSource

__all__ = ["IsolatedSourceData"]

CURRENT_SCHEMA = "1.0.1"
SOURCE_TYPE = "isolated"
scl.io.migration.MigrationRegistry.set_current(SOURCE_TYPE, CURRENT_SCHEMA)


def _encode_span_array(span_array: np.ndarray) -> str:
    """Pack a 2D 0/1 mask into a base64-encoded packed-bits string.

    The mask is flattened in C order, packed eight bits per byte via
    :func:`numpy.packbits`, then base64-encoded. The companion
    ``shape`` field carries the unflattened dimensions, since packing
    pads to the next byte boundary.

    Parameters
    ----------
    span_array : np.ndarray
        Truthy values become 1; falsy become 0.

    Returns
    -------
    encoded : str
        ASCII base64 string suitable for direct JSON embedding.
    """
    flat = np.asarray(span_array, dtype=bool).ravel().astype(np.uint8)
    return base64.b64encode(np.packbits(flat).tobytes()).decode("ascii")


def _decode_span_array(
    encoded: str, shape: tuple[int, ...], dtype: DTypeLike
) -> np.ndarray:
    """Decode the base64 packed-bits payload written by
    :func:`_encode_span_array`.

    Parameters
    ----------
    encoded : str
        Base64 string from a 1.0.1+ ``span_array`` field.
    shape : tuple[int, ...]
        Output shape. ``np.prod(shape)`` defines how many bits are
        valid; bytes past that are padding and are dropped.
    dtype : DTypeLike
        Output dtype. The decoded 0/1 values are cast to this type.

    Returns
    -------
    span_array : np.ndarray
        Mask of ``shape`` and ``dtype``.
    """
    packed = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    n = int(np.prod(shape))
    return np.unpackbits(packed)[:n].reshape(shape).astype(dtype)


@dataclass(kw_only=True)
class IsolatedSourceData(scl.io.blend.ScarletSourceBaseData):
    """A source data instance of an isolated source.

    This is used to represent sources that were not blended with any
    other sources, and therefore do not have any deblending information.

    Attributes
    ----------
    source_type : str
        The type of source data.
    version : str
        The schema version of the serialized data.
    span_array : np.ndarray
        The span mask of the source.
    origin : tuple[int, int]
        The (y, x) origin of the footprint in the observation.
    peak : tuple[int, int]
        The (y, x) coordinates of the source peak in the observation.
    metadata : dict | None
        Additional metadata associated with the source.
    """

    source_type: str = SOURCE_TYPE
    version: str = CURRENT_SCHEMA
    span_array: np.ndarray
    origin: tuple[int, int]
    peak: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization

        Returns
        -------
        result : dict[str, Any]
            The object encoded as a JSON-compatible dictionary.
        """
        result: dict[str, Any] = {
            "origin": tuple(int(o) for o in self.origin),
            "shape": tuple(int(s) for s in self.span_array.shape),
            "peak": tuple(int(p) for p in self.peak),
            "span_array": _encode_span_array(self.span_array),
            "source_type": self.source_type,
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
        data : dict
            Dictionary representation of the object
        dtype : DTypeLike
            Datatype of the resulting model.

        Returns
        -------
        result : IsolatedSourceData
            The reconstructed object.
        """
        data = scl.io.migration.MigrationRegistry.migrate(SOURCE_TYPE, data)
        shape = tuple(int(s) for s in data["shape"])
        origin = tuple(int(o) for o in data["origin"])
        span_array = _decode_span_array(data["span_array"], shape, dtype)
        peak = tuple(int(p) for p in data["peak"])
        metadata = scl.io.utils.decode_metadata(data.get("metadata", None))
        return cls(
            span_array=span_array,
            origin=origin,
            peak=peak,
            metadata=metadata,
        )

    def to_source(self, observation: scl.Observation) -> IsolatedSourceData:
        """Convert to a scarlet Source object

        Parameters
        ----------
        observation : scl.Observation
            The observation of the source.

        Returns
        -------
        result : IsolatedSourceData
            The scarlet Source object.
        """
        # Extract the image data that overlaps with the Footprint
        bbox = scl.Box(self.span_array.shape, origin=self.origin)
        image_data = observation.images[:, bbox].data

        # Mask the image data with the footprint spans
        model_data = image_data * self.span_array[None, :, :]

        # Convert the array and bounding box into a scarlet Image
        model = scl.Image(
            model_data,
            yx0=bbox.origin,
            bands=observation.bands,
        )
        return IsolatedSource(model=model, peak=self.peak, metadata=self.metadata)


IsolatedSourceData.register()


@scl.io.migration.migration(SOURCE_TYPE, "1.0.0")
def _to_1_0_1(data: dict) -> dict:
    """Migrate an ``IsolatedSourceData`` payload from schema 1.0.0 to
    1.0.1.

    1.0.0 stored ``span_array`` as a flat tuple of Python floats
    (one entry per pixel). 1.0.1 packs the same 0/1 mask into bytes
    with :func:`numpy.packbits` and base64-encodes the result, which
    cuts the on-disk representation by roughly an order of magnitude
    for typical isolated-source footprints. The migration re-encodes
    the legacy payload in place so the modern decoder handles it
    transparently.

    Parameters
    ----------
    data : dict
        The data to migrate.
    Returns
    -------
    result : dict
        The migrated data.
    """
    shape = tuple(int(s) for s in data["shape"])
    legacy = np.asarray(data["span_array"]).reshape(shape)
    data["span_array"] = _encode_span_array(legacy)
    data["version"] = "1.0.1"
    return data
