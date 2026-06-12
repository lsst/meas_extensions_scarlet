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

__all__ = ["StitchedPsfData"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import lsst.geom as geom
import numpy as np
from lsst.cell_coadds import GridContainer, UniformGrid
from lsst.scarlet.lite.io import ImagePsfData, PsfBaseData
from lsst.scarlet.lite.io.migration import PRE_SCHEMA, MigrationRegistry, migration
from lsst.skymap import Index2D
from numpy.typing import DTypeLike

if TYPE_CHECKING:
    from ..stitched_psf import ScarletStitchedPsf

CURRENT_SCHEMA = "1.0.0"
STITCHED_PSF_TYPE = "stitched"
MigrationRegistry.set_current(STITCHED_PSF_TYPE, CURRENT_SCHEMA)


def _grid_to_dict(grid: UniformGrid) -> dict[str, Any]:
    """Encode a `~lsst.cell_coadds.UniformGrid` as a JSON-compatible dict.

    Parameters
    ----------
    grid :
        The uniform grid to encode.

    Returns
    -------
    result :
        The grid encoded as ``cell_size``/``shape``/``padding``/``min``, the
        arguments needed to reconstruct it.
    """
    cell_size = grid.cell_size
    shape = grid.shape
    minimum = grid.bbox.getMin()
    return {
        "cell_size": [cell_size.x, cell_size.y],
        "shape": [shape.x, shape.y],
        "padding": grid.padding,
        "min": [minimum.getX(), minimum.getY()],
    }


def _grid_from_dict(data: dict[str, Any]) -> UniformGrid:
    """Reconstruct a `~lsst.cell_coadds.UniformGrid` from a dict.

    Parameters
    ----------
    data :
        The mapping produced by `_grid_to_dict`.

    Returns
    -------
    result :
        The reconstructed grid.
    """
    cell_size = data["cell_size"]
    shape = data["shape"]
    minimum = data["min"]
    return UniformGrid(
        geom.Extent2I(cell_size[0], cell_size[1]),
        Index2D(x=shape[0], y=shape[1]),
        padding=data.get("padding", 0),
        min=geom.Point2I(minimum[0], minimum[1]),
    )


@dataclass(kw_only=True)
class StitchedPsfData(PsfBaseData):
    """Data for a spatially-varying `ScarletStitchedPsf`.

    The per-cell image PSFs are carried as
    `~lsst.scarlet.lite.io.ImagePsfData`, so this data class composes the
    existing image-PSF codec rather than duplicating the array serialization.
    The cell layout is carried by the same
    `~lsst.cell_coadds.GridContainer`/`~lsst.cell_coadds.UniformGrid`
    primitives the domain `ScarletStitchedPsf` uses.

    Attributes
    ----------
    images :
        A `~lsst.cell_coadds.GridContainer` mapping each cell's grid index to
        the `~lsst.scarlet.lite.io.ImagePsfData` for that cell.
    grid :
        The `~lsst.cell_coadds.UniformGrid` whose inner cells partition the
        model frame.
    psf_type :
        The type of PSF being stored.
    version :
        The schema version of the stored data.
    """

    images: GridContainer[ImagePsfData]
    grid: UniformGrid
    psf_type: str = STITCHED_PSF_TYPE
    version: str = CURRENT_SCHEMA

    def to_psf(self) -> ScarletStitchedPsf:
        """Convert the storage data model into a `ScarletStitchedPsf`.

        Returns
        -------
        psf :
            The reconstructed `ScarletStitchedPsf`.
        """
        from ..stitched_psf import ScarletStitchedPsf

        images = self.images.rebuild_transformed(lambda data: data.to_psf())
        return ScarletStitchedPsf(images, self.grid)

    def as_dict(self) -> dict[str, Any]:
        """Return the object encoded into a dict for JSON serialization.

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict. The grid is encoded
            by its constructor arguments and each cell carries its grid index
            alongside the nested `ImagePsfData` dict.
        """
        return {
            "psf_type": self.psf_type,
            "version": self.version,
            "grid": _grid_to_dict(self.grid),
            "container": {
                "shape": [self.images.shape.x, self.images.shape.y],
                "offset": [self.images.offset.x, self.images.offset.y],
                "cells": [
                    {"index": [index.x, index.y], "psf": self.images[index].as_dict()}
                    for index in self.images.keys()
                ],
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], dtype: DTypeLike = np.float32) -> StitchedPsfData:
        """Reconstruct a `StitchedPsfData` from a JSON compatible dict.

        Parameters
        ----------
        data :
            Dictionary representation of the object.
        dtype :
            Datatype of the reconstructed PSF arrays.

        Returns
        -------
        result :
            The reconstructed object.
        """
        data = MigrationRegistry.migrate(STITCHED_PSF_TYPE, data)
        grid = _grid_from_dict(data["grid"])
        container = data["container"]
        shape = container["shape"]
        offset = container["offset"]
        images: GridContainer[ImagePsfData] = GridContainer(
            Index2D(x=shape[0], y=shape[1]),
            offset=Index2D(x=offset[0], y=offset[1]),
        )
        for cell in container["cells"]:
            index = Index2D(x=cell["index"][0], y=cell["index"][1])
            # Dispatch through the registry so the nested image PSF rides its
            # own codec/migration; the result is an ``ImagePsfData``.
            images[index] = PsfBaseData.from_dict(cell["psf"], dtype=dtype)
        return cls(images=images, grid=grid)


StitchedPsfData.register()


@migration(STITCHED_PSF_TYPE, PRE_SCHEMA)
def _to_1_0_0(data: dict) -> dict:
    """Migrate a pre-schema stitched PSF to schema version 1.0.0.

    There were no changes to this data model in v1.0.0 but we need to provide
    a way to migrate pre-schema data.

    Parameters
    ----------
    data :
        The data to migrate.

    Returns
    -------
    result :
        The migrated data.
    """
    data["version"] = "1.0.0"
    return data
