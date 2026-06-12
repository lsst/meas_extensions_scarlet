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

__all__ = ["ScarletStitchedPsf"]

import operator
from collections.abc import Mapping
from typing import TYPE_CHECKING

import lsst.geom as geom
import numpy as np
from lsst.cell_coadds import GridContainer, StitchedPsf, UniformGrid
from lsst.scarlet.lite import Box, Image, ImagePsf, Psf
from lsst.skymap import Index2D
from numpy.typing import DTypeLike

if TYPE_CHECKING:
    from .io.stitched_psf import StitchedPsfData


class ScarletStitchedPsf(Psf):
    """A spatially-varying PSF partitioned into disjoint rectangular cells.

    Each cell of an LSST cell coadd is an *independent* coadd, so its PSF is
    genuinely discontinuous at the inner-cell boundaries: the PSF field is
    piecewise constant, not a smoothly varying function. This `Psf` represents
    that field exactly by carrying one spatially-constant
    `~lsst.scarlet.lite.ImagePsf` per cell, keyed by the cell's grid index.

    The cells are stored using the same primitives as the LSST
    `lsst.cell_coadds.StitchedPsf` -- a `~lsst.cell_coadds.GridContainer` of
    per-cell PSFs plus a `~lsst.cell_coadds.UniformGrid` that maps pixel
    positions to cells -- so this class mirrors the stack representation while
    implementing the scarlet `~lsst.scarlet.lite.Psf` interface. The
    user-facing API (every method's arguments and returns) is scarlet_lite
    (`Image`, `Box`, `Psf`); the grid primitives are an internal detail.

    The forward convolution and its adjoint (the gradient pass) are genuine
    transposes of one another rather than the same operation, because the model
    is real everywhere (so the forward *slices* real neighbouring pixels) while
    a cell-restricted residual is zero outside its cell (so the adjoint
    *zero-pads*). See `convolve` and `grad`.

    Parameters
    ----------
    images :
        A `~lsst.cell_coadds.GridContainer` mapping each cell's grid index to
        the multi-band `~lsst.scarlet.lite.ImagePsf` for that cell. Every cell
        must carry the same bands and the same uniform kernel
        ``(height, width)``, and the container must be keyed in ``grid``'s
        index space.
    grid :
        The `~lsst.cell_coadds.UniformGrid` whose inner cells partition the
        model frame (the coadd pixel frame). The model `Image` passed to
        `convolve`/`grad` must live in this frame: its bounding box origin
        ``(y0, x0)`` equals ``(grid.bbox.minY, grid.bbox.minX)`` and the union
        of the inner cell boxes equals ``grid.bbox``.

    Raises
    ------
    ValueError
        If ``images`` is empty, or its cells do not fit on ``grid``.
    """

    def __init__(
        self,
        images: GridContainer[ImagePsf],
        grid: UniformGrid,
    ):
        if len(images) == 0:
            raise ValueError("A ScarletStitchedPsf requires at least one cell.")
        # Reuse the stack's own check that the cells fit on the grid, so this
        # PSF is validated exactly like the ``lsst.cell_coadds.StitchedPsf``
        # it mirrors.
        StitchedPsf._validate_args(images, grid)
        self._images = images
        self._grid = grid

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @property
    def images(self) -> GridContainer[ImagePsf]:
        """The per-cell PSFs, keyed by grid index.

        Returns
        -------
        result : `~lsst.cell_coadds.GridContainer`
            The container of each cell's `~lsst.scarlet.lite.ImagePsf`.
        """
        return self._images

    @property
    def grid(self) -> UniformGrid:
        """The grid whose inner cells partition the model frame.

        Returns
        -------
        result : `~lsst.cell_coadds.UniformGrid`
            The uniform cell grid.
        """
        return self._grid

    @property
    def bands(self) -> tuple:
        """The bands of the PSF (uniform across cells).

        Returns
        -------
        result : `tuple`
            The bands, read from an arbitrary cell.
        """
        return self._images.arbitrary.bands

    @property
    def dtype(self) -> DTypeLike:
        """The numpy dtype of the PSF (uniform across cells).

        Returns
        -------
        result : `~numpy.dtype`
            The dtype, read from an arbitrary cell.
        """
        return self._images.arbitrary.dtype

    @property
    def shape(self) -> tuple[int, int]:
        """The uniform per-cell kernel ``(height, width)``.

        Returns
        -------
        result : `tuple` [`int`, `int`]
            The kernel shape, read from an arbitrary cell.
        """
        return self._images.arbitrary.shape

    def _cell_box(self, index: Index2D) -> Box:
        """Return the inner (unpadded) scarlet `Box` of a cell.

        `~lsst.cell_coadds.UniformGrid.bbox_of` returns a *padded* box at the
        grid edges; clipping it to ``grid.bbox`` recovers the inner cell box,
        so the cells form a gap-free, disjoint partition of the model frame.
        The geom (``x, y``) box is converted to a scarlet (``y, x``) `Box` --
        the only place this class touches `lsst.geom` geometry.

        Parameters
        ----------
        index :
            The grid index of the cell.

        Returns
        -------
        result : `~lsst.scarlet.lite.Box`
            The cell's inner bounding box in the model frame.
        """
        inner = self._grid.bbox_of(index).clippedTo(self._grid.bbox)
        return Box(
            (inner.getHeight(), inner.getWidth()),
            origin=(inner.getMinY(), inner.getMinX()),
        )

    def astype(self, dtype: DTypeLike) -> ScarletStitchedPsf:
        """Return a copy of this PSF cast to a new dtype.

        Parameters
        ----------
        dtype :
            The numpy dtype of the returned PSF.

        Returns
        -------
        result : `ScarletStitchedPsf`
            A copy of this PSF with every cell cast to ``dtype``.
        """
        images = self._images.rebuild_transformed(lambda psf: psf.astype(dtype))
        return ScarletStitchedPsf(images, self._grid)

    def __getitem__(self, bands: object) -> ScarletStitchedPsf:
        """Select a subset (or reordering) of bands as a new PSF.

        Parameters
        ----------
        bands :
            A band, or tuple of bands, to select from this PSF.

        Returns
        -------
        result : `ScarletStitchedPsf`
            A new `ScarletStitchedPsf` with every cell restricted to the
            requested bands, in order.
        """
        images = self._images.rebuild_transformed(lambda psf: psf[bands])
        return ScarletStitchedPsf(images, self._grid)

    def get_image(self, center: tuple[int, int] | None = None) -> Image:
        """Return the PSF image of the cell containing ``center``.

        Parameters
        ----------
        center :
            The ``(y, x)`` location in the model frame whose cell PSF is
            requested. The containing cell is found with the internal grid
            (an O(1) lookup); the PSF varies across cells, so a center is
            required.

        Returns
        -------
        result : `~lsst.scarlet.lite.Image`
            The `~lsst.scarlet.lite.ImagePsf` of the cell containing
            ``center``, as an `Image`.

        Raises
        ------
        ValueError
            If ``center`` is `None`, or falls outside every populated cell.
        """
        if center is None:
            raise ValueError(
                "ScarletStitchedPsf.get_image requires a center; the PSF varies across cells."
            )
        y, x = center
        try:
            index = self._grid.index(geom.Point2I(x, y))
        except (ValueError, LookupError) as err:
            raise ValueError(
                f"No cell of this ScarletStitchedPsf contains the location {center}."
            ) from err
        if index not in self._images:
            raise ValueError(
                f"No cell of this ScarletStitchedPsf contains the location {center}."
            )
        return self._images[index].get_image(center)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_data(self) -> StitchedPsfData:
        """Convert this PSF into a persistable data object.

        Returns
        -------
        result : `StitchedPsfData`
            The `StitchedPsfData` that serializes this PSF; each cell is
            carried as the `~lsst.scarlet.lite.io.ImagePsfData` of its
            `~lsst.scarlet.lite.ImagePsf`.
        """
        from .io.stitched_psf import StitchedPsfData

        images = self._images.rebuild_transformed(lambda psf: psf.to_data())
        return StitchedPsfData(images=images, grid=self._grid)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_stitched_psf(
        cls,
        psfs: Mapping[str, StitchedPsf],
        dtype: DTypeLike = np.float32,
    ) -> ScarletStitchedPsf:
        """Build a multi-band stitched PSF from per-band cell-coadd PSFs.

        Each `~lsst.cell_coadds.StitchedPsf` is single-band, so the per-band
        cell images are stacked into a multi-band
        `~lsst.scarlet.lite.ImagePsf` per cell. This is the shared entry point
        for building the scarlet observed PSF from the per-band cell coadds
        that feed deconvolution and deblending.

        Parameters
        ----------
        psfs :
            A mapping of band to that band's `~lsst.cell_coadds.StitchedPsf`.
            The mapping's key order sets the band order of the result, and all
            bands must share the same `~lsst.cell_coadds.UniformGrid`.
        dtype :
            The numpy dtype of the stacked per-cell PSF arrays.

        Returns
        -------
        result : `ScarletStitchedPsf`
            The assembled multi-band stitched PSF.

        Raises
        ------
        ValueError
            If ``psfs`` is empty, or its per-band grids do not match.
        """
        if len(psfs) == 0:
            raise ValueError("from_stitched_psf requires at least one band.")
        bands = tuple(psfs.keys())
        stitched = list(psfs.values())
        grid = stitched[0].grid
        for other in stitched[1:]:
            if other.grid != grid:
                raise ValueError("Per-band StitchedPsf objects must share a grid.")
        template = stitched[0].images
        images: GridContainer[ImagePsf] = GridContainer(template.shape, offset=template.offset)
        for index in template.keys():
            data = np.array([psf.images[index].array for psf in stitched], dtype=dtype)
            images[index] = ImagePsf(data, bands=bands)
        return cls(images, grid)

    # ------------------------------------------------------------------
    # Kernel algebra
    # ------------------------------------------------------------------
    def match(self, other: Psf, padding: int | None = None) -> ScarletStitchedPsf:
        """Build the difference kernel that matches ``other`` to ``self``.

        Each cell is matched independently, so the result is a
        `ScarletStitchedPsf` over the *same* partition whose cells are the
        per-cell difference kernels. This is what an
        `~lsst.scarlet.lite.Observation` convolves with when its observed PSF
        is stitched.

        Parameters
        ----------
        other :
            The PSF to match from. This is the single spatially-constant model
            `~lsst.scarlet.lite.ImagePsf`; each cell's `ImagePsf.match` already
            broadcasts a band-less model across the cell's bands.
        padding :
            Padding to use when generating the FFT. If `None`, each cell's own
            ``padding`` is used.

        Returns
        -------
        result : `ScarletStitchedPsf`
            The stitched difference kernel as a new `ScarletStitchedPsf`.
        """
        images = self._images.rebuild_transformed(
            lambda psf: psf.match(other, padding=padding)
        )
        return ScarletStitchedPsf(images, self._grid)

    # ------------------------------------------------------------------
    # Convolution (forward + adjoint)
    # ------------------------------------------------------------------
    def convolve(self, image: Image, mode: str | None = None, cache: bool = False) -> Image:
        """Apply this PSF's forward convolution operator to ``image``.

        The output is partitioned by cell, so every output pixel is convolved
        with exactly its own cell's kernel. For each cell ``c`` with kernel
        ``Kc`` and half-width ``r``, the model is sliced over the grown box
        ``c.grow(r)`` (pulling *real* neighbouring pixels, zero-filled only
        past the global model frame), convolved with ``Kc``, cropped back to
        ``c`` and **assigned** into the output.

        Parameters
        ----------
        image :
            The multi-band model image to convolve. Its bounding box must
            contain every cell.
        mode :
            The convolution mode, ``"fft"`` or ``"real"``. If `None`, each
            cell's `ImagePsf` chooses its default (``"fft"``).
        cache :
            Whether to cache the FFT of each cell kernel at the working shape.

        Returns
        -------
        result : `~lsst.scarlet.lite.Image`
            The convolved image, with the bands and bounding box of ``image``.
        """
        result = Image.from_box(image.bbox, bands=image.bands, dtype=image.dtype)
        for index in self._images.keys():
            kernel = self._images[index]
            box = self._cell_box(index)
            # The image may be a sub-region of the full grid (the deblender
            # slices the observation to each footprint), so only the cells
            # overlapping it contribute; the rest are skipped and the cell box
            # is clipped to the image for the output assignment.
            if not box.intersects(image.bbox):
                continue
            clipped = box & image.bbox
            radius = (kernel.shape[0] // 2, kernel.shape[1] // 2)
            pad_box = box.grow(radius)
            # Slice the model over the grown cell box; ``project`` zero-fills
            # where the halo extends past the image bounds.
            sub = image.project(bbox=pad_box)
            convolved = kernel.convolve(sub, mode=mode, cache=cache)
            # Crop to the cell (clipped to the image) and assign -- cells
            # partition the output.
            result[clipped] = convolved[clipped]
        return result

    def grad(self, image: Image, mode: str | None = None, cache: bool = False) -> Image:
        """Apply the adjoint of `convolve` to ``image`` (the gradient pass).

        This is the exact transpose of `convolve`, *not* the same operation.
        For each cell ``c`` with kernel ``Kc`` and half-width ``r``, the
        gradient is restricted to the unpadded cell box ``c``, **zero-padded**
        to ``c.grow(r)`` (correct precisely because a cell-restricted residual
        is zero outside ``c``), convolved with the flipped kernel
        ``flip(Kc)``, and the full grown-box result is **additively inserted**
        into the model-gradient image. The additive insert is what makes it
        the transpose: adjacent cells' halos overlap, so a model pixel in a
        boundary halo accumulates a contribution from each neighbour, each
        weighted by that neighbour's own kernel.

        Parameters
        ----------
        image :
            The multi-band gradient image to convolve. Its bounding box must
            contain every cell.
        mode :
            The convolution mode, ``"fft"`` or ``"real"``. If `None`, each
            cell's `ImagePsf` chooses its default (``"fft"``).
        cache :
            Whether to cache the FFT of each cell adjoint kernel at the working
            shape.

        Returns
        -------
        result : `~lsst.scarlet.lite.Image`
            The result of applying the adjoint convolution to ``image``.
        """
        result = Image.from_box(image.bbox, bands=image.bands, dtype=image.dtype)
        for index in self._images.keys():
            kernel = self._images[index]
            box = self._cell_box(index)
            # Mirror ``convolve``: the gradient image may be a sub-region of
            # the full grid, so skip non-overlapping cells and restrict the
            # residual to the overlap. Clipping both directions to the image
            # keeps the adjoint the exact transpose of the forward.
            if not box.intersects(image.bbox):
                continue
            clipped = box & image.bbox
            radius = (kernel.shape[0] // 2, kernel.shape[1] // 2)
            pad_box = box.grow(radius)
            # Restrict the residual to the (clipped) cell, then zero-pad it to
            # the grown box so the adjoint convolution can spread into it.
            sub = image[clipped].project(bbox=pad_box)
            convolved = kernel.grad(sub, mode=mode, cache=cache)
            # Keep the full halo and accumulate it (scatter-add) -- ``insert``
            # clips it to the image bounds. This, not a crop-and-assign, is
            # what makes the operator the true transpose.
            result.insert(convolved, op=operator.add)
        return result
