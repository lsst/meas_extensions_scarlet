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
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, NamedTuple, cast

import lsst.geom as geom
import numpy as np
from lsst.cell_coadds import GridContainer, StitchedPsf, UniformGrid
from lsst.scarlet.lite import Box, Image, ImagePsf, Psf
from lsst.scarlet.lite.fft import Fourier, get_fft_shape
from lsst.skymap import Index2D
from numpy.typing import DTypeLike

if TYPE_CHECKING:
    from .io.stitched_psf import StitchedPsfData

# A spatial slice ``(slice(None), slice(y0, y1), slice(x0, x1))`` into a
# ``(bands, height, width)`` array: the band axis is taken whole.
_BandSlice = tuple


class _CellSlices(NamedTuple):
    """Precomputed gather/scatter slices for one cell at a fixed image bbox.

    All four slices are ``(band, y, x)`` index tuples into either the image /
    result array (``*_image``) or the per-cell ``(bands, height, width)``
    convolution buffer (``*_pad``). They describe two overlap regions, computed
    once per image bounding box and reused by every `convolve`/`grad` call:

    - the *halo* region (the haloed cell box clipped to the image), used as the
      forward gather and the gradient scatter, and
    - the *cell* region (the inner cell box clipped to the image), used as the
      forward scatter and the gradient gather.

    The forward gathers the halo and scatters the cell; the gradient is its
    exact transpose -- it gathers the cell and scatters (adds) the halo.
    """

    halo_image: _BandSlice
    halo_pad: _BandSlice
    cell_image: _BandSlice
    cell_pad: _BandSlice


class _ConvolvePlan(NamedTuple):
    """The cached gather/scatter plan for an image of a fixed bounding box.

    The grid, cell boxes and kernels are fixed, so for a given image bounding
    box the set of overlapping cells and their gather/scatter slices never
    change across optimizer iterations. Caching this plan removes all per-cell
    `~lsst.scarlet.lite.Box` arithmetic and `~lsst.scarlet.lite.Image`
    construction from the hot loop, leaving only raw NumPy slice copies.
    """

    positions: list[int]
    pad_shape: tuple[int, int]
    cells: list[_CellSlices]


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
        # Lazily-built batched-convolution helpers (see ``_ensure_batched``).
        # The grid, cell boxes and kernels are fixed at construction, so the
        # per-cell geometry and the stacked forward/adjoint kernels are built
        # once and reused by every ``convolve``/``grad`` call.
        self._geom: list[tuple[Index2D, Box, Box]] | None = None
        self._kernel_stack: np.ndarray | None = None
        self._adjoint_stack: np.ndarray | None = None
        self._kernel_fourier: Fourier | None = None
        self._adjoint_fourier: Fourier | None = None
        # Gather/scatter slice plans, keyed by the convolved image's bounding
        # box (see ``_plan``). The hot loop convolves a single, fixed bbox, so
        # this typically holds one entry per re-matched kernel instance.
        self._plans: dict[tuple, _ConvolvePlan] = {}

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
    # Batched-convolution helpers
    # ------------------------------------------------------------------
    def _ensure_batched(self) -> None:
        """Build and cache the per-cell geometry and the stacked kernels.

        The grid, cell boxes and kernels never change after construction, so
        the per-cell bounding boxes and the ``(n_cells, bands, height, width)``
        stacks of the forward and adjoint (spatially flipped) kernels are
        computed once and reused by every `convolve`/`grad` call. Stacking the
        cells onto a leading axis lets a single batched FFT replace the former
        per-cell loop of many small FFTs.
        """
        if self._geom is not None:
            return
        order = list(self._images.keys())
        height, width = self._images.arbitrary.shape
        radius = (height // 2, width // 2)
        self._geom = []
        for index in order:
            box = self._cell_box(index)
            self._geom.append((index, box, box.grow(radius)))
        kernel_stack = np.stack([self._images[index].data for index in order])
        self._kernel_stack = kernel_stack
        # The adjoint of a convolution is a convolution with the spatially
        # flipped kernel; flip the spatial axes of every stacked cell kernel.
        self._adjoint_stack = np.ascontiguousarray(kernel_stack[:, :, ::-1, ::-1])
        self._kernel_fourier = Fourier(self._kernel_stack)
        self._adjoint_fourier = Fourier(self._adjoint_stack)

    def _intersecting(self, bbox: Box) -> Iterator[tuple[Index2D, Box, Box]]:
        """Yield the ``(index, cell_box, haloed_box)`` of cells overlapping
        ``bbox``.

        Parameters
        ----------
        bbox :
            The bounding box of the image being convolved.

        Yields
        ------
        cell : `tuple` [`~lsst.skymap.Index2D`, `~lsst.scarlet.lite.Box`, \
                `~lsst.scarlet.lite.Box`]
            The grid index, inner cell box and haloed (grown) box of every cell
            whose inner box intersects ``bbox``. The image may be a sub-region
            of the full grid (the deblender slices the observation to each
            footprint), so only the overlapping cells contribute.
        """
        self._ensure_batched()
        assert self._geom is not None
        for index, box, pad_box in self._geom:
            if box.intersects(bbox):
                yield index, box, pad_box

    @staticmethod
    def _band_slice(region: Box, origin: tuple[int, int]) -> _BandSlice:
        """Build a ``(band, y, x)`` index tuple for ``region`` within a frame.

        Parameters
        ----------
        region :
            The overlap box to index, in absolute model coordinates.
        origin :
            The ``(y, x)`` origin of the array frame ``region`` is indexed into
            (the image origin, or a cell's haloed-box origin).

        Returns
        -------
        result : `tuple`
            ``(slice(None), slice(y0, y1), slice(x0, x1))`` taking the whole
            band axis and ``region`` along the spatial axes.
        """
        (ry, rx) = region.origin
        (rh, rw) = region.shape
        (oy, ox) = origin
        return (slice(None), slice(ry - oy, ry - oy + rh), slice(rx - ox, rx - ox + rw))

    def _plan(self, bbox: Box) -> _ConvolvePlan:
        """Return the cached gather/scatter slice plan for ``bbox``.

        The plan -- which cells overlap ``bbox`` and the raw slices that gather
        each cell's sub-image and scatter its convolved result -- depends only
        on the (fixed) grid and ``bbox``, so it is computed once per bounding
        box and reused by every `convolve`/`grad` call on that box.

        Parameters
        ----------
        bbox :
            The bounding box of the image being convolved.

        Returns
        -------
        result : `_ConvolvePlan`
            The gather/scatter plan for ``bbox``.
        """
        key = (bbox.origin, bbox.shape)
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        positions: list[int] = []
        cells: list[_CellSlices] = []
        pad_shape = (0, 0)
        for k, (_index, box, pad_box) in self._intersecting_indexed(bbox):
            positions.append(k)
            pad_shape = cast(tuple[int, int], pad_box.shape)
            halo = pad_box & bbox
            cell = box & bbox
            cells.append(
                _CellSlices(
                    halo_image=self._band_slice(halo, bbox.origin),
                    halo_pad=self._band_slice(halo, cast(tuple[int, int], pad_box.origin)),
                    cell_image=self._band_slice(cell, bbox.origin),
                    cell_pad=self._band_slice(cell, cast(tuple[int, int], pad_box.origin)),
                )
            )
        plan = _ConvolvePlan(positions=positions, pad_shape=pad_shape, cells=cells)
        self._plans[key] = plan
        return plan

    def _intersecting_indexed(self, bbox: Box) -> Iterator[tuple[int, tuple[Index2D, Box, Box]]]:
        """Yield ``(position, geometry)`` for cells overlapping ``bbox``.

        Like `_intersecting`, but pairs each overlapping cell with its position
        in the cached cell ordering (and hence in the stacked kernels), which
        `_batched_fft` needs to select the right kernels.

        Parameters
        ----------
        bbox :
            The bounding box of the image being convolved.

        Yields
        ------
        item : `tuple` [`int`, `tuple`]
            The cell's position in the cached ordering and its
            ``(index, cell_box, haloed_box)`` geometry.
        """
        self._ensure_batched()
        assert self._geom is not None
        for k, (index, box, pad_box) in enumerate(self._geom):
            if box.intersects(bbox):
                yield k, (index, box, pad_box)

    def _batched_fft(
        self,
        sub_stack: np.ndarray,
        positions: list[int],
        adjoint: bool,
        cache: bool,
    ) -> np.ndarray:
        """Convolve a stack of per-cell sub-images with their cell kernels.

        A single batched FFT over the leading cell axis replaces the former
        per-cell loop of small FFTs. When every cell participates (the
        full-frame case) the cached kernel `~lsst.scarlet.lite.fft.Fourier` is
        reused, so its FFT is computed once and shared across optimizer
        iterations; when only a subset overlaps, the kernels for those cells
        are stacked on the fly.

        The FFT is sized to the larger of the haloed sub-image and the kernel
        (``use_max``), *not* their sum. The halo (``box.grow(radius)``) already
        supplies a guard band of real neighbouring pixels equal to the kernel
        radius, so a circular convolution at this size is wraparound-free over
        the central cell; the extra linear-convolution padding that the generic
        `~lsst.scarlet.lite.fft.convolve` adds on top would be a redundant
        second pad of the same guard band, roughly doubling the transform.

        Parameters
        ----------
        sub_stack :
            The ``(n_selected, bands, height, width)`` stack of per-cell
            sub-images, ordered to match ``positions``.
        positions :
            The positions, in the cached cell ordering, of the selected cells.
        adjoint :
            Whether to convolve with the adjoint (spatially flipped) kernels
            (the gradient pass) instead of the forward kernels.
        cache :
            Whether to cache the kernel FFT at the working shape.

        Returns
        -------
        result : `numpy.ndarray`
            The ``(n_selected, bands, height, width)`` stack of convolved
            sub-images.
        """
        if adjoint:
            full_fourier, full_stack = self._adjoint_fourier, self._adjoint_stack
        else:
            full_fourier, full_stack = self._kernel_fourier, self._kernel_stack
        assert full_fourier is not None and full_stack is not None and self._geom is not None
        if len(positions) == len(self._geom):
            kernel_fourier = full_fourier
        else:
            kernel_fourier = Fourier(full_stack[positions])
        axes = (-2, -1)
        # ``use_max`` sizes the FFT to the haloed sub-image rather than to the
        # sub-image *plus* the kernel, so the kernel halo is padded once, not
        # twice (see this method's docstring).
        fft_shape = get_fft_shape(sub_stack, full_stack, padding=0, axes=axes, use_max=True)
        image_fft = Fourier(sub_stack).fft(fft_shape, axes, cache=cache)
        kernel_fft = kernel_fourier.fft(fft_shape, axes, cache=cache)
        convolved = Fourier.from_fft(
            image_fft * kernel_fft, fft_shape, sub_stack.shape, axes, sub_stack.dtype
        )
        return cast(np.ndarray, np.real(convolved.image))

    def _gather(self, image: Image, plan: _ConvolvePlan, halo: bool) -> np.ndarray:
        """Gather each cell's sub-image into a zero-padded stack.

        Using the plan's precomputed raw slices (no `~lsst.scarlet.lite.Box`
        arithmetic or `~lsst.scarlet.lite.Image` construction), copy the
        relevant region of ``image`` into each cell's
        ``(bands, height, width)`` buffer, leaving the rest zero.

        Parameters
        ----------
        image :
            The multi-band image being convolved.
        plan :
            The cached gather/scatter plan for ``image``'s bounding box.
        halo :
            If `True`, gather the full haloed overlap (the forward pass, which
            pulls real neighbouring pixels); if `False`, gather only the cell
            overlap (the gradient pass, whose residual is zero outside the
            cell).

        Returns
        -------
        result : `numpy.ndarray`
            The ``(n_selected, bands, height, width)`` stack of gathered
            sub-images.
        """
        height, width = plan.pad_shape
        sub_stack = np.zeros((len(plan.positions), image.data.shape[0], height, width), dtype=image.dtype)
        data = image.data
        if halo:
            for buf, cell in zip(sub_stack, plan.cells):
                buf[cell.halo_pad] = data[cell.halo_image]
        else:
            for buf, cell in zip(sub_stack, plan.cells):
                buf[cell.cell_pad] = data[cell.cell_image]
        return sub_stack

    def _require_intersection(self, bbox: Box) -> None:
        """Raise if no cell of this PSF overlaps ``bbox``.

        A stitched PSF only convolves the cells its image overlaps, so an image
        whose bounding box misses the grid entirely would otherwise silently
        produce an all-zero result. That almost always means the image is in
        the wrong coordinate frame -- e.g. a coadd array wrapped at
        ``yx0=(0, 0)`` instead of the grid's absolute patch coordinates -- so
        fail loudly rather than return zeros.

        Parameters
        ----------
        bbox :
            The bounding box of the image being convolved.

        Raises
        ------
        ValueError
            If ``bbox`` intersects none of the PSF's cells.
        """
        self._ensure_batched()
        assert self._geom is not None
        if any(box.intersects(bbox) for _, box, _ in self._geom):
            return
        grid_bbox = self._grid.bbox
        raise ValueError(
            f"The image bounding box {bbox} does not intersect any cell of this "
            f"ScarletStitchedPsf (grid x=[{grid_bbox.getMinX()}, {grid_bbox.getMaxX()}], "
            f"y=[{grid_bbox.getMinY()}, {grid_bbox.getMaxY()}]). Check that the image's "
            "yx0 is in the PSF grid's absolute coordinate frame."
        )

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
        if mode is None:
            mode = "fft"
        if mode not in ("fft", "real"):
            raise ValueError(f"mode must be either 'fft' or 'real', got {mode}")
        result = Image.from_box(image.bbox, bands=image.bands, dtype=image.dtype)
        self._require_intersection(image.bbox)
        if mode == "real":
            # Real-space convolution does not benefit from batching; fall back
            # to the per-cell path (used by measurement/initialization, not the
            # FFT-based optimizer loop). The image may be a sub-region of the
            # full grid, so only the overlapping cells contribute.
            for index, box, pad_box in self._intersecting(image.bbox):
                clipped = box & image.bbox
                sub = image.project(bbox=pad_box)
                convolved = self._images[index].convolve(sub, mode=mode, cache=cache)
                result[clipped] = convolved[clipped]
            return result
        plan = self._plan(image.bbox)
        # Gather each cell's haloed sub-image (the halo zero-filled where it
        # extends past the image) with precomputed raw slices, convolve every
        # cell with a single batched FFT, then assign each cell's central
        # output back into the result -- cells partition the output.
        sub_stack = self._gather(image, plan, halo=True)
        convolved = self._batched_fft(sub_stack, plan.positions, adjoint=False, cache=cache)
        out = result.data
        for buf, cell in zip(convolved, plan.cells):
            out[cell.cell_image] = buf[cell.cell_pad]
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
        if mode is None:
            mode = "fft"
        if mode not in ("fft", "real"):
            raise ValueError(f"mode must be either 'fft' or 'real', got {mode}")
        result = Image.from_box(image.bbox, bands=image.bands, dtype=image.dtype)
        self._require_intersection(image.bbox)
        if mode == "real":
            # Mirror ``convolve``'s real-mode fallback: the gradient image may
            # be a sub-region of the full grid, so only the overlapping cells
            # contribute and the residual is restricted to the overlap.
            for index, box, pad_box in self._intersecting(image.bbox):
                clipped = box & image.bbox
                sub = image[clipped].project(bbox=pad_box)
                convolved = self._images[index].grad(sub, mode=mode, cache=cache)
                result.insert(convolved, op=operator.add)
            return result
        plan = self._plan(image.bbox)
        # Gather each cell's residual restricted to the cell (zero outside),
        # convolve with the flipped kernels in one batched FFT, then
        # scatter-add the full halo back: adjacent cells' halos overlap, so a
        # boundary pixel accumulates a contribution from each neighbour. This
        # additive halo scatter, not a crop-and-assign, makes the gradient the
        # transpose of the forward.
        sub_stack = self._gather(image, plan, halo=False)
        convolved = self._batched_fft(sub_stack, plan.positions, adjoint=True, cache=cache)
        out = result.data
        for buf, cell in zip(convolved, plan.cells):
            out[cell.halo_image] += buf[cell.halo_pad]
        return result
