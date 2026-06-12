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

"""Tests for the spatially-varying ``ScarletStitchedPsf``."""

from __future__ import annotations

import json
import unittest

import lsst.geom as geom
import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np
from lsst.afw.detection import GaussianPsf
from lsst.cell_coadds import GridContainer, StitchedPsf, UniformGrid
from lsst.meas.extensions.scarlet import ScarletStitchedPsf
from lsst.meas.extensions.scarlet.io.model_data import LsstScarletModelData
from lsst.meas.extensions.scarlet.io.stitched_psf import StitchedPsfData
from lsst.scarlet.lite import Image, ImagePsf
from lsst.skymap import Index2D
from numpy.typing import DTypeLike

# Geometry shared by the fixtures: a 2x2 grid of square cells.
BANDS = ("g", "r")
CELL = 8
GRID = 2
KERNEL = 5  # odd, so the half-width is unambiguous
SIZE = CELL * GRID

# Geometry for the factory fixtures, where the cells are larger than
# the PSF kernels (as in a real cell coadd).
FACTORY_CELL = 15
FACTORY_KERNEL = 11
FACTORY_GRID = 2
FACTORY_SIZE = FACTORY_CELL * FACTORY_GRID


def _make_grid(cell: int = CELL, grid: int = GRID) -> UniformGrid:
    """Return a `~lsst.cell_coadds.UniformGrid` of square cells at the origin.

    Parameters
    ----------
    cell :
        The side length of each (square) cell.
    grid :
        The number of cells along each axis.

    Returns
    -------
    result :
        The uniform grid spanning ``[0, cell * grid)`` on each axis.
    """
    return UniformGrid(
        geom.Extent2I(cell, cell),
        Index2D(x=grid, y=grid),
        padding=0,
        min=geom.Point2I(0, 0),
    )


def _normalized_kernel(rng: np.random.RandomState) -> ImagePsf:
    """Return a random, peak-normalized ``(bands, KERNEL, KERNEL)`` ImagePsf.

    Parameters
    ----------
    rng :
        The seeded generator used to draw the kernel.

    Returns
    -------
    psf :
        A multi-band image PSF whose per-band kernels each sum to one.
    """
    data = rng.rand(len(BANDS), KERNEL, KERNEL).astype(np.float32)
    data /= data.sum(axis=(1, 2))[:, None, None]
    return ImagePsf(data, bands=BANDS)


def _stitched_psf(
    rng: np.random.RandomState, kernel: ImagePsf | None = None
) -> tuple[ScarletStitchedPsf, UniformGrid]:
    """Build a `ScarletStitchedPsf` over a ``GRID x GRID`` partition.

    Parameters
    ----------
    rng :
        The seeded generator used to draw per-cell kernels.
    kernel :
        If given, every cell shares this single kernel (a uniform field);
        otherwise each cell draws its own distinct kernel.

    Returns
    -------
    psf :
        The assembled stitched PSF.
    grid :
        The grid partitioned by the cells.
    """
    grid = _make_grid()
    images = GridContainer(Index2D(x=GRID, y=GRID))
    for iy in range(GRID):
        for ix in range(GRID):
            images[Index2D(x=ix, y=iy)] = (
                kernel if kernel is not None else _normalized_kernel(rng)
            )
    return ScarletStitchedPsf(images, grid), grid


def _random_image(rng: np.random.RandomState, dtype: DTypeLike = np.float64) -> Image:
    """Return a random multi-band `Image` over the fixture bounding box.

    Parameters
    ----------
    rng :
        The seeded generator used to draw the image.
    dtype :
        The dtype of the image data.

    Returns
    -------
    image :
        A random image spanning the ``SIZE x SIZE`` fixture box.
    """
    return Image(rng.rand(len(BANDS), SIZE, SIZE).astype(dtype), bands=BANDS, yx0=(0, 0))


def _gaussian_stitched(sigma: float, cell: int = FACTORY_CELL) -> StitchedPsf:
    """Return a single-band `~lsst.cell_coadds.StitchedPsf` of Gaussians.

    Every cell carries the same normalized Gaussian kernel image, so the
    resulting PSF is uniform; the factory test only needs the per-band cell
    images to be distinguishable, which ``sigma`` provides.

    Parameters
    ----------
    sigma :
        The Gaussian sigma of the per-cell kernel.
    cell :
        The side length of each grid cell.

    Returns
    -------
    result :
        The assembled single-band stitched PSF.
    """
    grid = _make_grid(cell=cell, grid=FACTORY_GRID)
    images = GridContainer(Index2D(x=FACTORY_GRID, y=FACTORY_GRID))
    kernel = GaussianPsf(FACTORY_KERNEL, FACTORY_KERNEL, sigma).computeKernelImage(
        geom.Point2D(0, 0)
    )
    for iy in range(FACTORY_GRID):
        for ix in range(FACTORY_GRID):
            images[Index2D(x=ix, y=iy)] = kernel
    return StitchedPsf(images, grid)


class ScarletStitchedPsfTestCase(lsst.utils.tests.TestCase):
    """Tests for the geometry, convolution and IO of a stitched PSF."""

    def setUp(self):
        super().setUp()
        self.rng = np.random.RandomState(42)
        self.psf, self.grid = _stitched_psf(self.rng)

    def test_geometry(self):
        """The stitched PSF reports uniform per-cell geometry."""
        self.assertEqual(self.psf.shape, (KERNEL, KERNEL))
        self.assertEqual(self.psf.bands, BANDS)
        self.assertEqual(self.psf.dtype, np.float32)
        self.assertEqual(self.psf.grid, self.grid)
        self.assertEqual(len(self.psf.images), GRID * GRID)

    def test_astype(self):
        """``astype`` casts every cell without altering geometry."""
        cast = self.psf.astype(np.float64)
        self.assertEqual(cast.dtype, np.float64)
        self.assertEqual(cast.grid, self.grid)
        for index in self.psf.images.keys():
            cell = cast.images[index]
            self.assertEqual(cell.dtype, np.float64)
            np.testing.assert_array_almost_equal(cell.data, self.psf.images[index].data)

    def test_getitem_bands(self):
        """Band selection restricts every cell to the requested bands."""
        single = self.psf["r"]
        self.assertEqual(single.bands, ("r",))
        for index in self.psf.images.keys():
            cell = single.images[index]
            self.assertEqual(cell.bands, ("r",))
            np.testing.assert_array_equal(cell.data, self.psf.images[index]["r"].data)

    def test_get_image(self):
        """``get_image`` returns the PSF of the cell containing the center."""
        # A location inside the lower-right cell of the 2x2 grid.
        center = (CELL + 1, CELL + 1)
        target = self.psf.images[Index2D(x=1, y=1)]
        image = self.psf.get_image(center)
        np.testing.assert_array_equal(image.data, target.get_image(center).data)

    def test_get_image_requires_center(self):
        """``get_image`` refuses an ambiguous (center-less) request."""
        with self.assertRaises(ValueError):
            self.psf.get_image(None)

    def test_get_image_outside(self):
        """``get_image`` raises when the location falls outside every cell."""
        with self.assertRaises(ValueError):
            self.psf.get_image((SIZE + 5, SIZE + 5))

    def test_match(self):
        """``match`` returns a same-partition stitched difference kernel."""
        model_psf = ImagePsf(np.ones((1, KERNEL, KERNEL), dtype=np.float32) / KERNEL**2)
        diff = self.psf.match(model_psf)
        self.assertIsInstance(diff, ScarletStitchedPsf)
        self.assertEqual(set(diff.images.keys()), set(self.psf.images.keys()))
        self.assertEqual(diff.grid, self.grid)
        # Each cell of the result is the cell's own difference kernel.
        for index in diff.images.keys():
            expected = self.psf.images[index].match(model_psf)
            np.testing.assert_array_almost_equal(diff.images[index].data, expected.data)

    def test_adjoint(self):
        """``grad`` is the exact transpose of ``convolve``.

        For random images ``x`` and ``y``, ``<A x, y> == <x, A^T y>`` to
        numerical precision. This is the load-bearing correctness property of
        the stitched convolution: if the forward/adjoint stitch geometry or
        centering is off, the optimizer would not converge to the right model.
        """
        x = _random_image(self.rng)
        y = _random_image(self.rng)
        forward = float(np.sum(self.psf.convolve(x).data * y.data))
        adjoint = float(np.sum(x.data * self.psf.grad(y).data))
        self.assertFloatsAlmostEqual(forward, adjoint, rtol=1e-10)

    def test_adjoint_real_mode(self):
        """The transpose property also holds for real-space convolution."""
        x = _random_image(self.rng)
        y = _random_image(self.rng)
        forward = float(np.sum(self.psf.convolve(x, mode="real").data * y.data))
        adjoint = float(np.sum(x.data * self.psf.grad(y, mode="real").data))
        self.assertFloatsAlmostEqual(forward, adjoint, rtol=1e-10)

    def test_uniform_equivalent_to_image_psf(self):
        """A uniform stitched field equals a single ``ImagePsf`` convolution.

        Away from the global edge (within a kernel radius), every output pixel
        sees only real neighbours, so a stitched PSF whose cells all share one
        kernel must reproduce convolving the whole image with that kernel.
        """
        kernel = _normalized_kernel(self.rng)
        uniform, _ = _stitched_psf(self.rng, kernel=kernel)
        x = _random_image(self.rng)
        stitched = uniform.convolve(x)
        single = kernel.convolve(x)
        radius = KERNEL // 2
        interior = (slice(None), slice(radius, SIZE - radius), slice(radius, SIZE - radius))
        self.assertFloatsAlmostEqual(
            stitched.data[interior], single.data[interior], atol=1e-6
        )

    def test_to_data_roundtrip(self):
        """``to_data`` round-trips through the registry to an equal PSF."""
        data = self.psf.to_data()
        self.assertIsInstance(data, StitchedPsfData)
        # Mirror the scarlet_lite contract: the domain PSF does not serialize
        # itself; only its companion data object owns ``as_dict``.
        self.assertFalse(hasattr(self.psf, "as_dict"))

        encoded = data.as_dict()
        self.assertEqual(encoded["psf_type"], "stitched")
        # Dispatch purely on the ``psf_type`` tag, as persistence does.
        restored = scl.io.PsfBaseData.from_dict(encoded).to_psf()
        self.assertIsInstance(restored, ScarletStitchedPsf)
        self.assertEqual(set(restored.images.keys()), set(self.psf.images.keys()))
        self.assertEqual(restored.grid, self.psf.grid)
        for index in self.psf.images.keys():
            np.testing.assert_array_almost_equal(
                restored.images[index].data, self.psf.images[index].data
            )
            self.assertEqual(restored.images[index].bands, BANDS)

    def test_grid_roundtrip(self):
        """Serialization preserves the grid and the container layout."""
        restored = scl.io.PsfBaseData.from_dict(self.psf.to_data().as_dict()).to_psf()
        self.assertEqual(restored.grid, self.psf.grid)
        self.assertEqual(restored.images.shape, self.psf.images.shape)
        self.assertEqual(restored.images.offset, self.psf.images.offset)

    def test_model_data_roundtrip(self):
        """A stitched PSF survives ``LsstScarletModelData`` serialization.

        The container serializes its PSF generically through the registry
        (``psf.to_data().as_dict()`` into the transport metadata, then
        ``PsfBaseData.from_dict(...).to_psf()`` back out), so registering
        ``StitchedPsfData`` on import is all it takes for a cell-coadd PSF to
        persist end to end -- the container needs no stitched-specific code.
        """
        model_psf = scl.ImagePsf(np.ones((1, KERNEL, KERNEL), dtype=np.float32) / KERNEL**2)
        model = LsstScarletModelData(
            isolated={},
            blends={},
            bands=BANDS,
            model_psf=model_psf,
            psf=self.psf,
        )
        restored = LsstScarletModelData.from_dict(json.loads(model.json()))
        self.assertIsInstance(restored.psf, ScarletStitchedPsf)
        self.assertEqual(set(restored.psf.images.keys()), set(self.psf.images.keys()))
        for index in self.psf.images.keys():
            np.testing.assert_array_almost_equal(
                restored.psf.images[index].data, self.psf.images[index].data
            )

    def test_unknown_psf_type_raises(self):
        """An unregistered ``psf_type`` raises a clear persistence error."""
        with self.assertRaises(scl.io.utils.PersistenceError):
            scl.io.PsfBaseData.from_dict({"psf_type": "not-a-real-type"})

    def test_empty_cells_rejected(self):
        """A stitched PSF requires at least one cell."""
        with self.assertRaises(ValueError):
            ScarletStitchedPsf(GridContainer(Index2D(x=GRID, y=GRID)), self.grid)

    def test_images_must_fit_grid(self):
        """Cells that do not fit on the grid are rejected.

        The constructor delegates to ``StitchedPsf._validate_args``, so a
        container whose index range exceeds the grid shape is refused exactly
        as the stack ``StitchedPsf`` would refuse it.
        """
        images = GridContainer(Index2D(x=GRID + 1, y=GRID + 1))
        images[Index2D(x=0, y=0)] = _normalized_kernel(self.rng)
        with self.assertRaises(ValueError):
            ScarletStitchedPsf(images, self.grid)


class ScarletStitchedPsfFactoryTestCase(lsst.utils.tests.TestCase):
    """Tests for building a stitched PSF from per-band cell-coadd PSFs."""

    def test_from_stitched_psf(self):
        """``from_stitched_psf`` stacks per-band cell images into one PSF."""
        sp_g = _gaussian_stitched(1.0)
        sp_r = _gaussian_stitched(1.5)
        psf = ScarletStitchedPsf.from_stitched_psf({"g": sp_g, "r": sp_r})

        self.assertIsInstance(psf, ScarletStitchedPsf)
        self.assertEqual(psf.bands, ("g", "r"))
        self.assertEqual(psf.shape, (FACTORY_KERNEL, FACTORY_KERNEL))
        self.assertEqual(len(psf.images), FACTORY_GRID * FACTORY_GRID)
        self.assertEqual(psf.grid, sp_g.grid)
        for index in psf.images.keys():
            cell = psf.images[index]
            self.assertEqual(cell.dtype, np.float32)
            self.assertEqual(cell.bands, ("g", "r"))
            np.testing.assert_array_almost_equal(cell.data[0], sp_g.images[index].array)
            np.testing.assert_array_almost_equal(cell.data[1], sp_r.images[index].array)

        # The assembled PSF is usable: convolution preserves the frame and a
        # cell PSF can be retrieved at a model location.
        rng = np.random.RandomState(7)
        image = Image(
            rng.rand(2, FACTORY_SIZE, FACTORY_SIZE).astype(np.float32),
            bands=("g", "r"),
            yx0=(0, 0),
        )
        convolved = psf.convolve(image)
        self.assertEqual(convolved.shape, image.shape)
        self.assertEqual(psf.get_image((1, 1)).shape, (2, FACTORY_KERNEL, FACTORY_KERNEL))

    def test_from_stitched_psf_grid_mismatch(self):
        """Per-band PSFs that disagree on the grid are rejected."""
        sp_g = _gaussian_stitched(1.0, cell=FACTORY_CELL)
        sp_r = _gaussian_stitched(1.5, cell=FACTORY_CELL + 1)
        with self.assertRaises(ValueError):
            ScarletStitchedPsf.from_stitched_psf({"g": sp_g, "r": sp_r})


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
