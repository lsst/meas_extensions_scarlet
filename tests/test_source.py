# This file is part of lsst.scarlet.lite.
#
# Developed for the LSST Data Management System.
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
# This product includes software developed by the LSST Project
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from unittest import TestCase

import numpy as np

import lsst.afw.image as afwImage
import lsst.geom as geom
import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
from lsst.afw.detection import Footprint, PeakTable
from lsst.afw.geom import SpanSet


class ScarletTestCase(TestCase):
    """A base TestCase for scarlet tests.
    """
    def setUp(self) -> None:
        super().setUp()
        self.bands = tuple("grizy")
        peak = (27, 32)
        bbox = scl.Box((15, 15), (20, 25))
        morph = scl.utils.integrated_circular_gaussian(sigma=0.8).astype(np.float32)
        spectrum = np.arange(5, dtype=np.float32)
        model = morph[None, :, :] * spectrum[:, None, None]
        model_image = scl.Image(model, yx0=bbox.origin, bands=self.bands)
        self.component = scl.component.CubeComponent(model=model_image, peak=peak)

    def test_constructor(self):
        source = mes.source.IsolatedSource(
            model=self.component._model,
            peak=self.component.peak,
        )

        np.testing.assert_array_equal(
            source.component._model.data,
            self.component._model.data,
        )
        self.assertEqual(source.component.peak, self.component.peak)

    def test_copy(self):
        source = mes.source.IsolatedSource(
            model=self.component._model,
            peak=self.component.peak,
        )
        source_copy = source.copy()

        self.assertIsNot(source_copy, source)
        self.assertIs(
            source_copy.component._model.data,
            source.component._model.data,
        )
        self.assertEqual(source_copy.component.peak, source.component.peak)

    def test_deep_copy(self):
        source = mes.source.IsolatedSource(
            model=self.component._model,
            peak=self.component.peak,
        )
        source_copy = source.copy(deep=True)

        self.assertTupleEqual(source_copy.component.peak, source.component.peak)

        np.testing.assert_array_equal(
            source_copy.component._model.data,
            source.component._model.data,
        )
        self.assertIsNot(
            source_copy.component._model.data,
            source.component._model.data,
        )

        with self.assertRaises(AssertionError):
            source_copy.component._model._data -= 1
            np.testing.assert_array_equal(
                source_copy.component._model.data,
                source.component._model.data,
            )

    def test_slice(self):
        source = mes.source.IsolatedSource(self.component._model, self.component.peak)
        source_sliced = source["g":"r"]
        self.assertTupleEqual(source_sliced.bands, ("g", "r"))
        np.testing.assert_array_equal(
            source_sliced.get_model().data,
            source.get_model().data[:2],
        )

    def test_reorder(self):
        source = mes.source.IsolatedSource(self.component._model, self.component.peak)
        indices = ("i", "g", "r")
        source_reordered = source[indices]
        self.assertTupleEqual(source_reordered.bands, indices)
        np.testing.assert_array_equal(
            source_reordered.get_model().data,
            source.get_model().data[[2, 0, 1]],
        )

        source_reordered = source["igr"]
        self.assertTupleEqual(source_reordered.bands, indices)
        np.testing.assert_array_equal(
            source_reordered.get_model().data,
            source.get_model().data[[2, 0, 1]],
        )

    def test_subset(self):
        source = mes.source.IsolatedSource(self.component._model, self.component.peak)
        source_subset = source[("r",)]
        self.assertTupleEqual(source_subset.bands, ("r",))
        np.testing.assert_array_equal(
            source_subset.get_model().data,
            source.get_model().data[1:2],
        )

    def test_indexing_errors(self):
        source = mes.source.IsolatedSource(self.component._model, self.component.peak)
        with self.assertRaises(IndexError):
            # "x" is not an a band in the model
            source["x"]

        with self.assertRaises(IndexError):
            # "x" is not an a band in the model
            source["r":"x"]

        with self.assertRaises(IndexError):
            # "x" is not an a band in the model
            source["x":"i"]

        with self.assertRaises(IndexError):
            # "x" is not an a band in the model
            source["g", "x", "i"]

        with self.assertRaises(IndexError):
            # The box doesn't overlap with the model
            source[scl.Box((0, 0), (10, 10))]

        with self.assertRaises(IndexError):
            # Users must provide a Box, not a tuple, for spatial dimensions
            source[:, 10:20, 10:20]

        with self.assertRaises(IndexError):
            # Users must provide a Box, not a slice, for spatial dimensions
            source[1:]

        with self.assertRaises(IndexError):
            # Users must provide a Box, not an int, for spatial dimensions
            source[1]

        with self.assertRaises(IndexError):
            # Users must provide a Box, not a tuple, for spatial dimensions
            source[0, 1]

    def test_from_footprint_roundtrip(self):
        """``IsolatedSource.from_footprint`` produces a source whose
        peak and bbox match the input afw Footprint.

        Uses a circular 7×7 mask so the bbox is a real bounding box
        rather than coincident with the footprint shape, and so the
        peak position is non-trivially distinct from the bbox origin.
        """
        bands = tuple("gri")
        h, w = 7, 7
        y0, x0 = 5, 3  # bbox origin in (y, x)

        circle = scl.utils.get_circle_mask(h, dtype=np.int32)
        afw_mask = afwImage.Mask(circle, xy0=geom.Point2I(x0, y0))
        footprint = Footprint(SpanSet.fromMask(afw_mask), PeakTable.makeMinimalSchema())
        peak_y, peak_x = y0 + h // 2, x0 + w // 2
        footprint.addPeak(peak_x, peak_y, 100.0)

        # 3-band 20×20 coadd of ones; ``from_footprint`` will mask
        # the coadd's data with the footprint spans inside the bbox.
        image_shape = (3, 20, 20)
        masked = afwImage.MultibandMaskedImage.fromArrays(
            bands,
            np.ones(image_shape, dtype=np.float32),
            None,
            np.ones(image_shape, dtype=np.float32),
        )
        coadds = [
            afwImage.Exposure(img, dtype=img.image.array.dtype) for img in masked
        ]
        mCoadd = afwImage.MultibandExposure.fromExposures(bands, coadds)

        source = mes.source.IsolatedSource.from_footprint(
            footprint, mCoadd, dtype=np.float32
        )

        self.assertTupleEqual(source.peak, (peak_y, peak_x))
        self.assertTupleEqual(source.bbox.origin, (y0, x0))
        self.assertTupleEqual(source.bbox.shape, (h, w))

    def test_to_data_from_data_roundtrip(self):
        """``to_data`` → ``IsolatedSourceData.to_source`` against an
        observation matching the source's model recovers the source's
        model data, peak, and bbox origin.

        Uses a circular morph so ``span_array`` is not trivially
        all-True; reconstruction must actually mask by the span set.
        """
        bands = tuple("gri")
        morph = scl.utils.get_circle_mask(15, dtype=np.float32)
        spectrum = np.arange(1, 1 + len(bands), dtype=np.float32)
        model_data = morph[None, :, :] * spectrum[:, None, None]
        peak = (27, 32)
        bbox = scl.Box((15, 15), (20, 25))
        model_image = scl.Image(model_data, yx0=bbox.origin, bands=bands)

        source = mes.source.IsolatedSource(model=model_image, peak=peak)
        data = source.to_data()

        self.assertTupleEqual(data.origin, bbox.origin)
        self.assertTupleEqual(data.peak, peak)
        np.testing.assert_array_equal(data.span_array, morph > 0)

        # Round-trip via to_source: build an Observation whose images
        # equal the source model (a delta PSF makes the observation
        # bypass any convolution).
        psfs = np.zeros((len(bands), 1, 1), dtype=np.float32)
        psfs[:, 0, 0] = 1.0
        observation = scl.Observation.empty(
            bands=bands,
            psfs=psfs,
            model_psf=psfs[:1],
            bbox=bbox,
            dtype=np.float32,
        )
        observation.images = model_image

        source2 = data.to_source(observation)

        self.assertTupleEqual(source2.peak, peak)
        self.assertTupleEqual(source2.bbox.origin, bbox.origin)
        np.testing.assert_array_equal(source2.get_model().data, model_data)
