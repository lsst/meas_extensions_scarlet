
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

import unittest

import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.extensions.scarlet import ScarletDeblendTask
from lsst.afw.table import SourceCatalog
from lsst.afw.detection import MultibandFootprint
from lsst.afw.image import Image, MultibandImage

from utils import initData


class TestDeblend(lsst.utils.tests.TestCase):
    def test_task(self):
        # Set the random seed so that the noise field is unaffected
        np.random.seed(0)
        # Test that executing the deblend task works
        # In the future we can have more detailed tests,
        # but for now this at least ensures that the task isn't broken
        shape = (5, 31, 55)
        coords = [(15, 25), (10, 30), (17, 38)]
        amplitudes = [80, 60, 90]
        result = initData(shape, coords, amplitudes)
        targetPsfImage, psfImages, images, channels, seds, morphs, targetPsf, psfs = result
        B, Ny, Nx = shape

        # Add some noise, otherwise the task will blow up due to
        # zero variance
        noise = 10*(np.random.rand(*images.shape)-.5)
        images += noise

        filters = "grizy"
        _images = afwImage.MultibandMaskedImage.fromArrays(filters, images.astype(np.float32), None,
                                                           noise.astype(np.float32))
        coadds = [afwImage.Exposure(img, dtype=img.image.array.dtype) for img in _images]
        coadds = afwImage.MultibandExposure.fromExposures(filters, coadds)
        for b, coadd in enumerate(coadds):
            coadd.setPsf(psfs[b])

        schema = SourceCatalog.Table.makeMinimalSchema()

        detectionTask = SourceDetectionTask(schema=schema)
        config = ScarletDeblendTask.ConfigClass()
        config.maxIter = 200
        deblendTask = ScarletDeblendTask(schema=schema, config=config)

        table = SourceCatalog.Table.make(schema)
        detectionResult = detectionTask.run(table, coadds["r"])
        catalog = detectionResult.sources
        self.assertEqual(len(catalog), 1)
        _, result = deblendTask.run(coadds, catalog)

        # Changes to the internal workings of scarlet will change these results
        # however we include these tests just to track changes
        parent = result["r"][0]
        self.assertEqual(parent["iterations"], 11)
        self.assertEqual(parent["deblend_nChild"], 3)

        heavies = []
        for k in range(1, len(result["g"])):
            heavy = MultibandFootprint(coadds.filters, [result[b][k].getFootprint() for b in filters])
            heavies.append(heavy)

        seds = np.array([heavy.getImage(fill=0).image.array.sum(axis=(1, 2)) for heavy in heavies])
        true_seds = np.array([
            [[1665.726318359375, 1745.5401611328125, 1525.91796875, 997.3868408203125, 0.0],
             [767.100341796875, 1057.0374755859375, 1312.89111328125, 1694.7535400390625, 2069.294921875],
             [8.08012580871582, 879.344970703125, 2246.90087890625, 4212.82470703125, 6987.0849609375]]
        ])

        self.assertFloatsAlmostEqual(true_seds, seds, rtol=1e-8, atol=1e-8)

        bbox = parent.getFootprint().getBBox()
        data = coadds[:, bbox]
        model = MultibandImage.fromImages(coadds.filters, [
            Image(bbox, dtype=np.float32)
            for b in range(len(filters))
        ])
        for heavy in heavies:
            model[:, heavy.getBBox()].array += heavy.getImage(fill=0).image.array

        residual = data.image.array - model.array
        self.assertFloatsAlmostEqual(np.abs(residual).sum(), 11601.3867187500)
        self.assertFloatsAlmostEqual(np.max(np.abs(residual)), 56.1048278809, rtol=1e-8, atol=1e-8)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
