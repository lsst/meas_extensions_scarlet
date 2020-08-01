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


from utils import initData


class TestDeblend(lsst.utils.tests.TestCase):
    def test_deblend_task(self):
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
        noise = 10*(np.random.rand(*images.shape).astype(np.float32)-.5)
        images += noise

        filters = "grizy"
        _images = afwImage.MultibandMaskedImage.fromArrays(filters, images.astype(np.float32), None, noise)
        coadds = [afwImage.Exposure(img, dtype=img.image.array.dtype) for img in _images]
        coadds = afwImage.MultibandExposure.fromExposures(filters, coadds)
        for b, coadd in enumerate(coadds):
            coadd.setPsf(psfs[b])

        schema = SourceCatalog.Table.makeMinimalSchema()

        detectionTask = SourceDetectionTask(schema=schema)
        config = ScarletDeblendTask.ConfigClass()
        config.maxIter = 300
        deblendTask = ScarletDeblendTask(schema=schema, config=config)

        table = SourceCatalog.Table.make(schema)
        detectionResult = detectionTask.run(table, coadds["r"])
        catalog = detectionResult.sources
        self.assertEqual(len(catalog), 1)
        _, result = deblendTask.run(coadds, catalog)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
