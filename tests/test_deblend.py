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

from lsst.geom import Point2I
import lsst.utils.tests
import lsst.afw.image as afwImage
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.extensions.scarlet import ScarletDeblendTask
from lsst.afw.table import SourceCatalog
from lsst.afw.detection import Footprint
from lsst.afw.detection.multiband import heavyFootprintToImage
from lsst.afw.geom import SpanSet, Stencil

from utils import initData


class TestDeblend(lsst.utils.tests.TestCase):
    def test_deblend_task(self):
        # Set the random seed so that the noise field is unaffected
        np.random.seed(0)
        shape = (5, 100, 115)
        coords = [
            # blend
            (15, 25), (10, 30), (17, 38),
            # isolated source
            (85, 90),
        ]
        amplitudes = [
            # blend
            80, 60, 90,
            # isolated source
            20,
        ]
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

        # Adjust config options to test skipping parents
        config = ScarletDeblendTask.ConfigClass()
        config.maxIter = 100
        config.maxFootprintArea = 1000
        config.maxNumberOfPeaks = 4
        deblendTask = ScarletDeblendTask(schema=schema, config=config)

        table = SourceCatalog.Table.make(schema)
        detectionResult = detectionTask.run(table, coadds["r"])
        catalog = detectionResult.sources

        # Add a footprint that is too large
        src = catalog.addNew()
        ss = SpanSet.fromShape(20, Stencil.BOX, offset=(50, 50))
        bigfoot = Footprint(ss)
        bigfoot.addPeak(50, 50, 100)
        src.setFootprint(bigfoot)

        # Add a footprint with too many peaks
        src = catalog.addNew()
        ss = SpanSet.fromShape(10, Stencil.BOX, offset=(75, 20))
        denseFoot = Footprint(ss)
        for n in range(5):
            denseFoot.addPeak(70+2*n, 15+2*n, 10*n)
        src.setFootprint(denseFoot)

        # Run the deblender
        result = deblendTask.run(coadds, catalog)

        # Make sure that the catalogs have the same sources in all bands,
        # and check that band-independent columns are equal
        bandIndependentColumns = [
            "id",
            "parent",
            "deblend_nPeaks",
            "deblend_nChild",
            "deblend_peak_center_x",
            "deblend_peak_center_y",
            "deblend_runtime",
            "deblend_iterations",
            "deblend_logL",
            "deblend_spectrumInitFlag",
            "deblend_blendConvergenceFailedFlag",
        ]
        self.assertEqual(len(filters), len(result))
        ref = result[filters[0]]
        for f in filters[1:]:
            for col in bandIndependentColumns:
                np.testing.assert_array_equal(result[f][col], ref[col])

        # Check that other columns are consistent
        for f, _catalog in result.items():
            parents = _catalog[_catalog["parent"] == 0]
            # Check that the number of deblended children is consistent
            self.assertEqual(np.sum(_catalog["deblend_nChild"]), len(_catalog)-len(parents))

            for parent in parents:
                children = _catalog[_catalog["parent"] == parent.get("id")]
                # Check that nChild is set correctly
                self.assertEqual(len(children), parent.get("deblend_nChild"))
                # Check that parent columns are propagated to their children
                for parentCol, childCol in config.columnInheritance.items():
                    np.testing.assert_array_equal(parent.get(parentCol), children[childCol])

            children = _catalog[_catalog["parent"] != 0]
            for child in children:
                fp = child.getFootprint()
                img = heavyFootprintToImage(fp)
                # Check that the flux at the center is correct.
                # Note: this only works in this test image because the
                # detected peak is in the same location as the scarlet peak.
                # If the peak is shifted, the flux value will be correct
                # but deblend_peak_center is not the correct location.
                px = child.get("deblend_peak_center_x")
                py = child.get("deblend_peak_center_y")
                flux = img.image[Point2I(px, py)]
                self.assertEqual(flux, child.get("deblend_peak_instFlux"))

                # Check that the peak positions match the catalog entry
                peaks = fp.getPeaks()
                self.assertEqual(px, peaks[0].getIx())
                self.assertEqual(py, peaks[0].getIy())

            # Check that all sources have the correct number of peaks
            for src in _catalog:
                fp = src.getFootprint()
                self.assertEqual(len(fp.peaks), src.get("deblend_nPeaks"))

            # Check that only the large foorprint was flagged as too big
            largeFootprint = np.zeros(len(_catalog), dtype=bool)
            largeFootprint[2] = True
            np.testing.assert_array_equal(largeFootprint, _catalog["deblend_parentTooBig"])

            # Check that only the dense foorprint was flagged as too dense
            denseFootprint = np.zeros(len(_catalog), dtype=bool)
            denseFootprint[3] = True
            np.testing.assert_array_equal(denseFootprint, _catalog["deblend_tooManyPeaks"])

            # Check that only the appropriate parents were skipped
            skipped = largeFootprint | denseFootprint
            np.testing.assert_array_equal(skipped, _catalog["deblend_skipped"])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
