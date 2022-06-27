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
from scarlet.bbox import Box
from scarlet.lite.measure import weight_sources

from lsst.geom import Point2I, Point2D
import lsst.utils.tests
import lsst.afw.image as afwImage
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.extensions.scarlet.scarletDeblendTask import ScarletDeblendTask, getFootprintMask
from lsst.meas.extensions.scarlet.source import bboxToScarletBox, scarletBoxToBBox
from lsst.meas.extensions.scarlet.io import dataToScarlet, DummyObservation
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
        _images = afwImage.MultibandMaskedImage.fromArrays(filters, images.astype(np.float32), None, noise**2)
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
        halfLength = int(np.ceil(np.sqrt(config.maxFootprintArea) + 1))
        ss = SpanSet.fromShape(halfLength, Stencil.BOX, offset=(50, 50))
        bigfoot = Footprint(ss)
        bigfoot.addPeak(50, 50, 100)
        src.setFootprint(bigfoot)

        # Add a footprint with too many peaks
        src = catalog.addNew()
        ss = SpanSet.fromShape(10, Stencil.BOX, offset=(75, 20))
        denseFoot = Footprint(ss)
        for n in range(config.maxNumberOfPeaks+1):
            denseFoot.addPeak(70+2*n, 15+2*n, 10*n)
        src.setFootprint(denseFoot)

        # Run the deblender
        catalog, modelData = deblendTask.run(coadds, catalog)

        # Attach the footprints in each band and compare to the full
        # data model. This is done in each band, both with and without
        # flux re-distribution to test all of the different possible
        # options of loading catalog footprints.
        for useFlux in [False, True]:
            for band in filters:
                bandIndex = filters.index(band)
                coadd = coadds[band]
                psfModel = coadd.getPsf()

                if useFlux:
                    redistributeImage = coadd.image
                else:
                    redistributeImage = None

                modelData.updateCatalogFootprints(
                    catalog,
                    band=band,
                    psfModel=psfModel,
                    redistributeImage=redistributeImage,
                    removeScarletData=False,
                )

                # Check that the number of deblended children is consistent
                parents = catalog[catalog["parent"] == 0]
                self.assertEqual(np.sum(catalog["deblend_nChild"]), len(catalog)-len(parents))

                # Check that the models have not been cleared
                # from the modelData
                self.assertEqual(len(modelData.blends), np.sum(~parents["deblend_skipped"]))

                for parent in parents:
                    children = catalog[catalog["parent"] == parent.get("id")]
                    # Check that nChild is set correctly
                    self.assertEqual(len(children), parent.get("deblend_nChild"))
                    # Check that parent columns are propagated
                    # to their children
                    for parentCol, childCol in config.columnInheritance.items():
                        np.testing.assert_array_equal(parent.get(parentCol), children[childCol])

                children = catalog[catalog["parent"] != 0]
                for child in children:
                    fp = child.getFootprint()
                    img = heavyFootprintToImage(fp, fill=0.0)
                    # Check that the flux at the center is correct.
                    # Note: this only works in this test image because the
                    # detected peak is in the same location as the
                    # scarlet peak.
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

                    # Load the data to check against the HeavyFootprint
                    blendData = modelData.blends[child["parent"]]
                    blend = dataToScarlet(
                        blendData=blendData,
                        nBands=1,
                        bandIndex=bandIndex,
                    )
                    # We need to set an observation in order to convolve
                    # the model.
                    position = Point2D(*blendData.psfCenter)
                    _psfs = coadds[band].getPsf().computeKernelImage(position).array[None, :, :]
                    modelBox = Box((1,) + tuple(blendData.extent[::-1]), origin=(0, 0, 0))
                    blend.observation = DummyObservation(
                        psfs=_psfs,
                        model_psf=modelData.psf[None, :, :],
                        bbox=modelBox,
                        dtype=np.float32,
                    )

                    # Get the scarlet model for the source
                    source = [src for src in blend.sources if src.recordId == child.getId()][0]
                    parentBox = catalog.find(child["parent"]).getFootprint().getBBox()
                    self.assertEqual(source.center[1], px - parentBox.getMinX())
                    self.assertEqual(source.center[0], py - parentBox.getMinY())

                    if useFlux:
                        # Get the flux re-weighted model and test against
                        # the HeavyFootprint.
                        # The HeavyFootprint needs to be projected onto
                        # the image of the flux-redistributed model,
                        # since the HeavyFootprint may trim rows or columns.
                        parentFootprint = catalog[catalog["id"] == child["parent"]][0].getFootprint()
                        blend.observation.images = redistributeImage[parentFootprint.getBBox()].array
                        blend.observation.images = blend.observation.images[None, :, :]
                        blend.observation.weights = ~getFootprintMask(parentFootprint)[None, :, :]
                        weight_sources(blend)
                        model = source.flux[0]
                        bbox = scarletBoxToBBox(source.flux_box, Point2I(*blendData.xy0))
                        image = afwImage.ImageF(model, xy0=bbox.getMin())
                        fp.insert(image)
                        np.testing.assert_almost_equal(image.array, model)
                    else:
                        # Get the model for the source and test
                        # against the HeavyFootprint
                        bbox = fp.getBBox()
                        bbox = bboxToScarletBox(1, bbox, Point2I(*blendData.xy0))
                        model = blend.observation.convolve(source.get_model(bbox=bbox))[0]
                        np.testing.assert_almost_equal(img.image.array, model)

        # Check that all sources have the correct number of peaks
        for src in catalog:
            fp = src.getFootprint()
            self.assertEqual(len(fp.peaks), src.get("deblend_nPeaks"))

        # Check that only the large footprint was flagged as too big
        largeFootprint = np.zeros(len(catalog), dtype=bool)
        largeFootprint[2] = True
        np.testing.assert_array_equal(largeFootprint, catalog["deblend_parentTooBig"])

        # Check that only the dense footprint was flagged as too dense
        denseFootprint = np.zeros(len(catalog), dtype=bool)
        denseFootprint[3] = True
        np.testing.assert_array_equal(denseFootprint, catalog["deblend_tooManyPeaks"])

        # Check that only the appropriate parents were skipped
        skipped = largeFootprint | denseFootprint
        np.testing.assert_array_equal(skipped, catalog["deblend_skipped"])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
