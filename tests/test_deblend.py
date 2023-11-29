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
from numpy.testing import assert_almost_equal

from lsst.geom import Point2I, Point2D
import lsst.utils.tests
import lsst.afw.image as afwImage
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.extensions.scarlet.scarletDeblendTask import ScarletDeblendTask
from lsst.meas.extensions.scarlet.utils import bboxToScarletBox, scarletBoxToBBox
from lsst.meas.extensions.scarlet.io import monochromaticDataToScarlet, updateCatalogFootprints
import lsst.scarlet.lite as scl
from lsst.afw.table import SourceCatalog
from lsst.afw.detection import Footprint
from lsst.afw.geom import SpanSet, Stencil

from utils import initData


class TestDeblend(lsst.utils.tests.TestCase):
    def setUp(self):
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
        targetPsfImage, psfImages, images, channels, seds, morphs, psfs = result
        B, Ny, Nx = shape

        # Add some noise, otherwise the task will blow up due to
        # zero variance
        noise = 10*(np.random.rand(*images.shape).astype(np.float32)-.5)
        images += noise

        self.bands = "grizy"
        _images = afwImage.MultibandMaskedImage.fromArrays(
            self.bands,
            images.astype(np.float32),
            None,
            noise**2
        )
        coadds = [afwImage.Exposure(img, dtype=img.image.array.dtype) for img in _images]
        self.coadds = afwImage.MultibandExposure.fromExposures(self.bands, coadds)
        for b, coadd in enumerate(self.coadds):
            coadd.setPsf(psfs[b])

    def _deblend(self, version):
        schema = SourceCatalog.Table.makeMinimalSchema()
        # Adjust config options to test skipping parents
        config = ScarletDeblendTask.ConfigClass()
        config.maxIter = 100
        config.maxFootprintArea = 1000
        config.maxNumberOfPeaks = 4
        config.catchFailures = False
        config.version = version

        # Detect sources
        detectionTask = SourceDetectionTask(schema=schema)
        deblendTask = ScarletDeblendTask(schema=schema, config=config)
        table = SourceCatalog.Table.make(schema)
        detectionResult = detectionTask.run(table, self.coadds["r"])
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
        catalog, modelData = deblendTask.run(self.coadds, catalog)
        return catalog, modelData, config

    def test_deblend_task(self):
        catalog, modelData, config = self._deblend("lite")

        # Attach the footprints in each band and compare to the full
        # data model. This is done in each band, both with and without
        # flux re-distribution to test all of the different possible
        # options of loading catalog footprints.
        for useFlux in [False, True]:
            for band in self.bands:
                bandIndex = self.bands.index(band)
                coadd = self.coadds[band]

                if useFlux:
                    imageForRedistribution = coadd
                else:
                    imageForRedistribution = None

                updateCatalogFootprints(
                    modelData,
                    catalog,
                    band=band,
                    imageForRedistribution=imageForRedistribution,
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
                    img = fp.extractImage(fill=0.0)
                    # Check that the flux at the center is correct.
                    # Note: this only works in this test image because the
                    # detected peak is in the same location as the
                    # scarlet peak.
                    # If the peak is shifted, the flux value will be correct
                    # but deblend_peak_center is not the correct location.
                    px = child.get("deblend_peak_center_x")
                    py = child.get("deblend_peak_center_y")
                    flux = img[Point2I(px, py)]
                    self.assertEqual(flux, child.get("deblend_peak_instFlux"))

                    # Check that the peak positions match the catalog entry
                    peaks = fp.getPeaks()
                    self.assertEqual(px, peaks[0].getIx())
                    self.assertEqual(py, peaks[0].getIy())

                    # Load the data to check against the HeavyFootprint
                    blendData = modelData.blends[child["parent"]]
                    # We need to set an observation in order to convolve
                    # the model.
                    position = Point2D(*blendData.psf_center[::-1])
                    _psfs = self.coadds[band].getPsf().computeKernelImage(position).array[None, :, :]
                    modelBox = scl.Box(blendData.shape, origin=blendData.origin)
                    observation = scl.Observation.empty(
                        bands=("dummy", ),
                        psfs=_psfs,
                        model_psf=modelData.psf[None, :, :],
                        bbox=modelBox,
                        dtype=np.float32,
                    )
                    blend = monochromaticDataToScarlet(
                        blendData=blendData,
                        bandIndex=bandIndex,
                        observation=observation,
                    )
                    # The stored PSF should be the same as the calculated one
                    assert_almost_equal(blendData.psf[bandIndex:bandIndex+1], _psfs)

                    # Get the scarlet model for the source
                    source = [src for src in blend.sources if src.record_id == child.getId()][0]
                    self.assertEqual(source.center[1], px)
                    self.assertEqual(source.center[0], py)

                    if useFlux:
                        # Get the flux re-weighted model and test against
                        # the HeavyFootprint.
                        # The HeavyFootprint needs to be projected onto
                        # the image of the flux-redistributed model,
                        # since the HeavyFootprint may trim rows or columns.
                        parentFootprint = catalog[catalog["id"] == child["parent"]][0].getFootprint()
                        _images = imageForRedistribution[parentFootprint.getBBox()].image.array
                        blend.observation.images = scl.Image(
                            _images[None, :, :],
                            yx0=blendData.origin,
                            bands=("dummy", ),
                        )
                        blend.observation.weights = scl.Image(
                            parentFootprint.spans.asArray()[None, :, :],
                            yx0=blendData.origin,
                            bands=("dummy", ),
                        )
                        blend.conserve_flux()
                        model = source.flux_weighted_image.data[0]
                        bbox = scarletBoxToBBox(source.flux_weighted_image.bbox)
                        image = afwImage.ImageF(model, xy0=bbox.getMin())
                        fp.insert(image)
                        np.testing.assert_almost_equal(image.array, model)
                    else:
                        # Get the model for the source and test
                        # against the HeavyFootprint
                        bbox = fp.getBBox()
                        bbox = bboxToScarletBox(bbox)
                        model = blend.observation.convolve(
                            source.get_model().project(bbox=bbox), mode="real"
                        ).data[0]
                        np.testing.assert_almost_equal(img.array, model)

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

    def test_continuity(self):
        """This test ensures that lsst.scarlet.lite gives roughly the same
        result as scarlet.lite

        TODO: This test can be removed once the deprecated scarlet.lite
        module is removed from the science pipelines.
        """
        oldCatalog, oldModelData, oldConfig = self._deblend("old_lite")
        catalog, modelData, config = self._deblend("lite")

        # Ensure that the deblender used different versions
        self.assertEqual(oldConfig.version, "old_lite")
        self.assertEqual(config.version, "lite")

        # Check that the PSF and other properties are the same
        assert_almost_equal(oldModelData.psf, modelData.psf)
        self.assertTupleEqual(tuple(oldModelData.blends.keys()), tuple(modelData.blends.keys()))

        # Make sure that the sources have the same IDs
        for i in range(len(catalog)):
            self.assertEqual(catalog[i]["id"], oldCatalog[i]["id"])

        for blendId in modelData.blends.keys():
            oldBlendData = oldModelData.blends[blendId]
            blendData = modelData.blends[blendId]

            # Check that blend properties are the same
            self.assertTupleEqual(oldBlendData.origin, blendData.origin)
            self.assertTupleEqual(oldBlendData.shape, blendData.shape)
            self.assertTupleEqual(oldBlendData.bands, blendData.bands)
            self.assertTupleEqual(oldBlendData.psf_center, blendData.psf_center)
            self.assertTupleEqual(tuple(oldBlendData.sources.keys()), tuple(blendData.sources.keys()))
            assert_almost_equal(oldBlendData.psf, blendData.psf)

            for sourceId in blendData.sources.keys():
                oldSourceData = oldBlendData.sources[sourceId]
                sourceData = blendData.sources[sourceId]
                # Check that source properties are the same
                self.assertEqual(len(oldSourceData.components), 0)
                self.assertEqual(len(sourceData.components), 0)
                self.assertEqual(
                    len(oldSourceData.factorized_components),
                    len(sourceData.factorized_components)
                )

                for c in range(len(sourceData.factorized_components)):
                    oldComponentData = oldSourceData.factorized_components[c]
                    componentData = sourceData.factorized_components[c]
                    # Check that component properties are the same
                    self.assertTupleEqual(oldComponentData.peak, componentData.peak)
                    self.assertTupleEqual(
                        tuple(oldComponentData.peak[i]-oldComponentData.shape[i]//2 for i in range(2)),
                        oldComponentData.origin,
                    )


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
