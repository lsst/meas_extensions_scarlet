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

import os
import unittest
import tempfile

import lsst.afw.image as afwImage
import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np
from lsst.afw.detection import Footprint
from lsst.afw.geom import SpanSet, Stencil
from lsst.afw.table import SourceCatalog
from lsst.daf.butler import DatasetType, StorageClass, FileDataset, DatasetRef
from lsst.daf.butler.tests import makeTestRepo, makeTestCollection
from lsst.geom import Point2I
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.extensions.scarlet.scarletDeblendTask import ScarletDeblendTask
from lsst.meas.extensions.scarlet.deconvolveExposureTask import DeconvolveExposureTask
from utils import initData

TESTDIR = os.path.abspath(os.path.dirname(__file__))


def printHierarchy(modelData: scl.io.ScarletModelData):
    """Print the hierarchy of the model data.

    This function is primarily for debugging purposes.

    Parameters
    ----------
    modelData : scl.io.ScarletModelData
        The model data containing blends and their sources.
    """
    print("Model Data Hierarchy:")
    for parentId, parentData in modelData.blends.items():
        print(f"Parent ID: {parentId}")
        print(f"nChildren: {len(parentData.children)}")
        for blendId, blendData in parentData.children.items():
            print(f"  Blend ID: {blendId}")
            print(f"    nSources: {len(blendData.sources)}")
            print(f"   Source Ids: {blendData.sources.keys()}")


class TestDeblend(lsst.utils.tests.TestCase):
    def setUp(self):
        # Set the random seed so that the noise field is unaffected
        np.random.seed(0)
        shape = (5, 100, 115)
        coords = [
            # blend
            (15, 25),
            (10, 30),
            (17, 38),
            # isolated source
            (85, 90),
        ]
        amplitudes = [
            # blend
            80,
            60,
            90,
            # isolated source
            20,
        ]
        result = initData(shape, coords, amplitudes)
        targetPsfImage, psfImages, images, channels, seds, morphs, psfs = result
        B, Ny, Nx = shape

        # Add some noise, otherwise the task will blow up due to
        # zero variance
        noise = 10 * (np.random.rand(*images.shape).astype(np.float32) - 0.5)
        images += noise

        self.bands = "grizy"
        _images = afwImage.MultibandMaskedImage.fromArrays(
            self.bands, images.astype(np.float32), None, noise**2
        )
        coadds = [
            afwImage.Exposure(img, dtype=img.image.array.dtype) for img in _images
        ]
        self.coadds = afwImage.MultibandExposure.fromExposures(self.bands, coadds)
        for b, coadd in enumerate(self.coadds):
            coadd.setPsf(psfs[b])

        # Initialize a Butler
        repo_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tempfile.TemporaryDirectory.cleanup, repo_dir)
        config = lsst.daf.butler.Config()
        config["datastore", "cls"] = "lsst.daf.butler.datastores.fileDatastore.FileDatastore"
        self.repo = makeTestRepo(repo_dir.name, config=config)
        storageClass = StorageClass(
            "ScarletModelData",
            pytype=scl.io.ScarletModelData,
            parameters=('blend_id',),
            delegate="lsst.meas.extensions.scarlet.io.ScarletModelDelegate",
        )
        datasetType = DatasetType(
            "scarlet_model_data",
            dimensions=(),
            storageClass=storageClass,
            universe=self.repo.dimensions,
        )
        self.repo.registry.registerDatasetType(datasetType)

    def _deblend(self):
        schema = SourceCatalog.Table.makeMinimalSchema()
        # Adjust config options to test skipping parents
        config = ScarletDeblendTask.ConfigClass()
        config.maxIter = 100
        config.maxFootprintArea = 1000
        config.maxNumberOfPeaks = 4
        config.catchFailures = False

        # Detect sources
        detectionTask = SourceDetectionTask(schema=schema)
        deblendTask = ScarletDeblendTask(schema=schema, config=config)
        table = SourceCatalog.Table.make(schema)
        detectionResult = detectionTask.run(table, self.coadds["r"])
        catalog = detectionResult.sources

        # Deconvolve the coadds
        deconvolvedCoadds = []
        deconvolveTask = DeconvolveExposureTask()
        for coadd in self.coadds:
            deconvolvedCoadd = deconvolveTask.run(coadd, catalog).deconvolved
            deconvolvedCoadds.append(deconvolvedCoadd)
        mDeconvolved = afwImage.MultibandExposure.fromExposures(self.bands, deconvolvedCoadds)

        # Add a footprint that is too large
        # src = catalog.addNew()
        # halfLength = int(np.ceil(np.sqrt(config.maxFootprintArea) + 1))
        # ss = SpanSet.fromShape(halfLength, Stencil.BOX, offset=(50, 50))
        # bigfoot = Footprint(ss)
        # bigfoot.addPeak(50, 50, 100)
        # src.setFootprint(bigfoot)

        # Add a footprint with too many peaks
        src = catalog.addNew()
        ss = SpanSet.fromShape(10, Stencil.BOX, offset=(75, 20))
        denseFoot = Footprint(ss)
        for n in range(config.maxNumberOfPeaks + 1):
            denseFoot.addPeak(70 + 2 * n, 15 + 2 * n, 10 * n)
        src.setFootprint(denseFoot)

        # Run the deblender
        result = deblendTask.run(self.coadds, mDeconvolved, catalog)

        # For debugging
        self.deblendTask = deblendTask
        return result, config

    def test_deblend_task(self):
        result, config = self._deblend()
        catalog = result.catalog
        modelData = result.modelData
        blendCatalog = result.blendCatalog
        observedPsf = modelData.metadata["psf"]
        modelPsf = modelData.metadata["model_psf"]

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

                mes.io.updateCatalogFootprints(
                    modelData,
                    catalog,
                    band=band,
                    imageForRedistribution=imageForRedistribution,
                    removeScarletData=False,
                    updateFluxColumns=True,
                )

                # Check that the number of deblended children is consistent
                parents = catalog[catalog["parent"] == 0]
                self.assertEqual(
                    np.sum(catalog["deblend_nChild"]), len(catalog) - len(parents)
                )

                for parent in parents:
                    children = catalog[catalog["parent"] == parent.get("id")]
                    # Check that nChild is set correctly
                    self.assertEqual(len(children), parent.get("deblend_nChild"))
                    for parentCol, childCol in config.columnInheritance.items():
                        np.testing.assert_array_equal(
                            parent.get(parentCol), children[childCol]
                        )

                    # Extract the parent blend data
                    parentBlendData = modelData.blends[parent.getId()]
                    parentFootprint = parent.getFootprint()
                    x0, y0 = parentFootprint.getBBox().getMin()
                    width, height = parentFootprint.getBBox().getDimensions()
                    yx0 = (y0, x0)

                    for child in children:
                        fp = child.getFootprint()
                        img = fp.extractImage(fill=0.0)
                        # Check that the flux at the center is correct.
                        # Note: this only works in this test image because the
                        # detected peak is in the same location as the
                        # scarlet peak.
                        # If the peak is shifted,
                        # the flux value will be correct but
                        # deblend_peak_center is not the correct location.
                        px = child.get("deblend_peak_center_x")
                        py = child.get("deblend_peak_center_y")
                        flux = img[Point2I(px, py)]
                        self.assertEqual(flux, child.get("deblend_peak_instFlux"))

                        self.assertEqual(child.get("deblend_nPeaks"), len(fp.peaks))

                        # Check that the peak positions match the catalog entry
                        peaks = fp.getPeaks()
                        self.assertEqual(px, peaks[0].getIx())
                        self.assertEqual(py, peaks[0].getIy())

                        # Load the data to check against the HeavyFootprint
                        blendData = parentBlendData.children[child["deblend_blendId"]]
                        # We need to set an observation in order to convolve
                        # the model.
                        modelBox = scl.Box((height, width), origin=(y0, x0))
                        observation = scl.Observation.empty(
                            bands=("dummy",),
                            psfs=observedPsf[bandIndex][None, :, :],
                            model_psf=modelPsf[None, :, :],
                            bbox=modelBox,
                            dtype=np.float32,
                        )
                        blend = mes.io.monochromaticDataToScarlet(
                            blendData=blendData,
                            bandIndex=bandIndex,
                            observation=observation,
                        )

                        # Get the scarlet model for the source
                        source = [
                            src for src in blend.sources if src.record_id == child.getId()
                        ][0]
                        self.assertEqual(source.center[1], px)
                        self.assertEqual(source.center[0], py)

                        if useFlux:
                            assert imageForRedistribution is not None
                            # Get the flux re-weighted model and test against
                            # the HeavyFootprint.
                            # The HeavyFootprint needs to be projected onto
                            # the image of the flux-redistributed model,
                            # since the HeavyFootprint
                            # may trim rows or columns.
                            _images = imageForRedistribution[
                                parentFootprint.getBBox()
                            ].image.array
                            blend.observation.images = scl.Image(
                                _images[None, :, :],
                                yx0=yx0,
                                bands=("dummy",),
                            )
                            blend.observation.weights = scl.Image(
                                parentFootprint.spans.asArray()[None, :, :],
                                yx0=yx0,
                                bands=("dummy",),
                            )
                            blend.conserve_flux()
                            model = source.flux_weighted_image.data[0]
                            my0, mx0 = source.flux_weighted_image.yx0
                            image = afwImage.ImageF(model, xy0=Point2I(mx0, my0))
                            fp.insert(image)
                            np.testing.assert_almost_equal(image.array, model)
                        else:
                            # Get the model for the source and test
                            # against the HeavyFootprint
                            bbox = fp.getBBox()
                            bbox = mes.utils.bboxToScarletBox(bbox)
                            model = blend.observation.convolve(
                                source.get_model().project(bbox=bbox), mode="real"
                            ).data[0]
                            np.testing.assert_almost_equal(img.array, model)

        # Check that all sources have the correct number of peaks
        for src in catalog:
            fp = src.getFootprint()
            self.assertEqual(len(fp.peaks), src.get("deblend_nPeaks"))

        # For now I've removed this test.
        # In order to test large footprints now,
        # we need to to actually create a large footprint in the
        # image, in the deconvolved space. So this will likely
        # need to be a separate test and I don't think that creating
        # it should hold up the rest of the deconovled deblender
        # implementation ticket (DM-47738).

        # Check that only the large footprint was flagged as too big
        # largeFootprint = np.zeros(len(blendCatalog), dtype=bool)
        # largeFootprint[2] = True
        # np.testing.assert_array_equal(largeFootprint,
        #     blendCatalog["deblend_parentTooBig"])

        # Check that only the dense footprint was flagged as too dense
        denseFootprint = np.zeros(len(blendCatalog), dtype=bool)
        denseFootprint[2] = True
        np.testing.assert_array_equal(denseFootprint, blendCatalog["deblend_tooManyPeaks"])

        # Check that only the appropriate parents were skipped
        skipped = denseFootprint
        np.testing.assert_array_equal(skipped, blendCatalog["deblend_skipped"])

    def _test_blend(self, blendData1, blendData2, model_psf, psf, bands):
        # Test that two ScarletBlendData objects are equal
        # up to machine precision.
        self.assertTupleEqual(blendData1.origin, blendData2.origin)
        self.assertEqual(len(blendData1.sources), len(blendData2.sources))

        # Test that the two blends are equal up to machine precision
        # once converted into scarlet lite Blend objects.
        blend1 = blendData1.minimal_data_to_blend(
            model_psf,
            psf,
            bands,
            dtype=np.float32,
        )
        blend2 = blendData2.minimal_data_to_blend(
            model_psf,
            psf,
            bands,
            dtype=np.float32,
        )
        np.testing.assert_almost_equal(blend1.get_model().data, blend2.get_model().data)

    def test_persistence(self):
        # Test that the model data is persisted correctly
        result, _ = self._deblend()
        modelData = result.modelData
        bands = modelData.metadata["bands"]
        butler = makeTestCollection(self.repo, uniqueId="test_run1")
        butler.put(modelData, "scarlet_model_data", dataId={})
        modelData2 = butler.get("scarlet_model_data", dataId={})
        model_psf = modelData.metadata["model_psf"][None, :, :]
        model_psf2 = modelData2.metadata["model_psf"][None, :, :]
        np.testing.assert_almost_equal(model_psf2, model_psf)
        psf = modelData.metadata["psf"]
        psf2 = modelData2.metadata["psf"]
        np.testing.assert_almost_equal(psf2, psf)
        self.assertEqual(len(modelData2.blends), len(modelData.blends))

        for parentId in modelData.blends.keys():
            nChildren = len(modelData.blends[parentId].children)
            self.assertEqual(nChildren, len(modelData2.blends[parentId].children))
            for blendId in modelData.blends[parentId].children:
                blendData1 = modelData.blends[parentId].children[blendId]
                blendData2 = modelData2.blends[parentId].children[blendId]
                self._test_blend(blendData1, blendData2, model_psf, psf, bands)

        # Test extracting a single blend
        modelData2 = butler.get("scarlet_model_data", dataId={}, parameters={"blend_id": parentId})
        self.assertEqual(len(modelData2.blends), 1)

        for blendId, blendData1 in modelData.blends[parentId].children.items():
            blendData2 = modelData2.blends[parentId].children[blendId]
            self._test_blend(blendData1, blendData2, model_psf, psf, bands)

        # Test extracting two blends
        modelData2 = butler.get("scarlet_model_data", dataId={}, parameters={"blend_id": [1, 2]})
        self.assertEqual(len(modelData2.blends), 2)
        for parentId in [1, 2]:
            parentData1 = modelData.blends[parentId]
            parentData2 = modelData2.blends[parentId]
            self.assertEqual(len(parentData1.children), len(parentData2.children))
            for blendId in parentData1.children.keys():
                blendData1 = parentData1.children[blendId]
                blendData2 = parentData2.children[blendId]
                self._test_blend(blendData1, blendData2, model_psf, psf, bands)

    def test_legacy_model(self):
        storageClass = StorageClass(
            "ScarletModelData",
            pytype=scl.io.ScarletModelData,
        )
        datasetType = DatasetType(
            "old_scarlet_model_data",
            dimensions=(),
            storageClass=storageClass,
            universe=self.repo.dimensions,
        )
        ref = DatasetRef(
            datasetType,
            run="test_ingestion",
            dataId={},
        )
        dataset = FileDataset(
            path=os.path.join(TESTDIR, "data", "v29_models.json"),
            formatter="lsst.daf.butler.formatters.json.JsonFormatter",
            refs=[ref],
        )

        # Ingest the legacy model into the butler
        butler = makeTestCollection(self.repo, uniqueId="ingestion")
        self.repo.registry.registerDatasetType(datasetType)
        butler.ingest(dataset)

        model = butler.get("old_scarlet_model_data", dataId={})
        self.assertEqual(len(model.blends), 2)

        test = butler.get("old_scarlet_model_data", dataId={}, parameters={"blend_id": 3495976385350991873})
        self.assertEqual(len(test.blends), 1)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
