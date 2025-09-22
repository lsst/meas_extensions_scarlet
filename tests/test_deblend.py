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
from lsst.afw.detection import GaussianPsf
from lsst.afw.table import SourceCatalog, SourceTable, SchemaMapper
from lsst.daf.butler import Config, DatasetType, StorageClass, FileDataset, DatasetRef
from lsst.daf.butler.tests import makeTestRepo, makeTestCollection
from lsst.geom import Point2I
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.extensions.scarlet.scarletDeblendTask import ScarletDeblendTask
from lsst.meas.extensions.scarlet.deconvolveExposureTask import DeconvolveExposureTask
from lsst.pipe.base import Struct
from utils import initData, SersicModel, PsfModel

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class TestDeblend(lsst.utils.tests.TestCase):
    def setUp(self):
        self.modelPsf = scl.utils.integrated_circular_gaussian(sigma=0.8).astype(np.float32)
        psfRadius = 20
        psfShape = (2 * psfRadius + 1, 2 * psfRadius + 1)
        self.psfs = [
            GaussianPsf(psfShape[1], psfShape[0], 1.0),
            GaussianPsf(psfShape[1], psfShape[0], 1.2),
            GaussianPsf(psfShape[1], psfShape[0], 1.4),
        ]
        self.imagePsf = np.asarray(
            [psf.computeImage(psf.getAveragePosition()).array for psf in self.psfs]
        ).astype(np.float32)
        self.imagePsf /= self.imagePsf.sum(axis=(1, 2))[:, None, None]
        self.bands = tuple("gri")

        self.models = [
            # Isolated source
            PsfModel(
                center=(30, 15),
                spectrum=np.array([8, 2, 1]),
                bands=self.bands,
            ),
            # Two source blend
            SersicModel(
                center=(40, 20),
                major=5,
                minor=2,
                radius=15,
                theta=-np.pi/4,
                n=1,
                spectrum=np.array([2, 4, 8]),
                bands=self.bands,
            ),
            PsfModel(
                center=(12, 20),
                spectrum=np.array([1, 2, 8]),
                bands=self.bands,
            ),
            # 3 source blend
            SersicModel(
                center=(25, 70),
                major=5,
                minor=2,
                radius=20,
                theta=np.pi/48,
                n=1,
                spectrum=np.array([2, 5, 8]),
                bands=self.bands,
            ),
            PsfModel(
                center=(32, 60),
                spectrum=np.array([1, 2, 8]),
                bands=self.bands,
            ),
            PsfModel(
                center=(16, 80),
                spectrum=np.array([8, 2, 1]),
                bands=self.bands,
            ),
            # Large blend
            SersicModel(
                center=(70, 70),
                major=5,
                minor=2,
                radius=25,
                theta=0,
                n=1,
                spectrum=np.array([2, 10, 18]),
                bands=self.bands,
            ),
            SersicModel(
                center=(85, 85),
                major=5,
                minor=2,
                radius=25,
                theta=np.pi/2,
                n=1,
                spectrum=np.array([5, 10, 20]),
                bands=self.bands,
            ),
        ]

    def scarlet_image_to_exposure(
        self,
        image: scl.Image,
        noise: np.ndarray,
    ) -> afwImage.MultibandExposure:
        masked_image = afwImage.MultibandMaskedImage.fromArrays(
            image.bands, image.data, None, noise**2
        )
        coadds = [
            afwImage.Exposure(img, dtype=img.image.array.dtype) for img in masked_image
        ]
        mCoadd = afwImage.MultibandExposure.fromExposures(image.bands, coadds)
        for b, coadd in enumerate(mCoadd):
            coadd.setPsf(self.psfs[b])
        return mCoadd

    def initialize_data(
        self,
        models,
        deconvolveConfig=None,
        deblendConfig=None,
        doDetect: bool = True,
    ):
        if deconvolveConfig is None:
            deconvolveConfig = DeconvolveExposureTask.ConfigClass()
        if deblendConfig is None:
            deblendConfig = ScarletDeblendTask.ConfigClass()
        # Generate the data for the test
        deconvolved, convolved = initData(models, self.modelPsf, self.imagePsf)
        # Set the random seed so that the noise field is unaffected
        # and add noise to the image
        np.random.seed(0)
        noise = 0.05 * (np.random.rand(*convolved.shape).astype(np.float32) - 0.5)
        noisyImage = convolved.copy()
        noisyImage._data += noise
        # Create the multiband coadd
        mCoadd = self.scarlet_image_to_exposure(noisyImage, noise)
        # Initialze tasks
        inputSchema = SourceTable.makeMinimalSchema()
        table = SourceTable.make(inputSchema)
        detectionTask = SourceDetectionTask(schema=inputSchema)
        schemaMapper = SchemaMapper(inputSchema)
        schemaMapper.addMinimalSchema(inputSchema)
        schema = schemaMapper.getOutputSchema()
        deconvolveTask = DeconvolveExposureTask(config=deconvolveConfig)
        deblendTask = ScarletDeblendTask(schema=schema, config=deblendConfig)

        result = Struct(
            deconvolved=deconvolved,
            convolved=convolved,
            noise=noise,
            noisyImage=noisyImage,
            mCoadd=mCoadd,
            detectionTask=detectionTask,
            deconvolveTask=deconvolveTask,
            deblendTask=deblendTask,
        )

        if doDetect:
            # Generate a detection catalog
            detectionResult = detectionTask.run(table, mCoadd["r"])
            table = SourceCatalog.Table.make(schema)
            catalog = SourceCatalog(table)
            catalog.extend(detectionResult.sources, schemaMapper)
            result.catalog = catalog

        return result

    def deconvolve(self, data: Struct):
        deconvolvedCoadds = []
        deconvolveTask = data.deconvolveTask
        if deconvolveTask.config.useFootprints:
            catalog = data.catalog
        else:
            catalog = None
        for coadd in data.mCoadd:
            deconvolvedCoadd = deconvolveTask.run(coadd, catalog).deconvolved
            deconvolvedCoadds.append(deconvolvedCoadd)
        mDeconvolved = afwImage.MultibandExposure.fromExposures(self.bands, deconvolvedCoadds)
        return mDeconvolved

    def test_default_deconvolve(self):
        data = self.initialize_data(self.models)
        deconvolved = self.deconvolve(data)

        diff = data.deconvolved.data - deconvolved.image.array
        # Due to peakiness of Sersic models the center has a sharp peak,
        # so we ignore a 3x3 region around each source center
        for model in self.models:
            yc, xc = model.center
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    diff[:, yc+y, xc+x] = 0
        self.assertTrue(np.max(diff[:2]) < 10*np.std(data.noise))
        self.assertTrue(np.max(diff[2]) < 20*np.std(data.noise))

        context = mes.scarletDeblendTask.ScarletDeblendContext.build(
            data.mCoadd,
            deconvolved,
            data.catalog,
            data.deblendTask.ConfigClass()
        )

        self.assertEqual(len(context.footprints), 4)

    def test_catalog_free_deconvolve(self):
        config = DeconvolveExposureTask.ConfigClass()
        config.useFootprints = False
        data = self.initialize_data(self.models, deconvolveConfig=config)
        deconvolved = self.deconvolve(data)

        diff = data.deconvolved.data - deconvolved.image.array
        # Due to peakiness of Sersic models the center has a sharp peak,
        # so we ignore a 3x3 region around each source center
        for model in self.models:
            yc, xc = model.center
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    diff[:, yc+y, xc+x] = 0
        self.assertTrue(np.max(diff[:2]) < 10*np.std(data.noise))
        self.assertTrue(np.max(diff[2]) < 20*np.std(data.noise))

    def test_footprints(self):
        data = self.initialize_data(self.models)
        nParents = len(data.catalog)
        mDeconvolved = self.deconvolve(data)
        result = data.deblendTask.run(data.mCoadd, mDeconvolved, data.catalog)

        config = data.deblendTask.config
        catalog = result.deblendedCatalog
        modelData = result.scarletModelData
        observedPsf = modelData.metadata["psf"]
        modelPsf = modelData.metadata["model_psf"]

        # Attach the footprints in each band and compare to the full
        # data model. This is done in each band, both with and without
        # flux re-distribution to test all of the different possible
        # options of loading catalog footprints.
        for useFlux in [False, True]:
            for band in self.bands:
                bandIndex = self.bands.index(band)
                coadd = data.mCoadd[band]

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
                        source = next(
                            src for src in blend.sources if src.record_id == child.getId()
                        )
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

        # Check that the catalog matches the expected results
        nModels = len(self.models)
        self.assertEqual(len(catalog), nParents+nModels)

    def test_skipped(self):
        # Use tight configs to force skipping a 3 source footprint
        # and "large" footprint
        config = ScarletDeblendTask.ConfigClass()
        config.maxFootprintArea = 1000
        config.maxNumberOfPeaks = 2
        config.catchFailures = False

        data = self.initialize_data(self.models, deblendConfig=config)
        mDeconvolved = self.deconvolve(data)
        result = data.deblendTask.run(data.mCoadd, mDeconvolved, data.catalog)

        catalog = result.deblendedCatalog
        parents = catalog[catalog["parent"] == 0]
        self.assertEqual(np.sum(parents["deblend_skipped"]), 2)
        self.assertEqual(np.sum(parents["deblend_parentTooBig"]), 1)
        self.assertEqual(np.sum(parents["deblend_tooManyPeaks"]), 1)

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

    def _setup_butler(self):
        # Initialize a Butler to test persistence
        repo_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tempfile.TemporaryDirectory.cleanup, repo_dir)
        config = Config()
        config["datastore", "cls"] = "lsst.daf.butler.datastores.fileDatastore.FileDatastore"
        repo = makeTestRepo(repo_dir.name, config=config)
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
            universe=repo.dimensions,
        )
        repo.registry.registerDatasetType(datasetType)
        return repo

    def test_persistence(self):
        # Test that the model data is persisted correctly
        data = self.initialize_data(self.models)
        repo = self._setup_butler()
        mDeconvolved = self.deconvolve(data)
        result = data.deblendTask.run(data.mCoadd, mDeconvolved, data.catalog)
        modelData = result.scarletModelData
        bands = modelData.metadata["bands"]
        butler = makeTestCollection(repo, uniqueId="test_run1")
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
        repo = self._setup_butler()
        storageClass = StorageClass(
            "ScarletModelData",
            pytype=scl.io.ScarletModelData,
        )
        datasetType = DatasetType(
            "old_scarlet_model_data",
            dimensions=(),
            storageClass=storageClass,
            universe=repo.dimensions,
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
        butler = makeTestCollection(repo, uniqueId="ingestion")
        repo.registry.registerDatasetType(datasetType)
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
