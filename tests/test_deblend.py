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

import lsst.afw.image as afwImage
import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np
from lsst.afw.detection import GaussianPsf
from lsst.afw.table import SourceCatalog, SourceTable, SchemaMapper
from lsst.geom import Point2I
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.extensions.scarlet.scarletDeblendTask import ScarletDeblendTask
from lsst.meas.extensions.scarlet.deconvolveExposureTask import DeconvolveExposureTask
from lsst.pipe.base import Struct
from utils import initData, SersicModel, PsfModel


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


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
