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

import logging

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.scarlet.lite as scl
import numpy as np

from . import utils

log = logging.getLogger(__name__)

__all__ = [
    "DeconvolveExposureTask",
    "DeconvolveExposureConfig",
    "DeconvolveExposureConnections",
]


class DeconvolveExposureConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap", "band"),
    defaultTemplates={"inputCoaddName": "deep"},
):
    """Connections for DeconvolveExposureTask"""

    coadd = cT.Input(
        doc="Exposure to deconvolve",
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap"),
    )

    catalog = cT.Input(
        doc="Catalog of sources detected in the deconvolved image",
        name="{inputCoaddName}Coadd_mergeDet",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )

    deconvolved = cT.Output(
        doc="Deconvolved exposure",
        name="deconvolved_{inputCoaddName}_coadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap"),
    )

    def __init__(self, *, config=None):
        if not config.useFootprints:
            # Deconvolution does not use input catalog
            self.inputs.remove("catalog")


class DeconvolveExposureConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=DeconvolveExposureConnections,
):
    """Configuration for DeconvolveExposureTask"""

    maxIter = pexConfig.Field[int](
        doc="Maximum number of iterations",
        default=100,
    )
    minIter = pexConfig.Field[int](
        doc="Minimum number of iterations",
        default=10,
    )
    eRel = pexConfig.Field[float](
        doc="Relative error threshold",
        default=1e-3,
    )
    backgroundThreshold = pexConfig.Field[float](
        default=0,
        doc="Threshold for background subtraction. "
        "Pixels in the fit below this threshold will be set to zero",
    )
    useFootprints = pexConfig.Field[bool](
        default = True,
        doc="Use footprints to constrain the deconvolved model",
    )


class DeconvolveExposureTask(pipeBase.PipelineTask):
    """Deconvolve an Exposure using scarlet lite."""

    ConfigClass = DeconvolveExposureConfig
    _DefaultName = "deconvolveExposure"

    def __init__(self, initInputs=None, **kwargs):
        if initInputs is None:
            initInputs = {}
        super().__init__(initInputs=initInputs, **kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        coadd: afwImage.Exposure,
        catalog: afwTable.SourceCatalog | None = None,
    ) -> pipeBase.Struct:
        """Deconvolve an Exposure

        Parameters
        ----------
        coadd :
            Coadd image to deconvolve

        Returns
        -------
        deconvolved : `pipeBase.Struct`
            Deconvolved exposure
        """
        # Load the scarlet lite Observation
        observation = self._buildObservation(coadd)
        self.bbox = coadd.getBBox()

        # Deconvolve.
        # Store the loss history for debugging purposes.
        model, self.loss = self._deconvolve(observation, catalog)

        # Store the model in an Exposure
        exposure = self._modelToExposure(model.data[0], coadd)
        return pipeBase.Struct(deconvolved=exposure)

    def _buildObservation(self, coadd: afwImage.Exposure) -> scl.Observation:
        """Build a scarlet lite Observation from an Exposure.

        We don't actually use scarlet, but the optimized convolutions
        using scarlet data products are still useful.

        Parameters
        ----------
        coadd :
            Coadd image to deconvolve.
        """
        bands = ("dummy",)
        model_psf = scl.utils.integrated_circular_gaussian(sigma=0.8)

        image = coadd.image.array
        psf = coadd.getPsf().computeKernelImage(coadd.getBBox().getCenter()).array
        weights = np.ones_like(coadd.image.array)
        badPixelMasks = utils.defaultBadPixelMasks
        badPixels = coadd.mask.getPlaneBitMask(badPixelMasks)
        mask = coadd.mask.array & badPixels
        weights[mask > 0] = 0

        observation = scl.Observation(
            images=image.copy()[None],
            variance=coadd.variance.array.copy()[None],
            weights=weights[None],
            psfs=psf[None],
            model_psf=model_psf[None],
            convolution_mode="fft",
            bands=bands,
            bbox=utils.bboxToScarletBox(coadd.getBBox()),
        )
        return observation

    def _deconvolve(
        self,
        observation: scl.Observation,
        catalog: afwTable.SourceCatalog | None = None,
    ) -> tuple[scl.Image, list[float]]:
        """Deconvolve the observed image.

        Parameters
        ----------
        observation :
            Scarlet lite Observation.
        """
        model = observation.images.copy()
        loss = []
        if catalog is not None:
            width, height = self.bbox.getDimensions()
            x0, y0 = self.bbox.getMin()
            footprintImage = utils.footprintsToNumpy(catalog, (height, width), (y0, x0))
        for n in range(self.config.maxIter):
            residual = observation.images - observation.convolve(model)
            loss.append(-0.5 * np.sum(residual.data**2))
            update = observation.convolve(residual, grad=True)
            model += update
            model.data[model.data < 0] = 0
            if catalog is not None:
                model.data[:] *= footprintImage

            if n > self.config.minIter and np.abs(loss[-1] - loss[-2]) < self.config.eRel * np.abs(loss[-1]):
                break

        return model, loss

    def _modelToExposure(self, model: np.ndarray, coadd: afwImage.Exposure) -> afwImage.Exposure:
        """Convert a scarlet lite Image to an Exposure.

        Parameters
        ----------
        image :
            Scarlet lite Image.
        """
        image = afwImage.Image(
            array=model,
            xy0=coadd.getBBox().getMin(),
            deep=False,
            dtype=coadd.image.array.dtype,
        )
        maskedImage = afwImage.MaskedImage(
            image=image,
            mask=coadd.mask,
            variance=coadd.variance,
            dtype=coadd.image.array.dtype,
        )
        exposure = afwImage.Exposure(
            maskedImage=maskedImage,
            exposureInfo=coadd.getInfo(),
            dtype=coadd.image.array.dtype,
        )
        return exposure
