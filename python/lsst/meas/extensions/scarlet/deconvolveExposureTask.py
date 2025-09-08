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
from .footprint import footprintsToNumpy

log = logging.getLogger(__name__)

__all__ = [
    "DeconvolveExposureTask",
    "DeconvolveExposureConfig",
    "DeconvolveExposureConnections",
]


def calculate_update_step(observation: scl.Observation) -> float:
    """Calculate the scale factor for the update step in deconvolution.

    For most images this will be 1.0 but for images with low SNR
    and/or high sparsity (for example LSST u-band images) the scale
    factor will be less than 1.0.

    Parameters
    ----------
    observation :
        Scarlet lite Observation.

    Returns
    -------
    scale : float
        Scale factor for the update step.
    """
    # Calculate sparsity as fraction of pixels significantly above noise
    noise_level = observation.noise_rms[0]
    signal_mask = observation.images.data > 3*noise_level
    signal_pixels = np.sum(signal_mask)
    sparsity = signal_pixels / observation.images.data.size

    if np.any(signal_mask):
        median_signal = np.median(observation.images.data[signal_mask])
        snr = median_signal / noise_level
    else:
        snr = 1.0

    # Scale factor that decreases with sparsity and increases with SNR
    scale = min(1.0, (sparsity * np.sqrt(snr)) / 0.1)

    return max(0.01, scale)


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

    coadd_cell = cT.Input(
        doc="Exposure on which to run deblending",
        name="{inputCoaddName}CoaddCell",
        storageClass="MultipleCellCoadd",
        dimensions=("tract", "patch", "band", "skymap")
    )

    background = cT.Input(
        doc="Background model to subtract from the cell-based coadd",
        name="{inputCoaddName}Coadd_calexp_background",
        storageClass="Background",
        dimensions=("tract", "patch", "band", "skymap")
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
            # Deconvolution will not use input catalog if
            # footprints are not used
            self.inputs.remove("catalog")

        if config.useCellCoadds:
            del self.coadd
        else:
            del self.coadd_cell
            del self.background


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
        default=True,
        doc="Use footprints to constrain the deconvolved model",
    )
    useCellCoadds = pexConfig.Field[bool](
        doc="Use cell-based coadd instead of regular coadd?",
        default=False,
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
        inputs['band'] = inputRefs.coadd.dataId['band']

        # Stitch together cell-based coadds (if necessary)
        if self.config.useCellCoadds:
            cellCoadd = inputs.pop('coadd_cell')
            background = inputs.pop('background')
            coadd = cellCoadd.stitch().asExposure()
            coadd.image -= background.getImage()
            inputs['coadd'] = coadd

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        coadd: afwImage.Exposure,
        catalog: afwTable.SourceCatalog | None = None,
        band: str = 'dummy'
    ) -> pipeBase.Struct:
        """Deconvolve an Exposure

        Parameters
        ----------
        coadd :
            Coadd image to deconvolve

        catalog :
            Catalog of sources detected in the merged catalog.
            This is used to supress noise in regions with no
            significant flux about the noise in the coadds.

        band :
            Band of the coadd image.
            Since this is a single band task the band isn't really necessary
            but can be useful for debugging so we keep it as a parameter.

        Returns
        -------
        deconvolved : `pipeBase.Struct`
            Deconvolved exposure
        """
        observation = self._buildObservation(coadd, catalog, band)
        self.bbox = coadd.getBBox()

        # Deconvolve.
        # Store the loss history for debugging purposes.
        model, self.loss = self._deconvolve(observation, catalog)

        # Store the model in an Exposure
        exposure = self._modelToExposure(model.data[0], coadd)
        return pipeBase.Struct(deconvolved=exposure)

    def _buildObservation(
        self,
        coadd: afwImage.Exposure,
        catalog: afwTable.SourceCatalog | None = None,
        band: str = 'dummy'
    ) -> scl.Observation:
        """Build a scarlet lite Observation from an Exposure.

        We don't actually use scarlet, but the optimized convolutions
        using scarlet data products are still useful.

        Parameters
        ----------
        coadd :
            Coadd image to deconvolve.
        catalog :
            Catalog of sources.
            This is used to find a location for the PSF if it cannot be
            generated at the center of the coadd.

        band :
            Band of the coadd image.

        """
        bands = (band,)
        model_psf = scl.utils.integrated_circular_gaussian(sigma=0.8)

        image = coadd.image.array
        psfCenter = coadd.getBBox().getCenter()
        if catalog is not None:
            psf, _, _ = utils.computeNearestPsf(coadd, catalog, band, psfCenter)
            if psf is None:
                # There were no valid locations from
                # which a PSF could be obtained
                raise pipeBase.NoWorkFound("No valid PSF could be obtained for deconvolution")
            psf = psf.array
        else:
            psf = coadd.getPsf().computeKernelImage(psfCenter).array

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
        catalog :
            Catalog of sources detected in the deconvolved image.
            This is used to mask the deconvolved image so that
            the deconvolved footprints detected downstream will always
            fit inside of the original footprints.
        """
        model = observation.images.copy()
        loss = []
        step = calculate_update_step(observation)
        if catalog is not None:
            width, height = self.bbox.getDimensions()
            x0, y0 = self.bbox.getMin()
            footprintImage = footprintsToNumpy(catalog, (height, width), (x0, y0))
        for n in range(self.config.maxIter):
            residual = observation.images - observation.convolve(model)
            loss.append(-0.5 * np.sum(residual.data**2))
            update = observation.convolve(residual, grad=True)
            update.data[:] *= step
            model += update
            model.data[model.data < 0] = 0
            if catalog is not None:
                # Ensure that the deconvolved model footprints fit
                # inside of the original footprints by setting regions
                # outside of the original footprints to zero.
                model.data[:] *= footprintImage

            # Check for a diverging model
            if len(loss) > 1 and loss[-1] < loss[-2]:
                step = step / 2
                self.log.warning(f"Loss increased at iteration {n}, decreasing scale to {step}")

            # Check for convergence
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
