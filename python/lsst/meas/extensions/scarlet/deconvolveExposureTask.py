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
import sys

import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.scarlet.lite as scl
import numpy as np
from lsst.afw.detection import PeakCatalog

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

    peaks = cT.Input(
        doc="Catalog of detected peak positions",
        name="{inputCoaddName}_coadd_multiband_peaks",
        storageClass="PeakCatalog",
        dimensions=("tract", "patch", "skymap"),
        deferLoad=True,
    )

    deconvolved = cT.Output(
        doc="Deconvolved exposure",
        name="deconvolved_{inputCoaddName}_coadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap"),
    )

    def __init__(self, *, config=None):
        if not config.usePeaks:
            # Deconvolution does not use input catalog
            self.inputs.remove("peaks")


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
    usePeaks = pexConfig.Field[bool](
        doc="Require pixels to be connected to peaks",
        default=False,
    )
    useWavelets = pexConfig.Field[bool](
        doc="Deconvolve using wavelets to supress high frequency noise",
        default=True,
    )
    waveletGeneration = pexConfig.ChoiceField[int](
        default=2,
        doc="Generation of the starlet wavelet used for peak detection. "
        "Only used if useWavelets is True",
        allowed={1: "First generation wavelets", 2: "Second generation wavelets"},
    )
    waveletScales = pexConfig.Field[int](
        default=1,
        doc="Number of wavelet scales used for peak detection. Only used if useWavelets is True",
    )
    backgroundThreshold = pexConfig.Field[float](
        default=0,
        doc="Threshold for background subtraction. "
        "Pixels in the fit below this threshold will be set to zero",
    )
    minFootprintArea = pexConfig.Field[int](
        default=0,
        doc="Minimum area of a footprint to be considered detectable. "
        "Regions with fewer than minFootprintArea connected pixels will be set to zero.",
    )
    modelStepSize = pexConfig.Field[float](
        default=0.5,
        doc="Step size for the FISTA algorithm.",
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
        inputs["band"] = inputRefs.coadd.dataId["band"]
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        coadd: afwImage.Exposure,
        band: str,
        peaks: PeakCatalog | None = None,
        **kwargs
    ) -> pipeBase.Struct:
        """Deconvolve an Exposure

        Parameters
        ----------
        coadd :
            Coadd image to deconvolve
        band :
            Band of the coadd image
        peaks :
            Catalog of detected peak positions
        """
        # Load the scarlet lite Observation
        observation = self._buildObservation(coadd, band)
        # Initialize the model
        scarletModel = self._initializeModel(observation, peaks)

        # Iteratively deconvolve the image
        scarletModel.fit(
            max_iter=self.config.maxIter,
            e_rel=self.config.eRel,
            min_iter=self.config.minIter,
        )

        # Store the model in an Exposure
        model = scarletModel.get_model().data[0]
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
        return pipeBase.Struct(deconvolved=exposure)

    def _removeHighFrequencySignal(
        self, coadd: afwImage.Exposure
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove high frequency signal from the image and PSF.

        This is done by performing a wavelet decomposition of the image
        and PSF, setting the high frequency wavelets to zero, and
        reconstructing the image and PSF from the remaining wavelets.

        Parameters
        ----------
        coadd : `afwImage.Exposure`
            Coadd image to deconvolve

        Returns
        -------
        image : `np.ndarray`
            Low frequency image
        psf : `np.ndarray`
            Low frequency PSF
        """
        psf = coadd.getPsf().computeKernelImage(coadd.getBBox().getCenter()).array
        # Build the wavelet coefficients
        wavelets = scl.detect.get_wavelets(
            coadd.image.array[None, :, :],
            coadd.variance.array[None, :, :],
            scales=self.config.waveletScales,
            generation=self.config.waveletGeneration,
        )
        # Remove the high frequency wavelets.
        # This has the effect of preventing high frequency noise
        # from interfering with the detection of peak positions.
        wavelets[0] = 0
        # Reconstruct the image from the remaining wavelet coefficients
        image = scl.wavelet.starlet_reconstruction(
            wavelets[:, 0],
            generation=self.config.waveletGeneration,
        )
        # Remove the high frequency wavelets from the PSF.
        # This is necesary for the image and PSF to have the
        # same frequency content.
        # See the document attached to DM-41840 for a more detailed
        # explanation.
        psf_wavelets = scl.wavelet.multiband_starlet_transform(
            psf[None, :, :],
            scales=self.config.waveletScales,
            generation=self.config.waveletGeneration,
        )
        psf_wavelets[0] = 0
        psf = scl.wavelet.starlet_reconstruction(
            psf_wavelets[:, 0],
            generation=self.config.waveletGeneration,
        )
        return image, psf

    def _buildObservation(self, coadd: afwImage.Exposure, band: str):
        """Build a scarlet lite Observation from an Exposure.

        Parameters
        ----------
        coadd :
            Coadd image to deconvolve.
        band :
            Band of the coadd image.
        """
        bands = (band,)
        model_psf = scl.utils.integrated_circular_gaussian(sigma=0.8)

        if self.config.useWavelets:
            image, psf = self._removeHighFrequencySignal(coadd)
        else:
            image = coadd.image.array
            psf = coadd.getPsf().computeKernelImage(coadd.getBBox().getCenter()).array
        weights = np.ones_like(coadd.image.array)
        badPixelMasks = ["SAT", "INTRP", "NO_DATA"]
        badPixels = coadd.mask.getPlaneBitMask(badPixelMasks)
        mask = coadd.mask.array & badPixels
        weights[mask > 0] = 0

        observation = scl.Observation(
            images=np.array([image.copy()]),
            variance=np.array([coadd.variance.array.copy()]),
            weights=np.array([weights]),
            psfs=np.array([psf]),
            model_psf=model_psf[None, :, :],
            convolution_mode="fft",
            bands=bands,
            bbox=utils.bboxToScarletBox(coadd.getBBox()),
        )
        return observation

    def _initializeModel(
        self, observation: scl.Observation, peaks: PeakCatalog | None = None
    ):
        """Initialize the model for the deconvolution."""
        if peaks is None:
            component_peaks = None
        else:
            component_peaks = [(peak["i_y"], peak["i_x"]) for peak in peaks]

        # Initialize the model as a single source with a single component:
        # the entire image.
        component = scl.models.free_form.FreeFormComponent(
            bands=observation.bands,
            model=observation.images.data.copy(),
            model_bbox=observation.bbox,
            bg_thresh=self.config.backgroundThreshold,
            bg_rms=observation.noise_rms,
            peaks=component_peaks,
            min_area=0,
        )
        source = scl.Source([component])
        blend = scl.Blend([source], observation)

        # Initialize the FISTA optimizer
        def _parameterization(component):
            component._model = scl.parameters.FistaParameter(
                component.model, step=self.config.modelStepSize
            )

        blend.parameterize(_parameterization)
        return blend
