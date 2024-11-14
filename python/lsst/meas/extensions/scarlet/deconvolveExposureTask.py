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

from scipy import ndimage

import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.scarlet.lite as scl
import numpy as np
from lsst.afw.table import SourceCatalog
from lsst.scarlet.lite.detect_pybind11 import get_connected_multipeak

from . import utils

log = logging.getLogger(__name__)

__all__ = [
    "DeconvolveExposureTask",
    "DeconvolveExposureConfig",
    "DeconvolveExposureConnections",
]

def multibandDeconvolve(
    observation: scl.Observation,
    model: scl.Image | None = None,
    mask: np.ndarray | None = None,
    maxIter: int = 300,
    eRel: float = 1e-2,
    minIter: int = 20,
    peaks: list | None = None,
) -> tuple[scl.Image, list[float]]:
    """Deconvolve a multi-band observation

    Parameters
    ----------
    observation :
        The observation to deconvolve.
    model :
        The initial model. If None, a copy of the observation images is used.
    mask :
        Mask to apply to the model.
        This is usually a boolean array of footprints for the entire image
        that is created after the first round of processing.
        If ``None``, no mask is applied.
    maxIter :
        Maximum number of iterations.
    e_rel :
        Relative error threshold.
    min_iter :
        Minimum number of iterations.
    peaks :
        List of (y, x) peak positions to use for deconvolution.
        If not ``None`` then all pixels not connected to a peak
        will be set to zero.
    """
    if model is None:
        model = observation.images.copy()
    loss = []
    for n in range(maxIter):
        residual = observation.images - observation.convolve(model)
        loss.append(-0.5 * np.sum(residual.data**2))
        update = observation.convolve(residual, grad=True).data
        model_data = np.zeros(model.data.shape, dtype=model.dtype)
        for b in range(len(model.data)):
            model_data[b] = model.data[b] + update[b]
            if mask is not None:
                model_data[b] *= mask
        model_data[model_data < 0] = 0

        if peaks is not None:
            valid = np.sum(model_data > 0, axis=0) > 1
            footprint = get_connected_multipeak(valid, peaks, 0)
            for b in range(len(model.data)):
                model_data[b] *= footprint

        model.data[:] = model_data
        if n > minIter and np.abs(loss[-1] - loss[-2]) < eRel * np.abs(loss[-1]):
            break

    return model, loss


def generate_circle(radius: int) -> np.ndarray:
    """Generate a circular mask of a given radius"""
    size = 2 * radius + 1
    x = np.linspace(-radius, radius, size)
    x, y = np.meshgrid(x, x)
    r2 = radius**2
    circle = x**2 + y**2 <= r2
    return circle


def insert_psf(psf: np.ndarray, image: scl.Image, center: tuple[int, int]):
    """Insert a PSF into an image at a given center

    Parameters
    ----------
    psf :
        The PSF to insert.
    image :
        The image to insert the PSF into.
    center :
        The center (y, x) of the PSF in the image.
    """
    if len(psf.shape) != len(image.shape):
        raise ValueError("PSF and image must have the same number of dimensions")
    if psf.shape[0] % 2 != 1:
        raise ValueError("height must have an odd number of pixels")
    if psf.shape[1] % 2 != 1:
        raise ValueError("width must have an odd number of pixels")

    y_radius = psf.shape[0] // 2
    x_radius = psf.shape[1] // 2
    yc, xc = center
    if psf.shape == 3:
        bands = image.bands
    else:
        bands = None
    psf_image = scl.Image(psf, bands=bands, yx0=(yc-y_radius, xc-x_radius))
    image.insert(psf_image)


class DeconvolveExposureConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap", "band"),
    defaultTemplates={"inputCoaddName": "deep"},
):
    """Connections for DeconvolveExposureTask"""

    coadds = cT.Input(
        doc="Exposures to deconvolve",
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap"),
    )

    sources = cT.Input(
        doc="Input detection catalog",
        name="{inputCoaddName}Coadd_mergeDet",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
        deferLoad=True,
    )

    deconvolved = cT.Output(
        doc="Deconvolved exposure",
        name="deconvolved_{inputCoaddName}_coadd",
        storageClass="ExposureF",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap"),
    )

    def __init__(self, *, config=None):
        if not config.usePeaks:
            # Deconvolution does not use input catalog
            self.inputs.remove("sources")


class DeconvolveExposureConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=DeconvolveExposureConnections,
):
    """Configuration for DeconvolveExposureTask"""

    maxIter = pexConfig.Field[int](
        doc="Maximum number of iterations.",
        default=100,
    )
    minIter = pexConfig.Field[int](
        doc="Minimum number of iterations.",
        default=10,
    )
    eRelInitial = pexConfig.Field[float](
        doc="Relative error threshold for the initial deconvolution.",
        default=1e-2,
    )
    eRelFinal = pexConfig.Field[float](
        doc="Relative error threshold for the final deconvolution.",
        default=1e-2,
    )
    usePeaks = pexConfig.Field[bool](
        doc="Require pixels to be connected to peaks",
        default=False,
    )
    useWavelets = pexConfig.Field[bool](
        doc="Use wavelets for the deconvolution",
        default=False,
    )
    waveletGeneration = pexConfig.ChoiceField[int](
        default=2,
        doc="Generation of the starlet wavelet used for peak detection.",
        allowed={1: "First generation starlets", 2: "Second generation starlets"},
    )
    backgroundThreshold = pexConfig.Field[float](
        default=0,
        doc="Threshold for background subtraction. "
        "Pixels in the fit below this threshold will be set to zero.",
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
    badMask = pexConfig.ListField[str](
        default=utils.defaultBadPixelMasks,
        doc="Whether or not to process isolated sources in the deblender.",
    )
    cacheIntermediateProducts = pexConfig.Field[bool](
        default=False,
        doc="Whether or not to cache intermediate products for debugging.",
    )
    footprintGrowSize = pexConfig.Field[int](
        default=2,
        doc="Number of pixels to grow the footprint mask.",
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
        inputs["bands"] = [dRef.dataId["band"] for dRef in inputRefs.coadds]
        outputs = self.run(**inputs)
        for ref in outputRefs.deconvolved:
            band = ref.dataId["band"]
            butlerQC.put(outputs.coaddDict[band], ref)

    def run(
        self,
        coaddList: list[afwImage.Exposure],
        bands: list[str],
        sources: SourceCatalog | None = None,
        **kwargs
    ) -> pipeBase.Struct:
        """Deconvolve an Exposure

        Parameters
        ----------
        coadds :
            Coadd images to deconvolve.
        bands :
            Band for each coadd in coadds, in the same order.
        sources :
            Catalog with detected peak positions.
        """
        mCoadd = afwImage.MultibandExposure.fromExposures(bands, coaddList)
        # Load the scarlet lite Observation
        modelPsf = scl.utils.integrated_circular_gaussian(sigma=0.8)
        observation = utils.buildObservation(
            modelPsf=modelPsf,
            psfCenter=mCoadd.getBBox().getCenter(),
            mExposure=mCoadd,
            badPixelMasks=self.config.badMask,
            convolutionType="fft",
            useWavelets=False,
        )
        # Build an observation with the high frequency signal removed
        low_freq_obs = utils.buildObservation(
            modelPsf=modelPsf,
            psfCenter=mCoadd.getBBox().getCenter(),
            mExposure=mCoadd,
            badPixelMasks=self.config.badMask,
            convolutionType="fft",
            useWavelets=True,
        )
        # Perform the initial, low resolution deconvolution.
        # This is used to supress the high frequency noise in the model
        # and generate an initial set of compact footprints.
        print("deconvolving")
        model, loss = multibandDeconvolve(low_freq_obs, eRel=self.config.eRelInitial)

        # Generate a mask of the footprints.
        # This is a slightly different algorithm than the one that
        # creates the final footprints because it uses multi-resolution
        # denoising to remove noise in the image.
        print("footprints 1")
        sigma = np.median(np.sqrt(mCoadd.variance.array), axis=(1, 2))
        detect = np.sum(model.data/sigma[:, None, None], axis=0)
        footprints = scl.detect.detect_footprints(
            images=np.array([detect]),
            variance = np.ones((1, detect.shape[0], detect.shape[1]), dtype=detect.dtype),
            scales=1,
            min_area=9,
            origin=observation.bbox.origin,
        )
        footprintMask = scl.detect.footprints_to_image(footprints, observation.bbox) > 0
        if self.config.cacheIntermediateProducts:
            self.initialMask = footprintMask.copy()

        # Insert low flux PSFs at the peak positions
        print("inserting PSFs")
        peaks = self._sourcesToPeaks(sources)
        self._insertLowFluxPsfs(footprintMask, peaks)

        # Grow the footprints to fill in gaps and allow for slightly more flux
        # after deconvolution.
        print("growing footprints")
        growthKernel = generate_circle(self.config.footprintGrowSize)
        footprintMask = ndimage.binary_dilation(footprintMask.data > 0, growthKernel)

        # Perform the final deconvolution
        print("deconvolving 2")
        model, loss = multibandDeconvolve(observation, eRel=self.config.eRelFinal, mask=footprintMask)
        if self.config.cacheIntermediateProducts:
            self.loss = loss
            self.finalModel = model
            self.expandedMask = footprintMask.copy()
        print("returning")

        # Convert the model to a set of coadds
        coaddDict = self._modelToCoadds(model, mCoadd)

        return pipeBase.Struct(coaddDict=coaddDict)

    def _sourcesToPeaks(self, sources: SourceCatalog) -> list[tuple[int, int]]:
        """Convert a SourceCatalog to a list of peak positions

        Parameters
        ----------
        sources :
            Source catalog

        Returns
        -------
        peaks :
            List of (y, x) peak positions.
        """
        peaks = [(peak["i_y"], peak["i_x"]) for src in sources for peak in src.getFootprint().peaks]
        peaks = [peak for peak in peaks if peak[0] < 500 and peak[1] < 500]
        if self.config.cacheIntermediateProducts:
            self.peaks = peaks
        return peaks

    def _insertLowFluxPsfs(
        self,
        footprint_mask: scl.Image,
        peaks: list[tuple[int, int]],
    ) -> None:
        """Insert PSFs at the peak positions.

        This ensures that the mask used for the final deconvolution
        includes a minimal footprint around each peak.

        Parameters
        ----------
        footprint_mask :
            Footprint mask
        peaks :
            List of (y, x) peak positions.
        """
        circle = generate_circle(3)
        for peak in peaks:
            insert_psf(circle, footprint_mask, peak)

    def _modelToCoadds(
        self,
        model: scl.Image,
        mCoadd: afwImage.MultibandExposure
    ) -> dict[str, afwImage.Exposure]:
        """Convert a deconvolved model to a dict of coadds

        Parameters
        ----------
        model :
            Deconvolved model.
        mCoadd :
            Multiband input coadd that was deconvolved.


        Returns
        -------
        coaddDict :
            Dictionary of coadds.
        """
        coaddDict = {}
        for b, band in enumerate(mCoadd.filters):
            image = afwImage.Image(model.data[b], xy0=mCoadd.getBBox().getMin(), dtype=model.dtype)
            maskedImage = afwImage.MaskedImage(image, dtype=model.dtype)
            coaddDict[band] = afwImage.Exposure(maskedImage, dtype=model.dtype)
        return coaddDict
