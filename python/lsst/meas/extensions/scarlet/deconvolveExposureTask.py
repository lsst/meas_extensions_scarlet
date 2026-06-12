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

import dataclasses
import logging

import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.images as imgs
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.scarlet.lite as scl
import numpy as np
from lsst.images.cells import CellCoadd
from deprecated.sphinx import deprecated
from lsst.cell_coadds import StitchedPsf

from . import utils
from .stitched_psf import ScarletStitchedPsf

log = logging.getLogger(__name__)

__all__ = [
    "DeconvolveExposureTask",
    "DeconvolveExposureConfig",
    "DeconvolveExposureConnections",
]


def calculateUpdateStep(
    observation: scl.Observation,
    minScale: float = 0.01,
    defaultScale: float = 0.1,
) -> float:
    """Calculate the scale factor for the update step in deconvolution.

    For most images this will be 1.0 but for images with low SNR
    and/or high sparsity (for example LSST u-band images) the scale
    factor will be less than 1.0.

    Parameters
    ----------
    observation :
        Scarlet lite Observation.

    minScale :
        Minimum allowed scale factor.

    defaultScale :
        Default scale factor to return if noise level is non-finite.

    Returns
    -------
    scale : float
        Scale factor for the update step.
    """
    # Calculate sparsity as fraction of unmasked pixels significantly
    # above noise. Pixels with zero weight (border, NO_DATA, BAD) are
    # excluded from both numerator and denominator so heavily masked
    # inputs are not biased toward a small step.
    noiseLevel = observation.noise_rms[0]
    # Guard against non-finite or non-positive noise levels
    if noiseLevel <= 0 or not np.isfinite(noiseLevel):
        return defaultScale
    image = observation.images.data[0]
    validMask = observation.weights.data[0] > 0
    validPixels = np.sum(validMask)
    if validPixels == 0:
        return defaultScale
    signalMask = (image > 3*noiseLevel) & validMask
    signalPixels = np.sum(signalMask)
    sparsity = signalPixels / validPixels

    if np.any(signalMask):
        medianSignal = np.median(image[signalMask])
        snr = medianSignal / noiseLevel
    else:
        snr = 1.0

    # Scale factor that decreases with sparsity and increases with SNR
    scale = min(1.0, (sparsity * np.sqrt(snr)) / 0.1)

    return max(minScale, scale)


@deprecated(
    reason=(
        "Use `calculateUpdateStep` instead; the snake_case name is kept "
        "as a shim. Will be removed after v31."
    ),
    version="v30.0",
    category=FutureWarning,
)
def calculate_update_step(
    observation: scl.Observation,
    min_scale: float = 0.01,
    default_scale: float = 0.1,
) -> float:
    """Deprecated snake_case alias for `calculateUpdateStep`."""
    return calculateUpdateStep(
        observation, minScale=min_scale, defaultScale=default_scale,
    )


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
        if config.imageType == "future":
            self.coadd = dataclasses.replace(self.coadd, storageClass="CellCoadd")
            self.deconvolved = dataclasses.replace(self.deconvolved, storageClass="MaskedImageV2")
            del self.coadd_cell
            del self.background
        elif config.useCellCoadds:
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
    imageType = pexConfig.ChoiceField[str](
        "Which image type to read and write. "
        "This option only directly affects connection storage classes and hence 'runQuantum'; the 'run' "
        "method behavior is determined by which type is actually passed in.",
        allowed={
            "legacy": (
                "Read a lsst.cell_coadds.MultipleCellCoadd and restore 'background' (if useCellCoadd) or "
                "lsst.afw.image.Exposure (if not useCellCoadd), and write an lsst.afw.image.Exposure."
            ),
            "future": (
                "Read a lsst.images.cells.CellCoadd via 'connections.coadd' and write an "
                "lsst.images.MaskedImage.  The useCellCoadd option will be ignored."
            ),
        },
        optional=False,
        default="legacy",
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

        match self.config.imageType:
            case "legacy":
                if self.config.useCellCoadds:
                    band = inputRefs.coadd_cell.dataId['band']
                    cellCoadd = inputs.pop('coadd_cell')
                    background = inputs.pop('background')
                    coadd = cellCoadd.stitch().asExposure()
                    coadd.image -= background.getImage()
                else:
                    coadd = inputs.pop("coadd")
                    band = inputRefs.coadd.dataId['band']
            case "future":
                coadd = inputs.pop("coadd")
                band = inputRefs.coadd.dataId['band']
            case _:
                raise AssertionError(f"Invalid choice {self.config.imageType!r} for imageType.")

        catalog = inputs.pop('catalog', None)

        assert not inputs, "runQuantum got more inputs than expected."
        outputs = self.run(
            coadd=coadd,
            catalog=catalog,
            band=band,
        )
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        coadd: afwImage.Exposure | CellCoadd,
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
            Deconvolved exposure (if an `lsst.afw.image.Exposure` is provided;
            an `lsst.images.MaskedImage` if an `lsst.images.cells.CellCoadd`
            is provided).
        """
        futureInputImage = None
        if isinstance(coadd, CellCoadd):
            # For now we just convert the future CellCoadd into an Exposure for
            # the bulk of the work, and convert the result back at the end (we
            # just convert to MaskedImage because we don't need to duplicate
            # the structured metadata). Converting the internals to use
            # lsst.images types would be disruptive but could take advantage of
            # the fact that the lsst.images.CellPointSpreadFunction type knows
            # which cells are missing and could probably do a better job of
            # picking a decent representative PSF for the full image, but it
            # would be cleanest to do that while rewriting some of the utility
            # functions to work exclusively with lsst.images types, and that
            # looks like it might be disruptive.
            futureInputImage = coadd
            coadd = coadd.to_legacy()
        observation = self._buildObservation(coadd, catalog, band)

        # Build the per-pixel footprint mask from the catalog, if one
        # was supplied, so the deconvolution loop only needs to know
        # about the mask itself rather than how it was derived.
        if catalog is not None:
            bbox = coadd.getBBox()
            width, height = bbox.getDimensions()
            x0, y0 = bbox.getMin()
            footprintImage = afwDet.footprintsToNumpy(
                catalog, shape=(height, width), xy0=(x0, y0)
            )
        else:
            footprintImage = None

        model, loss = self._deconvolve(observation, footprintImage=footprintImage)

        deconvolved = self._modelToExposure(model.data[0], coadd)
        if futureInputImage:
            deconvolved = imgs.MaskedImage.from_legacy(
                deconvolved.maskedImage,
                unit=futureInputImage.unit,
                plane_map=imgs.get_legacy_deep_coadd_mask_planes(),
                sky_projection=futureInputImage.sky_projection,
            )
        return pipeBase.Struct(deconvolved=deconvolved, loss=loss)

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

        # Give zero weight to non-finite pixels
        weights = np.ones_like(coadd.image.array)
        weights[~np.isfinite(coadd.image.array)] = 0

        image = coadd.image.array.copy()
        # Set non-finite pixels to zero
        image[~np.isfinite(image)] = 0.0

        coaddPsf = coadd.getPsf()
        if isinstance(coaddPsf, StitchedPsf):
            # Cell-based coadd: the PSF is genuinely discontinuous across
            # cells, so build a spatially-varying ScarletStitchedPsf over the
            # cell grid instead of a single kernel image. A StitchedPsf can be
            # evaluated everywhere within the coadd, so the catalog-based
            # nearest-PSF fallback used by the flat path is unnecessary here.
            observedPsf: scl.Psf = ScarletStitchedPsf.from_stitched_psf(
                {band: coaddPsf}, dtype=image.dtype
            )
        else:
            psfCenter = coadd.getBBox().getCenter()
            if catalog is not None:
                psf, _, _ = utils.computeNearestPsf(coadd, catalog, band, psfCenter)
                if psf is None:
                    # There were no valid locations from
                    # which a PSF could be obtained
                    raise pipeBase.NoWorkFound("No valid PSF could be obtained for deconvolution")
                psf = psf.array
            else:
                psf = coaddPsf.computeKernelImage(psfCenter).array
            observedPsf = scl.ImagePsf(psf[None], bands=bands)

        badPixelMasks = utils.defaultBadPixelMasks
        badPixels = coadd.mask.getPlaneBitMask(badPixelMasks)
        mask = coadd.mask.array & badPixels
        weights[mask > 0] = 0

        observation = scl.Observation(
            images=image[None],
            variance=coadd.variance.array.copy()[None],
            weights=weights[None],
            psf=observedPsf,
            model_psf=scl.ImagePsf(model_psf[None]),
            convolution_mode="fft",
            bands=bands,
            bbox=utils.bboxToScarletBox(coadd.getBBox()),
        )
        return observation

    def _deconvolve(
        self,
        observation: scl.Observation,
        footprintImage: np.ndarray | None = None,
    ) -> tuple[scl.Image, list[float]]:
        """Deconvolve the observed image.

        Parameters
        ----------
        observation :
            Scarlet lite Observation.
        footprintImage :
            Per-pixel mask matching ``observation.images.shape[1:]``.
            When supplied, the deconvolved model is multiplied by this
            mask after each iteration so the recovered footprints stay
            inside the input footprints.
        """
        model = observation.images.copy()
        loss = []
        step = calculateUpdateStep(observation)
        for n in range(self.config.maxIter):
            # cache=True reuses the FFT plan across iterations; the
            # image shape is stable inside the loop so this is a free
            # speedup at zero correctness cost.
            residual = observation.images - observation.convolve(model, cache=True)
            if np.all(~np.isfinite(residual.data)):
                self.log.warning(f"Residual is non-finite at iteration {n}, stopping deconvolution")
                loss.append(-np.inf)
                break
            loss.append(-0.5 * np.nansum(residual.data**2))
            update = observation.convolve(residual, grad=True, cache=True)
            update.data[:] *= step
            model += update
            model.data[(model.data < 0) | ~np.isfinite(model.data)] = 0
            if footprintImage is not None:
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
        """Convert a deconvolved image array to an Exposure.

        The output exposure's mask is a deep copy of the input coadd's
        mask, and its variance plane is fresh and filled with ``inf``.
        Convolution-then-deconvolution alters the per-pixel noise
        covariance, so the input coadd's variance no longer describes
        the deconvolved pixel values; the infinite variance signals
        "no information about the noise here" and naturally zero-weights
        these pixels under any inverse-variance scheme. Downstream
        consumers that need a variance plane must supply their own.

        Parameters
        ----------
        model :
            Deconvolved image array.
        coadd :
            Input coadd exposure; its image dtype, bbox, ``ExposureInfo``,
            and mask contents are reused.
        """
        image = afwImage.Image(
            array=model,
            xy0=coadd.getBBox().getMin(),
            deep=False,
            dtype=coadd.image.array.dtype,
        )
        # Deep-copy the mask and build a fresh inf-filled variance so
        # the output exposure doesn't alias the input coadd's planes.
        # The variance is deliberately invalidated because the input's
        # variance does not describe the deconvolved pixel values.
        mask = coadd.mask.clone()
        variance = coadd.variance.Factory(coadd.variance.getBBox())
        variance.array[:] = np.inf
        maskedImage = afwImage.MaskedImage(
            image=image,
            mask=mask,
            variance=variance,
            dtype=coadd.image.array.dtype,
        )
        exposure = afwImage.Exposure(
            maskedImage=maskedImage,
            exposureInfo=coadd.getInfo(),
            dtype=coadd.image.array.dtype,
        )
        return exposure
