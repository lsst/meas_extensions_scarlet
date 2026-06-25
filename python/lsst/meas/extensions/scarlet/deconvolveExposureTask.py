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
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.images as imgs
import lsst.meas.algorithms as measAlg
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


def scarletImagePsfToLsst(psf: scl.ImagePsf) -> measAlg.KernelPsf:
    """Convert a scarlet lite `ImagePsf` to an LSST `Psf`.

    The deconvolved model lives in scarlet's model frame, whose PSF is
    a single fixed kernel image rather than a spatially-varying model.
    A `~lsst.meas.algorithms.KernelPsf` wrapping a
    `~lsst.afw.math.FixedKernel` is the LSST representation of exactly
    that: one image-based kernel that is constant across the exposure.

    Parameters
    ----------
    psf :
        Single-band scarlet lite image PSF. Only the first band is used;
        scarlet's model PSF is band-independent.

    Returns
    -------
    lsstPsf : `lsst.meas.algorithms.KernelPsf`
        The LSST PSF wrapping the same kernel image.
    """
    # FixedKernel needs a contiguous double-precision ImageD; the kernel
    # image must have odd dimensions, which scarlet's model PSF always does.
    kernelImage = afwImage.ImageD(np.ascontiguousarray(psf.data[0], dtype=np.float64))
    kernel = afwMath.FixedKernel(kernelImage)
    return measAlg.KernelPsf(kernel)


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
    modelPsfSigma = pexConfig.Field[float](
        default=0.8, doc="Define sigma for the model frame PSF"
    )
    backgroundThreshold = pexConfig.Field[float](
        default=0,
        doc="Threshold for background subtraction. "
        "Pixels in the fit below this threshold will be set to zero",
    )
    badMask = pexConfig.ListField[str](
        default=utils.defaultBadPixelMasks,
        doc="Mask planes flagged as bad. Pixels with any of these planes set "
        "are zero-weighted, and the residual is zeroed there during "
        "deconvolution so they exert no pull on the fit.",
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
    useStitchedPsf = pexConfig.Field[bool](
        doc="When the coadd PSF is a cell-coadd ``StitchedPsf``, build a "
        "spatially-varying ``ScarletStitchedPsf`` that convolves each cell "
        "with its own kernel. This is more accurate but convolves every cell "
        "with a separate FFT, so it is far slower on a full patch. Set to "
        "`False` to use a single PSF kernel at the image center (an "
        "``ImagePsf``) even for cell coadds -- much faster, slightly less "
        "accurate. Ignored for non-cell coadds, which are always flat.",
        default=True,
    )
    useFista = pexConfig.Field[bool](
        doc="Use FISTA (Beck & Teboulle 2009), an accelerated proximal "
        "gradient method that adds Nesterov momentum to the gradient "
        "descent and converges as O(1/k^2) instead of O(1/k) on this "
        "convex least-squares problem, giving a better fit in the same "
        "number of iterations. Set to `False` for plain gradient descent, "
        "which additionally halves the step size whenever the loss "
        "increases (FISTA requires a constant step, so it skips that).",
        default=False,
    )
    useZeroInit = pexConfig.Field[bool](
        doc="Initialize the deconvolved image at zero. Set to `False` to "
        "initialize with the observed image instead, which starts the fit "
        "closer to the (still convolved) data.",
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

        deconvolved = self._modelToExposure(model.data[0], coadd, observation.model_psf)
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
        model_psf = scl.utils.integrated_circular_gaussian(sigma=self.config.modelPsfSigma)

        # Give zero weight to non-finite pixels
        weights = np.ones_like(coadd.image.array)
        weights[~np.isfinite(coadd.image.array)] = 0

        image = coadd.image.array.copy()
        # Set non-finite pixels to zero
        image[~np.isfinite(image)] = 0.0

        coaddPsf = coadd.getPsf()
        if self.config.useStitchedPsf and isinstance(coaddPsf, StitchedPsf):
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

        badPixels = coadd.mask.getPlaneBitMask(self.config.badMask)
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

        Fits the deconvolved image ``x`` such that ``A.x ~ data``, where
        ``A`` is the PSF convolution carried by ``observation``. The fit
        maximizes the Gaussian log-likelihood ``-0.5 * sum(residual**2)``
        by proximal gradient ascent, projecting onto non-negativity (and
        the input footprints, when supplied) after every step.

        Two solvers are available, selected by ``config.useFista``:

        - FISTA (the default), which adds Nesterov momentum and converges
          as O(1/k^2). It requires a constant step size.
        - Plain gradient ascent, which uses no momentum but halves the
          step size whenever the loss increases, guarding against a
          diverging model.

        Parameters
        ----------
        observation :
            Scarlet lite Observation, providing the forward
            (``convolve``) and adjoint (``convolve(..., grad=True)``) PSF
            operators as well as the observed ``images`` and ``weights``.
        footprintImage :
            Per-pixel mask matching ``observation.images.shape[1:]``.
            When supplied, the deconvolved model is multiplied by this
            mask after each iteration so the recovered footprints stay
            inside the input footprints.

        Returns
        -------
        model : `lsst.scarlet.lite.Image`
            The deconvolved image in scarlet's model frame.
        loss : `list` [`float`]
            Per-iteration log-likelihood ``-0.5 * sum(residual**2)``.
        """
        band = observation.bands[0]
        yx0 = observation.bbox.origin
        dtype = observation.images.dtype
        step = calculateUpdateStep(observation)

        def prox(image: np.ndarray) -> np.ndarray:
            """Project the model onto the constraint set.

            Enforces non-negativity (scrubbing non-finite values to zero
            first) and, when a footprint mask was supplied, restricts the
            model to lie inside the input footprints.

            Parameters
            ----------
            image :
                Candidate model array.

            Returns
            -------
            projected : `numpy.ndarray`
                The constrained model array.
            """
            # Scrub every non-finite value (NaN and +/-inf) to zero, matching
            # the original clamp, then enforce non-negativity. Leaving +inf as
            # the ``nan_to_num`` default (~1.8e308) would overflow when the
            # residual is squared for the loss.
            projected = np.clip(np.nan_to_num(image, posinf=0, neginf=0), 0, None)
            if footprintImage is not None:
                projected = projected * footprintImage
            return projected

        # Initialize either at zero or at the observed image, per config.
        if self.config.useZeroInit:
            x = np.zeros(observation.images.shape, dtype=dtype)
        else:
            x = observation.images.data.copy()

        # FISTA holds the iterate (and its momentum extrapolation) in a
        # `FistaParameter`; plain gradient ascent owns ``x`` directly.
        if self.config.useFista:
            parameter = scl.FistaParameter(
                x,
                step=step,
                grad=lambda input_grad, _x: input_grad,  # identity; update() needs a callable
                prox=prox,
            )
        else:
            parameter = None

        loss = []
        for n in range(self.config.maxIter):
            # cache=True reuses the FFT plan across iterations; the
            # image shape is stable inside the loop so this is a free
            # speedup at zero correctness cost.
            model = scl.Image(x, bands=(band,), yx0=yx0)
            residual = observation.images - observation.convolve(model, cache=True)
            if np.all(~np.isfinite(residual.data)):
                self.log.warning(f"Residual is non-finite at iteration {n}, stopping deconvolution")
                loss.append(-np.inf)
                break
            # Zero the residual in masked pixels (bad-mask planes and
            # non-finite pixels, both flagged by zero weight in
            # ``_buildObservation``) so they exert no pull on the fit. The
            # gradient is deliberately left unmasked: the convolution below
            # still propagates flux from good pixels into the model at masked
            # pixels, partially filling in the model there.
            residual.data[observation.weights.data == 0] = 0
            loss.append(-0.5 * np.nansum(residual.data**2))
            # A^T . residual is the +gradient of the log-likelihood (the
            # ascent direction).
            gradient = observation.convolve(residual, grad=True, cache=True)
            if parameter is not None:
                # FISTA's update descends (y = z - step . grad), so negate
                # the gradient to ascend the log-likelihood. ``prox`` is
                # applied inside ``update``.
                parameter.update(n, -np.nan_to_num(gradient.data))
                x = parameter.x
            else:
                # ``step`` is a Python float, so the arithmetic promotes to
                # float64; cast back so the model keeps the image dtype (the
                # FISTA path stays in dtype because ``update`` writes back
                # into its float32 buffer in place).
                x = prox(x + step * gradient.data).astype(dtype, copy=False)
                # Check for a diverging model. FISTA requires a constant
                # step, so step-halving is restricted to plain gradient
                # ascent.
                if len(loss) > 1 and loss[-1] < loss[-2]:
                    step = step / 2
                    self.log.warning(f"Loss increased at iteration {n}, decreasing scale to {step}")

            # Check for convergence
            if n > self.config.minIter and np.abs(loss[-1] - loss[-2]) < self.config.eRel * np.abs(loss[-1]):
                break

        return scl.Image(x, bands=(band,), yx0=yx0), loss

    def _modelToExposure(
        self,
        model: np.ndarray,
        coadd: afwImage.Exposure,
        modelPsf: scl.ImagePsf,
    ) -> afwImage.Exposure:
        """Convert a deconvolved image array to an Exposure.

        The output exposure's mask is a deep copy of the input coadd's
        mask, and its variance plane is fresh and filled with ``inf``.
        Convolution-then-deconvolution alters the per-pixel noise
        covariance, so the input coadd's variance no longer describes
        the deconvolved pixel values; the infinite variance signals
        "no information about the noise here" and naturally zero-weights
        these pixels under any inverse-variance scheme. Downstream
        consumers that need a variance plane must supply their own.

        The deconvolved image lives in scarlet's model frame, so its PSF
        is the narrow model PSF used during deconvolution rather than the
        input coadd's observed PSF. That model PSF is converted to an LSST
        `~lsst.afw.detection.Psf` and attached to the output exposure,
        overriding the observed PSF carried over in ``ExposureInfo``.

        Parameters
        ----------
        model :
            Deconvolved image array.
        coadd :
            Input coadd exposure; its image dtype, bbox, ``ExposureInfo``,
            and mask contents are reused.
        modelPsf :
            Scarlet lite model-frame PSF that the deconvolved image is
            matched to; attached to the output exposure as an LSST PSF.
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
        # Copy-construct a fresh ExposureInfo rather than sharing the
        # coadd's by reference: the components (WCS, filter, etc.) are
        # carried over, but the copy is decoupled so the setPsf below
        # swaps the output's PSF without mutating the input coadd's.
        exposureInfo = afwImage.ExposureInfo(coadd.getInfo())
        exposure = afwImage.Exposure(
            maskedImage=maskedImage,
            exposureInfo=exposureInfo,
            dtype=coadd.image.array.dtype,
        )
        # Replace the observed PSF inherited from the coadd's ExposureInfo
        # with the model-frame PSF the deconvolved image is matched to.
        exposure.setPsf(scarletImagePsfToLsst(modelPsf))
        return exposure
