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

from dataclasses import dataclass
from functools import partial
import logging
import numpy as np
import scarlet
from scarlet.psf import ImagePSF, GaussianPSF
from scarlet import Blend, Frame, Observation
from scarlet.renderer import ConvolutionRenderer
from scarlet.detect import get_detect_wavelets
from scarlet.initialization import init_all_sources
from scarlet import lite

import lsst.pex.config as pexConfig
from lsst.pex.exceptions import InvalidParameterError
import lsst.pipe.base as pipeBase
from lsst.geom import Point2I, Box2I, Point2D
import lsst.afw.geom.ellipses as afwEll
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.table as afwTable
from lsst.utils.logging import PeriodicLogger
from lsst.utils.timer import timeMethod

from .source import bboxToScarletBox, modelToHeavy, liteModelToHeavy

# Scarlet and proxmin have a different definition of log levels than the stack,
# so even "warnings" occur far more often than we would like.
# So for now we only display scarlet and proxmin errors, as all other
# scarlet outputs would be considered "TRACE" by our standards.
scarletLogger = logging.getLogger("scarlet")
scarletLogger.setLevel(logging.ERROR)
proxminLogger = logging.getLogger("proxmin")
proxminLogger.setLevel(logging.ERROR)

__all__ = ["deblend", "ScarletDeblendConfig", "ScarletDeblendTask"]

logger = logging.getLogger(__name__)


class IncompleteDataError(Exception):
    """The PSF could not be computed due to incomplete data
    """
    pass


class ScarletGradientError(Exception):
    """An error occurred during optimization

    This error occurs when the optimizer encounters
    a NaN value while calculating the gradient.
    """
    def __init__(self, iterations, sources):
        self.iterations = iterations
        self.sources = sources
        msg = ("ScalarGradientError in iteration {0}. "
               "NaN values introduced in sources {1}")
        self.message = msg.format(iterations, sources)

    def __str__(self):
        return self.message


def _checkBlendConvergence(blend, f_rel):
    """Check whether or not a blend has converged
    """
    deltaLoss = np.abs(blend.loss[-2] - blend.loss[-1])
    convergence = f_rel * np.abs(blend.loss[-1])
    return deltaLoss < convergence


def _computePsfImage(self, position):
    """Get a multiband PSF image
    The PSF Kernel Image is computed for each band
    and combined into a (filter, y, x) array and stored
    as `self._psfImage`.
    The result is not cached, so if the same PSF is expected
    to be used multiple times it is a good idea to store the
    result in another variable.
    Note: this is a temporary fix during the deblender sprint.
    In the future this function will replace the current method
    in `afw.MultibandExposure.computePsfImage` (DM-19789).
    Parameters
    ----------
    position : `Point2D` or `tuple`
        Coordinates to evaluate the PSF.
    Returns
    -------
    self._psfImage: array
        The multiband PSF image.
    """
    psfs = []
    # Make the coordinates into a Point2D (if necessary)
    if not isinstance(position, Point2D):
        position = Point2D(position[0], position[1])

    for bidx, single in enumerate(self.singles):
        try:
            psf = single.getPsf().computeKernelImage(position)
            psfs.append(psf)
        except InvalidParameterError:
            # This band failed to compute the PSF due to incomplete data
            # at that location. This is unlikely to be a problem for Rubin,
            # however the edges of some HSC COSMOS fields contain incomplete
            # data in some bands, so we track this error to distinguish it
            # from unknown errors.
            msg = "Failed to compute PSF at {} in band {}"
            raise IncompleteDataError(msg.format(position, self.filters[bidx]))

    left = np.min([psf.getBBox().getMinX() for psf in psfs])
    bottom = np.min([psf.getBBox().getMinY() for psf in psfs])
    right = np.max([psf.getBBox().getMaxX() for psf in psfs])
    top = np.max([psf.getBBox().getMaxY() for psf in psfs])
    bbox = Box2I(Point2I(left, bottom), Point2I(right, top))
    psfs = [afwImage.utils.projectImage(psf, bbox) for psf in psfs]
    psfImage = afwImage.MultibandImage.fromImages(self.filters, psfs)
    return psfImage


def getFootprintMask(footprint, mExposure):
    """Mask pixels outside the footprint

    Parameters
    ----------
    mExposure : `lsst.image.MultibandExposure`
        - The multiband exposure containing the image,
          mask, and variance data
    footprint : `lsst.detection.Footprint`
        - The footprint of the parent to deblend

    Returns
    -------
    footprintMask : array
        Boolean array with pixels not in the footprint set to one.
    """
    bbox = footprint.getBBox()
    fpMask = afwImage.Mask(bbox)
    footprint.spans.setMask(fpMask, 1)
    fpMask = ~fpMask.getArray().astype(bool)
    return fpMask


def isPseudoSource(source, pseudoColumns):
    """Check if a source is a pseudo source.

    This is mostly for skipping sky objects,
    but any other column can also be added to disable
    deblending on a parent or individual source when
    set to `True`.

    Parameters
    ----------
    source : `lsst.afw.table.source.source.SourceRecord`
        The source to check for the pseudo bit.
    pseudoColumns : `list` of `str`
        A list of columns to check for pseudo sources.
    """
    isPseudo = False
    for col in pseudoColumns:
        try:
            isPseudo |= source[col]
        except KeyError:
            pass
    return isPseudo


def deblend(mExposure, footprint, config):
    """Deblend a parent footprint

    Parameters
    ----------
    mExposure : `lsst.image.MultibandExposure`
        - The multiband exposure containing the image,
          mask, and variance data
    footprint : `lsst.detection.Footprint`
        - The footprint of the parent to deblend
    config : `ScarletDeblendConfig`
        - Configuration of the deblending task

    Returns
    -------
    blend : `scarlet.Blend`
        The scarlet blend class that contains all of the information
        about the parameters and results from scarlet
    skipped : `list` of `int`
        The indices of any children that failed to initialize
        and were skipped.
    spectrumInit : `bool`
        Whether or not all of the sources were initialized by jointly
        fitting their SED's. This provides a better initialization
        but created memory issues when a blend is too large or
        contains too many sources.
    """
    # Extract coordinates from each MultiColorPeak
    bbox = footprint.getBBox()

    # Create the data array from the masked images
    images = mExposure.image[:, bbox].array

    # Use the inverse variance as the weights
    if config.useWeights:
        weights = 1/mExposure.variance[:, bbox].array
    else:
        weights = np.ones_like(images)
        badPixels = mExposure.mask.getPlaneBitMask(config.badMask)
        mask = mExposure.mask[:, bbox].array & badPixels
        weights[mask > 0] = 0

    # Mask out the pixels outside the footprint
    mask = getFootprintMask(footprint, mExposure)
    weights *= ~mask

    psfs = _computePsfImage(mExposure, footprint.getCentroid()).array.astype(np.float32)
    psfs = ImagePSF(psfs)
    model_psf = GaussianPSF(sigma=(config.modelPsfSigma,)*len(mExposure.filters))

    frame = Frame(images.shape, psf=model_psf, channels=mExposure.filters)
    observation = Observation(images, psf=psfs, weights=weights, channels=mExposure.filters)
    if config.convolutionType == "fft":
        observation.match(frame)
    elif config.convolutionType == "real":
        renderer = ConvolutionRenderer(observation, frame, convolution_type="real")
        observation.match(frame, renderer=renderer)
    else:
        raise ValueError("Unrecognized convolution type {}".format(config.convolutionType))

    assert(config.sourceModel in ["single", "double", "compact", "fit"])

    # Set the appropriate number of components
    if config.sourceModel == "single":
        maxComponents = 1
    elif config.sourceModel == "double":
        maxComponents = 2
    elif config.sourceModel == "compact":
        maxComponents = 0
    elif config.sourceModel == "point":
        raise NotImplementedError("Point source photometry is currently not implemented")
    elif config.sourceModel == "fit":
        # It is likely in the future that there will be some heuristic
        # used to determine what type of model to use for each source,
        # but that has not yet been implemented (see DM-22551)
        raise NotImplementedError("sourceModel 'fit' has not been implemented yet")

    # Convert the centers to pixel coordinates
    xmin = bbox.getMinX()
    ymin = bbox.getMinY()
    centers = [
        np.array([peak.getIy() - ymin, peak.getIx() - xmin], dtype=int)
        for peak in footprint.peaks
        if not isPseudoSource(peak, config.pseudoColumns)
    ]

    # Choose whether or not to use the improved spectral initialization
    if config.setSpectra:
        if config.maxSpectrumCutoff <= 0:
            spectrumInit = True
        else:
            spectrumInit = len(centers) * bbox.getArea() < config.maxSpectrumCutoff
    else:
        spectrumInit = False

    # Only deblend sources that can be initialized
    sources, skipped = init_all_sources(
        frame=frame,
        centers=centers,
        observations=observation,
        thresh=config.morphThresh,
        max_components=maxComponents,
        min_snr=config.minSNR,
        shifting=False,
        fallback=config.fallback,
        silent=config.catchFailures,
        set_spectra=spectrumInit,
    )

    # Attach the peak to all of the initialized sources
    srcIndex = 0
    for k, center in enumerate(centers):
        if k not in skipped:
            # This is just to make sure that there isn't a coding bug
            assert np.all(sources[srcIndex].center == center)
            # Store the record for the peak with the appropriate source
            sources[srcIndex].detectedPeak = footprint.peaks[k]
            srcIndex += 1

    # Create the blend and attempt to optimize it
    blend = Blend(sources, observation)
    try:
        blend.fit(max_iter=config.maxIter, e_rel=config.relativeError)
    except ArithmeticError:
        # This occurs when a gradient update produces a NaN value
        # This is usually due to a source initialized with a
        # negative SED or no flux, often because the peak
        # is a noise fluctuation in one band and not a real source.
        iterations = len(blend.loss)
        failedSources = []
        for k, src in enumerate(sources):
            if np.any(~np.isfinite(src.get_model())):
                failedSources.append(k)
        raise ScarletGradientError(iterations, failedSources)

    return blend, skipped, spectrumInit


def deblend_lite(mExposure, footprint, config, wavelets=None):
    """Deblend a parent footprint

    Parameters
    ----------
    mExposure : `lsst.image.MultibandExposure`
        - The multiband exposure containing the image,
          mask, and variance data
    footprint : `lsst.detection.Footprint`
        - The footprint of the parent to deblend
    config : `ScarletDeblendConfig`
        - Configuration of the deblending task
    """
    # Extract coordinates from each MultiColorPeak
    bbox = footprint.getBBox()

    # Create the data array from the masked images
    images = mExposure.image[:, bbox].array
    variance = mExposure.variance[:, bbox].array

    # Use the inverse variance as the weights
    if config.useWeights:
        weights = 1/mExposure.variance[:, bbox].array
    else:
        weights = np.ones_like(images)
        badPixels = mExposure.mask.getPlaneBitMask(config.badMask)
        mask = mExposure.mask[:, bbox].array & badPixels
        weights[mask > 0] = 0

    # Mask out the pixels outside the footprint
    mask = getFootprintMask(footprint, mExposure)
    weights *= ~mask

    psfs = _computePsfImage(mExposure, footprint.getCentroid()).array.astype(np.float32)
    modelPsf = lite.integrated_circular_gaussian(sigma=config.modelPsfSigma)

    observation = lite.LiteObservation(
        images=images,
        variance=variance,
        weights=weights,
        psfs=psfs,
        model_psf=modelPsf[None, :, :],
        convolution_mode=config.convolutionType,
    )

    # Convert the centers to pixel coordinates
    xmin = bbox.getMinX()
    ymin = bbox.getMinY()
    centers = [
        np.array([peak.getIy() - ymin, peak.getIx() - xmin], dtype=int)
        for peak in footprint.peaks
        if not isPseudoSource(peak, config.pseudoColumns)
    ]

    # Initialize the sources
    if config.morphImage == "chi2":
        sources = lite.init_all_sources_main(
            observation,
            centers,
            min_snr=config.minSNR,
            thresh=config.morphThresh,
        )
    elif config.morphImage == "wavelet":
        _bbox = bboxToScarletBox(len(mExposure.filters), bbox, bbox.getMin())
        _wavelets = wavelets[(slice(None), *_bbox[1:].slices)]
        sources = lite.init_all_sources_wavelets(
            observation,
            centers,
            use_psf=False,
            wavelets=_wavelets,
            min_snr=config.minSNR,
        )
    else:
        raise ValueError("morphImage must be either 'chi2' or 'wavelet'.")

    # Set the optimizer
    if config.optimizer == "adaprox":
        parameterization = partial(
            lite.init_adaprox_component,
            bg_thresh=config.backgroundThresh,
            max_prox_iter=config.maxProxIter,
        )
    elif config.optimizer == "fista":
        parameterization = partial(
            lite.init_fista_component,
            bg_thresh=config.backgroundThresh,
        )
    else:
        raise ValueError("Unrecognized optimizer. Must be either 'adaprox' or 'fista'.")
    sources = lite.parameterize_sources(sources, observation, parameterization)

    # Attach the peak to all of the initialized sources
    for k, center in enumerate(centers):
        # This is just to make sure that there isn't a coding bug
        if len(sources[k].components) > 0 and np.any(sources[k].center != center):
            raise ValueError("Misaligned center, expected {center} but got {sources[k].center}")
        # Store the record for the peak with the appropriate source
        sources[k].detectedPeak = footprint.peaks[k]

    blend = lite.LiteBlend(sources, observation)

    # Initialize each source with its best fit spectrum
    # This significantly cuts down on the number of iterations
    # that the optimizer needs and usually results in a better
    # fit, but using least squares on a very large blend causes memory issues.
    # This is typically the most expensive operation in deblending, memorywise.
    spectrumInit = False
    if config.setSpectra:
        if config.maxSpectrumCutoff <= 0 or len(centers) * bbox.getArea() < config.maxSpectrumCutoff:
            spectrumInit = True
            blend.fit_spectra()

    # Set the sources that could not be initialized and were skipped
    skipped = [src for src in sources if src.is_null]

    blend.fit(max_iter=config.maxIter, e_rel=config.relativeError, min_iter=config.minIter)

    return blend, skipped, spectrumInit


@dataclass
class DeblenderMetrics:
    """Metrics and measurements made on single sources.

    Store deblender metrics to be added as attributes to a scarlet source
    before it is converted into a `SourceRecord`.
    When DM-34414 is finished this class will be eliminated and the metrics
    will be added to the schema using a pipeline task that calculates them
    from the stored deconvolved models.

    All of the parameters are one dimensional numpy arrays,
    with an element for each band in the observed images.

    `maxOverlap` is useful as a metric for determining how blended a source
    is because if it only overlaps with other sources at or below
    the noise level, it is likely to be a mostly isolated source
    in the deconvolved model frame.

    `fluxOverlapFraction` is potentially more useful than the canonical
    "blendedness" (or purity) metric because it accounts for potential
    biases created during deblending by not weighting the overlapping
    flux with the flux of this sources model.

    Attributes
    ----------
    maxOverlap : `numpy.ndarray`
        The maximum overlap that the source has with its neighbors in
        a single pixel.
    fluxOverlap : `numpy.ndarray`
        The total flux from neighbors overlapping with the current source.
    fluxOverlapFraction : `numpy.ndarray`
        The fraction of `flux from neighbors/source flux` for a
        given source within the source's footprint.
    blendedness : `numpy.ndarray`
        The metric for determining how blended a source is using the
        Bosch et al. 2018 metric for "blendedness." Note that some
        surveys use the term "purity," which is `1-blendedness`.
    """
    maxOverlap: np.array
    fluxOverlap: np.array
    fluxOverlapFraction: np.array
    blendedness: np.array


def setDeblenderMetrics(blend):
    """Set metrics that can be used to evalute the deblender accuracy

    This function calculates the `DeblenderMetrics` for each source in the
    blend, and assigns it to that sources `metrics` property in place.

    Parameters
    ----------
    blend : `scarlet.lite.Blend`
        The blend containing the sources to measure.
    """
    # Store the full model of the scene for comparison
    blendModel = blend.get_model()
    for k, src in enumerate(blend.sources):
        # Extract the source model in the full bounding box
        model = src.get_model(bbox=blend.bbox)
        # The footprint is the 2D array of non-zero pixels in each band
        footprint = np.bitwise_or.reduce(model > 0, axis=0)
        # Calculate the metrics.
        # See `DeblenderMetrics` for a description of each metric.
        neighborOverlap = (blendModel-model) * footprint[None, :, :]
        maxOverlap = np.max(neighborOverlap, axis=(1, 2))
        fluxOverlap = np.sum(neighborOverlap, axis=(1, 2))
        fluxModel = np.sum(model, axis=(1, 2))
        fluxOverlapFraction = np.zeros((len(model), ), dtype=float)
        isFinite = fluxModel > 0
        fluxOverlapFraction[isFinite] = fluxOverlap[isFinite]/fluxModel[isFinite]
        blendedness = 1 - np.sum(model*model, axis=(1, 2))/np.sum(blendModel*model, axis=(1, 2))
        src.metrics = DeblenderMetrics(maxOverlap, fluxOverlap, fluxOverlapFraction, blendedness)


class ScarletDeblendConfig(pexConfig.Config):
    """MultibandDeblendConfig

    Configuration for the multiband deblender.
    The parameters are organized by the parameter types, which are
    - Stopping Criteria: Used to determine if the fit has converged
    - Position Fitting Criteria: Used to fit the positions of the peaks
    - Constraints: Used to apply constraints to the peaks and their components
    - Other: Parameters that don't fit into the above categories
    """
    # Stopping Criteria
    minIter = pexConfig.Field(dtype=int, default=1,
                              doc="Minimum number of iterations before the optimizer is allowed to stop.")
    maxIter = pexConfig.Field(dtype=int, default=300,
                              doc=("Maximum number of iterations to deblend a single parent"))
    relativeError = pexConfig.Field(dtype=float, default=1e-2,
                                    doc=("Change in the loss function between iterations to exit fitter. "
                                         "Typically this is `1e-2` if measurements will be made on the "
                                         "flux re-distributed models and `1e-4` when making measurements "
                                         "on the models themselves."))

    # Constraints
    morphThresh = pexConfig.Field(dtype=float, default=1,
                                  doc="Fraction of background RMS a pixel must have"
                                      "to be included in the initial morphology")
    # Lite Parameters
    # All of these parameters (except version) are only valid if version='lite'
    version = pexConfig.ChoiceField(
        dtype=str,
        default="lite",
        allowed={
            "scarlet": "main scarlet version (likely to be deprecated soon)",
            "lite": "Optimized version of scarlet for survey data from a single instrument",
        },
        doc="The version of scarlet to use.",
    )
    optimizer = pexConfig.ChoiceField(
        dtype=str,
        default="adaprox",
        allowed={
            "adaprox": "Proximal ADAM optimization",
            "fista": "Accelerated proximal gradient method",
        },
        doc="The optimizer to use for fitting parameters and is only used when version='lite'",
    )
    morphImage = pexConfig.ChoiceField(
        dtype=str,
        default="chi2",
        allowed={
            "chi2": "Initialize sources on a chi^2 image made from all available bands",
            "wavelet": "Initialize sources using a wavelet decomposition of the chi^2 image",
        },
        doc="The type of image to use for initializing the morphology. "
            "Must be either 'chi2' or 'wavelet'. "
    )
    backgroundThresh = pexConfig.Field(
        dtype=float,
        default=0.25,
        doc="Fraction of background to use for a sparsity threshold. "
            "This prevents sources from growing unrealistically outside "
            "the parent footprint while still modeling flux correctly "
            "for bright sources."
    )
    maxProxIter = pexConfig.Field(
        dtype=int,
        default=1,
        doc="Maximum number of proximal operator iterations inside of each "
            "iteration of the optimizer. "
            "This config field is only used if version='lite' and optimizer='adaprox'."
    )
    waveletScales = pexConfig.Field(
        dtype=int,
        default=5,
        doc="Number of wavelet scales to use for wavelet initialization. "
            "This field is only used when `version`='lite' and `morphImage`='wavelet'."
    )

    # Other scarlet paremeters
    useWeights = pexConfig.Field(
        dtype=bool, default=True,
        doc=("Whether or not use use inverse variance weighting."
             "If `useWeights` is `False` then flat weights are used"))
    modelPsfSize = pexConfig.Field(
        dtype=int, default=11,
        doc="Model PSF side length in pixels")
    modelPsfSigma = pexConfig.Field(
        dtype=float, default=0.8,
        doc="Define sigma for the model frame PSF")
    minSNR = pexConfig.Field(
        dtype=float, default=50,
        doc="Minimum Signal to noise to accept the source."
            "Sources with lower flux will be initialized with the PSF but updated "
            "like an ordinary ExtendedSource (known in scarlet as a `CompactSource`).")
    saveTemplates = pexConfig.Field(
        dtype=bool, default=True,
        doc="Whether or not to save the SEDs and templates")
    processSingles = pexConfig.Field(
        dtype=bool, default=True,
        doc="Whether or not to process isolated sources in the deblender")
    convolutionType = pexConfig.Field(
        dtype=str, default="fft",
        doc="Type of convolution to render the model to the observations.\n"
            "- 'fft': perform convolutions in Fourier space\n"
            "- 'real': peform convolutions in real space.")
    sourceModel = pexConfig.Field(
        dtype=str, default="double",
        doc=("How to determine which model to use for sources, from\n"
             "- 'single': use a single component for all sources\n"
             "- 'double': use a bulge disk model for all sources\n"
             "- 'compact': use a single component model, initialzed with a point source morphology, "
             " for all sources\n"
             "- 'point': use a point-source model for all sources\n"
             "- 'fit: use a PSF fitting model to determine the number of components (not yet implemented)"),
        deprecated="This field will be deprecated when the default for `version` is changed to `lite`.",
    )
    setSpectra = pexConfig.Field(
        dtype=bool, default=True,
        doc="Whether or not to solve for the best-fit spectra during initialization. "
            "This makes initialization slightly longer, as it requires a convolution "
            "to set the optimal spectra, but results in a much better initial log-likelihood "
            "and reduced total runtime, with convergence in fewer iterations."
            "This option is only used when "
            "peaks*area < `maxSpectrumCutoff` will use the improved initialization.")

    # Mask-plane restrictions
    badMask = pexConfig.ListField(
        dtype=str, default=["BAD", "CR", "NO_DATA", "SAT", "SUSPECT", "EDGE"],
        doc="Whether or not to process isolated sources in the deblender")
    statsMask = pexConfig.ListField(dtype=str, default=["SAT", "INTRP", "NO_DATA"],
                                    doc="Mask planes to ignore when performing statistics")
    maskLimits = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        default={},
        doc=("Mask planes with the corresponding limit on the fraction of masked pixels. "
             "Sources violating this limit will not be deblended. "
             "If the fraction is `0` then the limit is a single pixel."),
    )

    # Size restrictions
    maxNumberOfPeaks = pexConfig.Field(
        dtype=int, default=200,
        doc=("Only deblend the brightest maxNumberOfPeaks peaks in the parent"
             " (<= 0: unlimited)"))
    maxFootprintArea = pexConfig.Field(
        dtype=int, default=100_000,
        doc=("Maximum area for footprints before they are ignored as large; "
             "non-positive means no threshold applied"))
    maxAreaTimesPeaks = pexConfig.Field(
        dtype=int, default=10_000_000,
        doc=("Maximum rectangular footprint area * nPeaks in the footprint. "
             "This was introduced in DM-33690 to prevent fields that are crowded or have a "
             "LSB galaxy that causes memory intensive initialization in scarlet from dominating "
             "the overall runtime and/or causing the task to run out of memory. "
             "(<= 0: unlimited)")
    )
    maxFootprintSize = pexConfig.Field(
        dtype=int, default=0,
        doc=("Maximum linear dimension for footprints before they are ignored "
             "as large; non-positive means no threshold applied"))
    minFootprintAxisRatio = pexConfig.Field(
        dtype=float, default=0.0,
        doc=("Minimum axis ratio for footprints before they are ignored "
             "as large; non-positive means no threshold applied"))
    maxSpectrumCutoff = pexConfig.Field(
        dtype=int, default=1_000_000,
        doc=("Maximum number of pixels * number of sources in a blend. "
             "This is different than `maxFootprintArea` because this isn't "
             "the footprint area but the area of the bounding box that "
             "contains the footprint, and is also multiplied by the number of"
             "sources in the footprint. This prevents large skinny blends with "
             "a high density of sources from running out of memory. "
             "If `maxSpectrumCutoff == -1` then there is no cutoff.")
    )
    # Failure modes
    fallback = pexConfig.Field(
        dtype=bool, default=True,
        doc="Whether or not to fallback to a smaller number of components if a source does not initialize"
    )
    notDeblendedMask = pexConfig.Field(
        dtype=str, default="NOT_DEBLENDED", optional=True,
        doc="Mask name for footprints not deblended, or None")
    catchFailures = pexConfig.Field(
        dtype=bool, default=True,
        doc=("If True, catch exceptions thrown by the deblender, log them, "
             "and set a flag on the parent, instead of letting them propagate up"))

    # Other options
    columnInheritance = pexConfig.DictField(
        keytype=str, itemtype=str, default={
            "deblend_nChild": "deblend_parentNChild",
            "deblend_nPeaks": "deblend_parentNPeaks",
            "deblend_spectrumInitFlag": "deblend_spectrumInitFlag",
            "deblend_blendConvergenceFailedFlag": "deblend_blendConvergenceFailedFlag",
        },
        doc="Columns to pass from the parent to the child. "
            "The key is the name of the column for the parent record, "
            "the value is the name of the column to use for the child."
    )
    pseudoColumns = pexConfig.ListField(
        dtype=str, default=['merge_peak_sky', 'sky_source'],
        doc="Names of flags which should never be deblended."
    )

    # Logging option(s)
    loggingInterval = pexConfig.Field(
        dtype=int, default=600,
        doc="Interval (in seconds) to log messages (at VERBOSE level) while deblending sources.",
        deprecated="This field is no longer used and will be removed in v25.",
    )
    # Testing options
    # Some obs packages and ci packages run the full pipeline on a small
    # subset of data to test that the pipeline is functioning properly.
    # This is not meant as scientific validation, so it can be useful
    # to only run on a small subset of the data that is large enough to
    # test the desired pipeline features but not so long that the deblender
    # is the tall pole in terms of execution times.
    useCiLimits = pexConfig.Field(
        dtype=bool, default=False,
        doc="Limit the number of sources deblended for CI to prevent long build times")
    ciDeblendChildRange = pexConfig.ListField(
        dtype=int, default=[5, 10],
        doc="Only deblend parent Footprints with a number of peaks in the (inclusive) range indicated."
            "If `useCiLimits==False` then this parameter is ignored.")
    ciNumParentsToDeblend = pexConfig.Field(
        dtype=int, default=10,
        doc="Only use the first `ciNumParentsToDeblend` parent footprints with a total peak count "
            "within `ciDebledChildRange`. "
            "If `useCiLimits==False` then this parameter is ignored.")


class ScarletDeblendTask(pipeBase.Task):
    """ScarletDeblendTask

    Split blended sources into individual sources.

    This task has no return value; it only modifies the SourceCatalog in-place.
    """
    ConfigClass = ScarletDeblendConfig
    _DefaultName = "scarletDeblend"

    def __init__(self, schema, peakSchema=None, **kwargs):
        """Create the task, adding necessary fields to the given schema.

        Parameters
        ----------
        schema : `lsst.afw.table.schema.schema.Schema`
            Schema object for measurement fields; will be modified in-place.
        peakSchema : `lsst.afw.table.schema.schema.Schema`
            Schema of Footprint Peaks that will be passed to the deblender.
            Any fields beyond the PeakTable minimal schema will be transferred
            to the main source Schema.  If None, no fields will be transferred
            from the Peaks.
        filters : list of str
            Names of the filters used for the eposures. This is needed to store
            the SED as a field
        **kwargs
            Passed to Task.__init__.
        """
        pipeBase.Task.__init__(self, **kwargs)

        peakMinimalSchema = afwDet.PeakTable.makeMinimalSchema()
        if peakSchema is None:
            # In this case, the peakSchemaMapper will transfer nothing, but
            # we'll still have one
            # to simplify downstream code
            self.peakSchemaMapper = afwTable.SchemaMapper(peakMinimalSchema, schema)
        else:
            self.peakSchemaMapper = afwTable.SchemaMapper(peakSchema, schema)
            for item in peakSchema:
                if item.key not in peakMinimalSchema:
                    self.peakSchemaMapper.addMapping(item.key, item.field)
                    # Because SchemaMapper makes a copy of the output schema
                    # you give its ctor, it isn't updating this Schema in
                    # place. That's probably a design flaw, but in the
                    # meantime, we'll keep that schema in sync with the
                    # peakSchemaMapper.getOutputSchema() manually, by adding
                    # the same fields to both.
                    schema.addField(item.field)
            assert schema == self.peakSchemaMapper.getOutputSchema(), "Logic bug mapping schemas"
        self._addSchemaKeys(schema)
        self.schema = schema
        self.toCopyFromParent = [item.key for item in self.schema
                                 if item.field.getName().startswith("merge_footprint")]

    def _addSchemaKeys(self, schema):
        """Add deblender specific keys to the schema
        """
        # Parent (blend) fields
        self.runtimeKey = schema.addField('deblend_runtime', type=np.float32, doc='runtime in ms')
        self.iterKey = schema.addField('deblend_iterations', type=np.int32, doc='iterations to converge')
        self.nChildKey = schema.addField('deblend_nChild', type=np.int32,
                                         doc='Number of children this object has (defaults to 0)')
        self.nPeaksKey = schema.addField("deblend_nPeaks", type=np.int32,
                                         doc="Number of initial peaks in the blend. "
                                             "This includes peaks that may have been culled "
                                             "during deblending or failed to deblend")
        # Skipped flags
        self.deblendSkippedKey = schema.addField('deblend_skipped', type='Flag',
                                                 doc="Deblender skipped this source")
        self.isolatedParentKey = schema.addField('deblend_isolatedParent', type='Flag',
                                                 doc='The source has only a single peak '
                                                     'and was not deblended')
        self.pseudoKey = schema.addField('deblend_isPseudo', type='Flag',
                                         doc='The source is identified as a "pseudo" source and '
                                             'was not deblended')
        self.tooManyPeaksKey = schema.addField('deblend_tooManyPeaks', type='Flag',
                                               doc='Source had too many peaks; '
                                               'only the brightest were included')
        self.tooBigKey = schema.addField('deblend_parentTooBig', type='Flag',
                                         doc='Parent footprint covered too many pixels')
        self.maskedKey = schema.addField('deblend_masked', type='Flag',
                                         doc='Parent footprint had too many masked pixels')
        # Convergence flags
        self.sedNotConvergedKey = schema.addField('deblend_sedConvergenceFailed', type='Flag',
                                                  doc='scarlet sed optimization did not converge before'
                                                      'config.maxIter')
        self.morphNotConvergedKey = schema.addField('deblend_morphConvergenceFailed', type='Flag',
                                                    doc='scarlet morph optimization did not converge before'
                                                        'config.maxIter')
        self.blendConvergenceFailedFlagKey = schema.addField('deblend_blendConvergenceFailedFlag',
                                                             type='Flag',
                                                             doc='at least one source in the blend'
                                                                 'failed to converge')
        # Error flags
        self.deblendFailedKey = schema.addField('deblend_failed', type='Flag',
                                                doc="Deblending failed on source")
        self.deblendErrorKey = schema.addField('deblend_error', type="String", size=25,
                                               doc='Name of error if the blend failed')
        # Deblended source fields
        self.peakCenter = afwTable.Point2IKey.addFields(schema, name="deblend_peak_center",
                                                        doc="Center used to apply constraints in scarlet",
                                                        unit="pixel")
        self.peakIdKey = schema.addField("deblend_peakId", type=np.int32,
                                         doc="ID of the peak in the parent footprint. "
                                             "This is not unique, but the combination of 'parent'"
                                             "and 'peakId' should be for all child sources. "
                                             "Top level blends with no parents have 'peakId=0'")
        self.modelCenterFlux = schema.addField('deblend_peak_instFlux', type=float, units='count',
                                               doc="The instFlux at the peak position of deblended mode")
        self.modelTypeKey = schema.addField("deblend_modelType", type="String", size=25,
                                            doc="The type of model used, for example "
                                                "MultiExtendedSource, SingleExtendedSource, PointSource")
        self.parentNPeaksKey = schema.addField("deblend_parentNPeaks", type=np.int32,
                                               doc="deblend_nPeaks from this records parent.")
        self.parentNChildKey = schema.addField("deblend_parentNChild", type=np.int32,
                                               doc="deblend_nChild from this records parent.")
        self.scarletFluxKey = schema.addField("deblend_scarletFlux", type=np.float32,
                                              doc="Flux measurement from scarlet")
        self.scarletLogLKey = schema.addField("deblend_logL", type=np.float32,
                                              doc="Final logL, used to identify regressions in scarlet.")
        self.edgePixelsKey = schema.addField('deblend_edgePixels', type='Flag',
                                             doc='Source had flux on the edge of the parent footprint')
        self.scarletSpectrumInitKey = schema.addField("deblend_spectrumInitFlag", type='Flag',
                                                      doc="True when scarlet initializes sources "
                                                          "in the blend with a more accurate spectrum. "
                                                          "The algorithm uses a lot of memory, "
                                                          "so large dense blends will use "
                                                          "a less accurate initialization.")
        self.nComponentsKey = schema.addField("deblend_nComponents", type=np.int32,
                                              doc="Number of components in a ScarletLiteSource. "
                                                  "If `config.version != 'lite'`then "
                                                  "this column is set to zero.")
        self.psfKey = schema.addField('deblend_deblendedAsPsf', type='Flag',
                                      doc='Deblender thought this source looked like a PSF')
        # Blendedness/classification metrics
        self.maxOverlapKey = schema.addField("deblend_maxOverlap", type=np.float32,
                                             doc="Maximum overlap with all of the other neighbors flux "
                                                 "combined."
                                                 "This is useful as a metric for determining how blended a "
                                                 "source is because if it only overlaps with other sources "
                                                 "at or below the noise level, it is likely to be a mostly "
                                                 "isolated source in the deconvolved model frame.")
        self.fluxOverlapKey = schema.addField("deblend_fluxOverlap", type=np.float32,
                                              doc="This is the total flux from neighboring objects that "
                                                  "overlaps with this source.")
        self.fluxOverlapFractionKey = schema.addField("deblend_fluxOverlapFraction", type=np.float32,
                                                      doc="This is the fraction of "
                                                          "`flux from neighbors/source flux` "
                                                          "for a given source within the source's"
                                                          "footprint.")
        self.blendednessKey = schema.addField("deblend_blendedness", type=np.float32,
                                              doc="The Bosch et al. 2018 metric for 'blendedness.' ")

    @timeMethod
    def run(self, mExposure, mergedSources):
        """Get the psf from each exposure and then run deblend().

        Parameters
        ----------
        mExposure : `MultibandExposure`
            The exposures should be co-added images of the same
            shape and region of the sky.
        mergedSources : `SourceCatalog`
            The merged `SourceCatalog` that contains parent footprints
            to (potentially) deblend.

        Returns
        -------
        templateCatalogs: dict
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are catalogs with heavy footprints that are the templates
            created by the multiband templates.
        """
        return self.deblend(mExposure, mergedSources)

    @timeMethod
    def deblend(self, mExposure, catalog):
        """Deblend a data cube of multiband images

        Parameters
        ----------
        mExposure : `MultibandExposure`
            The exposures should be co-added images of the same
            shape and region of the sky.
        catalog : `SourceCatalog`
            The merged `SourceCatalog` that contains parent footprints
            to (potentially) deblend. The new deblended sources are
            appended to this catalog in place.

        Returns
        -------
        catalogs : `dict` or `None`
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are catalogs with heavy footprints that are the templates
            created by the multiband templates.
        """
        import time

        # Cull footprints if required by ci
        if self.config.useCiLimits:
            self.log.info("Using CI catalog limits, the original number of sources to deblend was %d.",
                          len(catalog))
            # Select parents with a number of children in the range
            # config.ciDeblendChildRange
            minChildren, maxChildren = self.config.ciDeblendChildRange
            nPeaks = np.array([len(src.getFootprint().peaks) for src in catalog])
            childrenInRange = np.where((nPeaks >= minChildren) & (nPeaks <= maxChildren))[0]
            if len(childrenInRange) < self.config.ciNumParentsToDeblend:
                raise ValueError("Fewer than ciNumParentsToDeblend children were contained in the range "
                                 "indicated by ciDeblendChildRange. Adjust this range to include more "
                                 "parents.")
            # Keep all of the isolated parents and the first
            # `ciNumParentsToDeblend` children
            parents = nPeaks == 1
            children = np.zeros((len(catalog),), dtype=bool)
            children[childrenInRange[:self.config.ciNumParentsToDeblend]] = True
            catalog = catalog[parents | children]
            # We need to update the IdFactory, otherwise the the source ids
            # will not be sequential
            idFactory = catalog.getIdFactory()
            maxId = np.max(catalog["id"])
            idFactory.notify(maxId)

        filters = mExposure.filters
        self.log.info("Deblending %d sources in %d exposure bands", len(catalog), len(mExposure))
        periodicLog = PeriodicLogger(self.log)

        # Create a set of wavelet coefficients if using wavelet initialization
        if self.config.version == "lite" and self.config.morphImage == "wavelet":
            images = mExposure.image.array
            variance = mExposure.variance.array
            wavelets = get_detect_wavelets(images, variance, scales=self.config.waveletScales)
        else:
            wavelets = None

        # Add the NOT_DEBLENDED mask to the mask plane in each band
        if self.config.notDeblendedMask:
            for mask in mExposure.mask:
                mask.addMaskPlane(self.config.notDeblendedMask)

        nParents = len(catalog)
        nDeblendedParents = 0
        skippedParents = []
        multibandColumns = {
            "heavies": [],
            "fluxes": [],
            "centerFluxes": [],
            "metrics": [],
        }
        weightedColumns = {
            "heavies": [],
            "fluxes": [],
            "centerFluxes": [],
        }
        for parentIndex in range(nParents):
            parent = catalog[parentIndex]
            foot = parent.getFootprint()
            bbox = foot.getBBox()
            peaks = foot.getPeaks()

            # Since we use the first peak for the parent object, we should
            # propagate its flags to the parent source.
            parent.assign(peaks[0], self.peakSchemaMapper)

            # Block of conditions for skipping a parent with multiple children
            if (skipArgs := self._checkSkipped(parent, mExposure)) is not None:
                self._skipParent(parent, *skipArgs)
                skippedParents.append(parentIndex)
                continue

            nDeblendedParents += 1
            self.log.trace("Parent %d: deblending %d peaks", parent.getId(), len(peaks))
            # Run the deblender
            blendError = None
            try:
                t0 = time.monotonic()
                # Build the parameter lists with the same ordering
                if self.config.version == "scarlet":
                    blend, skipped, spectrumInit = deblend(mExposure, foot, self.config)
                elif self.config.version == "lite":
                    blend, skipped, spectrumInit = deblend_lite(mExposure, foot, self.config, wavelets)
                tf = time.monotonic()
                runtime = (tf-t0)*1000
                converged = _checkBlendConvergence(blend, self.config.relativeError)
                # Store the number of components in the blend
                if self.config.version == "lite":
                    nComponents = len(blend.components)
                else:
                    nComponents = 0
                nChild = len(blend.sources)
            # Catch all errors and filter out the ones that we know about
            except Exception as e:
                blendError = type(e).__name__
                if isinstance(e, ScarletGradientError):
                    parent.set(self.iterKey, e.iterations)
                elif not isinstance(e, IncompleteDataError):
                    blendError = "UnknownError"
                    if self.config.catchFailures:
                        # Make it easy to find UnknownErrors in the log file
                        self.log.warn("UnknownError")
                        import traceback
                        traceback.print_exc()
                    else:
                        raise

                self._skipParent(
                    parent=parent,
                    skipKey=self.deblendFailedKey,
                    logMessage=f"Unable to deblend source {parent.getId}: {blendError}",
                )
                parent.set(self.deblendErrorKey, blendError)
                skippedParents.append(parentIndex)
                continue

            # Update the parent record with the deblending results
            if self.config.version == "scarlet":
                logL = -blend.loss[-1] + blend.observations[0].log_norm
            elif self.config.version == "lite":
                logL = blend.loss[-1]
                setDeblenderMetrics(blend)
            self._updateParentRecord(
                parent=parent,
                nPeaks=len(peaks),
                nChild=nChild,
                nComponents=nComponents,
                runtime=runtime,
                iterations=len(blend.loss),
                logL=logL,
                spectrumInit=spectrumInit,
                converged=converged,
            )

            # Add each deblended source to the catalog
            for k, scarletSource in enumerate(blend.sources):
                # Skip any sources with no flux or that scarlet skipped because
                # it could not initialize
                if k in skipped or (self.config.version == "lite" and scarletSource.is_null):
                    # No need to propagate anything
                    continue
                parent.set(self.deblendSkippedKey, False)
                if self.config.version == "lite":
                    mHeavy = liteModelToHeavy(scarletSource, mExposure, blend, xy0=bbox.getMin())
                    weightedHeavy = liteModelToHeavy(
                        scarletSource, mExposure, blend, xy0=bbox.getMin(), useFlux=True)
                    weightedColumns["heavies"].append(weightedHeavy)
                    flux = scarletSource.get_model(use_flux=True).sum(axis=(1, 2))
                    weightedColumns["fluxes"].append({
                        filters[fidx]: _flux
                        for fidx, _flux in enumerate(flux)
                    })
                    centerFlux = self._getCenterFlux(weightedHeavy, scarletSource, xy0=bbox.getMin())
                    weightedColumns["centerFluxes"].append(centerFlux)
                    multibandColumns["metrics"].append(scarletSource.metrics)
                else:
                    mHeavy = modelToHeavy(scarletSource, mExposure, blend, xy0=bbox.getMin())
                multibandColumns["heavies"].append(mHeavy)
                flux = scarlet.measure.flux(scarletSource)
                multibandColumns["fluxes"].append({
                    filters[fidx]: _flux
                    for fidx, _flux in enumerate(flux)
                })
                centerFlux = self._getCenterFlux(mHeavy, scarletSource, xy0=bbox.getMin())
                multibandColumns["centerFluxes"].append(centerFlux)

                # Add all fields except the HeavyFootprint to the
                # source record
                self._addChild(
                    parent=parent,
                    mHeavy=mHeavy,
                    catalog=catalog,
                    scarletSource=scarletSource,
                )

            # Log a message if it has been a while since the last log.
            periodicLog.log("Deblended %d parent sources out of %d", parentIndex + 1, nParents)

            # Clear the cached values in scarlet to clear out memory
            scarlet.cache.Cache._cache = {}

        # Make sure that the number of new sources matches the number of
        # entries in each of the band dependent columns.
        # This should never trigger and is just a sanity check.
        nChildren = len(catalog) - nParents
        if np.any([len(meas) != nChildren for meas in multibandColumns.values()]):
            msg = f"Added {len(catalog)-nParents} new sources, but have "
            msg += ", ".join([
                f"{len(value)} {key}"
                for key, value in multibandColumns.items()
            ])
            raise RuntimeError(msg)
        # Make a copy of the catlog in each band and update the footprints
        catalogs = {}
        for fidx, f in enumerate(filters):
            _catalog = afwTable.SourceCatalog(catalog.table.clone())
            _catalog.extend(catalog, deep=True)

            # Update the footprints and columns that are different
            # for each filter
            for sourceIndex, source in enumerate(_catalog[nParents:]):
                source.setFootprint(multibandColumns["heavies"][sourceIndex][f])
                source.set(self.scarletFluxKey, multibandColumns["fluxes"][sourceIndex][f])
                source.set(self.modelCenterFlux, multibandColumns["centerFluxes"][sourceIndex][f])
                if self.config.version == "lite":
                    self._setMetrics(source, multibandColumns["metrics"][sourceIndex], fidx)
            catalogs[f] = _catalog

        weightedCatalogs = {}
        if self.config.version == "lite":
            # Also create a catalog by reweighting the flux
            weightedCatalogs = {}
            for fidx, f in enumerate(filters):
                _catalog = afwTable.SourceCatalog(catalog.table.clone())
                _catalog.extend(catalog, deep=True)

                # Update the footprints and columns that are different
                # for each filter
                for sourceIndex, source in enumerate(_catalog[nParents:]):
                    source.setFootprint(weightedColumns["heavies"][sourceIndex][f])
                    source.set(self.scarletFluxKey, weightedColumns["fluxes"][sourceIndex][f])
                    source.set(self.modelCenterFlux, weightedColumns["centerFluxes"][sourceIndex][f])
                    self._setMetrics(source, multibandColumns["metrics"][sourceIndex], fidx)
                weightedCatalogs[f] = _catalog

        # Update the mExposure mask with the footprint of skipped parents
        if self.config.notDeblendedMask:
            for mask in mExposure.mask:
                for parentIndex in skippedParents:
                    fp = _catalog[parentIndex].getFootprint()
                    fp.spans.setMask(mask, mask.getPlaneBitMask(self.config.notDeblendedMask))

        self.log.info("Deblender results: of %d parent sources, %d were deblended, "
                      "creating %d children, for a total of %d sources",
                      nParents, nDeblendedParents, nChildren, len(catalog))
        return catalogs, weightedCatalogs

    def _setMetrics(self, source, metrics, filterIndex):
        """Set the metrics for a source in either a template or weighted
        catalog.

        Parameters
        ----------
        source : `lsst.afw.SourceRecord`
            The source that is being updated.
        metrics : `DeblenderMetric`
            The deblender metrics calculated in the deconvolved seeing.
        filterIndex : `int`
            The index of the filter to select a given element of the metric.
        """
        source.set(self.maxOverlapKey, metrics.maxOverlap[filterIndex])
        source.set(self.fluxOverlapKey, metrics.fluxOverlap[filterIndex])
        source.set(self.fluxOverlapFractionKey, metrics.fluxOverlapFraction[filterIndex])
        source.set(self.blendednessKey, metrics.blendedness[filterIndex])

    def _isLargeFootprint(self, footprint):
        """Returns whether a Footprint is large

        'Large' is defined by thresholds on the area, size and axis ratio,
        and total area of the bounding box multiplied by
        the number of children.
        These may be disabled independently by configuring them to be
        non-positive.
        """
        if self.config.maxFootprintArea > 0 and footprint.getArea() > self.config.maxFootprintArea:
            return True
        if self.config.maxFootprintSize > 0:
            bbox = footprint.getBBox()
            if max(bbox.getWidth(), bbox.getHeight()) > self.config.maxFootprintSize:
                return True
        if self.config.minFootprintAxisRatio > 0:
            axes = afwEll.Axes(footprint.getShape())
            if axes.getB() < self.config.minFootprintAxisRatio*axes.getA():
                return True
        if self.config.maxAreaTimesPeaks > 0:
            if footprint.getBBox().getArea() * len(footprint.peaks) > self.config.maxAreaTimesPeaks:
                return True
        return False

    def _isMasked(self, footprint, mExposure):
        """Returns whether the footprint violates the mask limits

        Parameters
        ----------
        footprint : `lsst.afw.detection.Footprint`
            The footprint to check for masked pixels
        mMask : `lsst.afw.image.MaskX`
            The mask plane to check for masked pixels in the `footprint`.

        Returns
        -------
        isMasked : `bool`
            `True` if `self.config.maskPlaneLimits` is less than the
            fraction of pixels for a given mask in
            `self.config.maskLimits`.
        """
        bbox = footprint.getBBox()
        mask = np.bitwise_or.reduce(mExposure.mask[:, bbox].array, axis=0)
        size = float(footprint.getArea())
        for maskName, limit in self.config.maskLimits.items():
            maskVal = mExposure.mask.getPlaneBitMask(maskName)
            _mask = afwImage.MaskX(mask & maskVal, xy0=bbox.getMin())
            # spanset of masked pixels
            maskedSpan = footprint.spans.intersect(_mask, maskVal)
            if (maskedSpan.getArea())/size > limit:
                return True
        return False

    def _skipParent(self, parent, skipKey, logMessage):
        """Update a parent record that is not being deblended.

        This is a fairly trivial function but is implemented to ensure
        that a skipped parent updates the appropriate columns
        consistently, and always has a flag to mark the reason that
        it is being skipped.

        Parameters
        ----------
        parent : `lsst.afw.table.source.source.SourceRecord`
            The parent record to flag as skipped.
        skipKey : `bool`
            The name of the flag to mark the reason for skipping.
        logMessage : `str`
            The message to display in a log.trace when a source
            is skipped.
        """
        if logMessage is not None:
            self.log.trace(logMessage)
        self._updateParentRecord(
            parent=parent,
            nPeaks=len(parent.getFootprint().peaks),
            nChild=0,
            nComponents=0,
            runtime=np.nan,
            iterations=0,
            logL=np.nan,
            spectrumInit=False,
            converged=False,
        )

        # Mark the source as skipped by the deblender and
        # flag the reason why.
        parent.set(self.deblendSkippedKey, True)
        parent.set(skipKey, True)

    def _checkSkipped(self, parent, mExposure):
        """Update a parent record that is not being deblended.

        This is a fairly trivial function but is implemented to ensure
        that a skipped parent updates the appropriate columns
        consistently, and always has a flag to mark the reason that
        it is being skipped.

        Parameters
        ----------
        parent : `lsst.afw.table.source.source.SourceRecord`
            The parent record to flag as skipped.
        mExposure : `MultibandExposure`
            The exposures should be co-added images of the same
            shape and region of the sky.
        Returns
        -------
        skip: `bool`
            `True` if the deblender will skip the parent
        """
        skipKey = None
        skipMessage = None
        footprint = parent.getFootprint()
        if len(footprint.peaks) < 2 and not self.config.processSingles:
            # Skip isolated sources unless processSingles is turned on.
            # Note: this does not flag isolated sources as skipped or
            # set the NOT_DEBLENDED mask in the exposure,
            # since these aren't really any skipped blends.
            skipKey = self.isolatedParentKey
        elif isPseudoSource(parent, self.config.pseudoColumns):
            # We also skip pseudo sources, like sky objects, which
            # are intended to be skipped.
            skipKey = self.pseudoKey
        if self._isLargeFootprint(footprint):
            # The footprint is above the maximum footprint size limit
            skipKey = self.tooBigKey
            skipMessage = f"Parent {parent.getId()}: skipping large footprint"
        elif self._isMasked(footprint, mExposure):
            # The footprint exceeds the maximum number of masked pixels
            skipKey = self.maskedKey
            skipMessage = f"Parent {parent.getId()}: skipping masked footprint"
        elif self.config.maxNumberOfPeaks > 0 and len(footprint.peaks) > self.config.maxNumberOfPeaks:
            # Unlike meas_deblender, in scarlet we skip the entire blend
            # if the number of peaks exceeds max peaks, since neglecting
            # to model any peaks often results in catastrophic failure
            # of scarlet to generate models for the brighter sources.
            skipKey = self.tooManyPeaksKey
            skipMessage = f"Parent {parent.getId()}: skipping blend with too many peaks"
        if skipKey is not None:
            return (skipKey, skipMessage)
        return None

    def setSkipFlags(self, mExposure, catalog):
        """Set the skip flags for all of the parent sources

        This is mostly used for testing which parent sources will be deblended
        and which will be skipped based on the current configuration options.
        Skipped sources will have the appropriate flags set in place in the
        catalog.

        Parameters
        ----------
        mExposure : `MultibandExposure`
            The exposures should be co-added images of the same
            shape and region of the sky.
        catalog : `SourceCatalog`
            The merged `SourceCatalog` that contains parent footprints
            to (potentially) deblend. The new deblended sources are
            appended to this catalog in place.
        """
        for src in catalog:
            if skipArgs := self._checkSkipped(src, mExposure) is not None:
                self._skipParent(src, *skipArgs)

    def _updateParentRecord(self, parent, nPeaks, nChild, nComponents,
                            runtime, iterations, logL, spectrumInit, converged):
        """Update a parent record in all of the single band catalogs.

        Ensure that all locations that update a parent record,
        whether it is skipped or updated after deblending,
        update all of the appropriate columns.

        Parameters
        ----------
        parent : `lsst.afw.table.source.source.SourceRecord`
            The parent record to update.
        nPeaks : `int`
            Number of peaks in the parent footprint.
        nChild : `int`
            Number of children deblended from the parent.
            This may differ from `nPeaks` if some of the peaks
            were culled and have no deblended model.
        nComponents : `int`
            Total number of components in the parent.
            This is usually different than the number of children,
            since it is common for a single source to have multiple
            components.
        runtime : `float`
            Total runtime for deblending.
        iterations : `int`
            Total number of iterations in scarlet before convergence.
        logL : `float`
            Final log likelihood of the blend.
        spectrumInit : `bool`
            True when scarlet used `set_spectra` to initialize all
            sources with better initial intensities.
        converged : `bool`
            True when the optimizer reached convergence before
            reaching the maximum number of iterations.
        """
        parent.set(self.nPeaksKey, nPeaks)
        parent.set(self.nChildKey, nChild)
        parent.set(self.nComponentsKey, nComponents)
        parent.set(self.runtimeKey, runtime)
        parent.set(self.iterKey, iterations)
        parent.set(self.scarletLogLKey, logL)
        parent.set(self.scarletSpectrumInitKey, spectrumInit)
        parent.set(self.blendConvergenceFailedFlagKey, converged)

    def _addChild(self, parent, mHeavy, catalog, scarletSource):
        """Add a child to a catalog.

        This creates a new child in the source catalog,
        assigning it a parent id, and adding all columns
        that are independent across all filter bands.

        Parameters
        ----------
        parent : `lsst.afw.table.source.source.SourceRecord`
            The parent of the new child record.
        mHeavy : `lsst.detection.MultibandFootprint`
            The multi-band footprint containing the model and
            peak catalog for the new child record.
        catalog : `lsst.afw.table.source.source.SourceCatalog`
            The merged `SourceCatalog` that contains parent footprints
            to (potentially) deblend.
        scarletSource : `scarlet.Component`
            The scarlet model for the new source record.
        """
        src = catalog.addNew()
        for key in self.toCopyFromParent:
            src.set(key, parent.get(key))
        # The peak catalog is the same for all bands,
        # so we just use the first peak catalog
        peaks = mHeavy[mHeavy.filters[0]].peaks
        src.assign(peaks[0], self.peakSchemaMapper)
        src.setParent(parent.getId())
        src.set(self.nPeaksKey, len(peaks))
        # Set the psf key based on whether or not the source was
        # deblended using the PointSource model.
        # This key is not that useful anymore since we now keep track of
        # `modelType`, but we continue to propagate it in case code downstream
        # is expecting it.
        src.set(self.psfKey, scarletSource.__class__.__name__ == "PointSource")
        src.set(self.modelTypeKey, scarletSource.__class__.__name__)
        # We set the runtime to zero so that summing up the
        # runtime column will give the total time spent
        # running the deblender for the catalog.
        src.set(self.runtimeKey, 0)

        # Set the position of the peak from the parent footprint
        # This will make it easier to match the same source across
        # deblenders and across observations, where the peak
        # position is unlikely to change unless enough time passes
        # for a source to move on the sky.
        peak = peaks[0]
        src.set(self.peakCenter, Point2I(peak["i_x"], peak["i_y"]))
        src.set(self.peakIdKey, peak["id"])

        # Store the number of components for the source
        src.set(self.nComponentsKey, len(scarletSource.components))

        # Propagate columns from the parent to the child
        for parentColumn, childColumn in self.config.columnInheritance.items():
            src.set(childColumn, parent.get(parentColumn))

    def _getCenterFlux(self, mHeavy, scarletSource, xy0):
        """Get the flux at the center of a HeavyFootprint

        Parameters
        ----------
        mHeavy : `lsst.detection.MultibandFootprint`
            The multi-band footprint containing the model for the source.
        scarletSource : `scarlet.Component`
            The scarlet model for the heavy footprint
        """
        # Store the flux at the center of the model and the total
        # scarlet flux measurement.
        mImage = mHeavy.getImage(fill=0.0).image

        # Set the flux at the center of the model (for SNR)
        try:
            cy, cx = scarletSource.center
            cy += xy0.y
            cx += xy0.x
            return mImage[:, cx, cy]
        except AttributeError:
            msg = "Did not recognize coordinates for source type of `{0}`, "
            msg += "could not write coordinates or center flux. "
            msg += "Add `{0}` to meas_extensions_scarlet to properly persist this information."
            logger.warning(msg.format(type(scarletSource)))
        return {f: np.nan for f in mImage.filters}
