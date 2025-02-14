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

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
import time
from typing import Any

import lsst.afw.detection as afwDet
import lsst.afw.geom.ellipses as afwEll
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.scarlet.lite as scl
import numpy as np
from lsst.scarlet.lite.detect_pybind11 import get_footprints
from scipy import ndimage
from lsst.utils.logging import PeriodicLogger
from lsst.utils.timer import timeMethod

from . import utils
from .footprint import footprintsToNumpy
from .footprint import scarletFootprintToAfw, getFootprintIntersection

# Scarlet and proxmin have a different definition of log levels than the stack,
# so even "warnings" occur far more often than we would like.
# So for now we only display scarlet and proxmin errors, as all other
# scarlet outputs would be considered "TRACE" by our standards.
scarletLogger = logging.getLogger("scarlet")
scarletLogger.setLevel(logging.ERROR)
proxminLogger = logging.getLogger("proxmin")
proxminLogger.setLevel(logging.ERROR)

__all__ = ["deblend", "ScarletDeblendContext", "ScarletDeblendConfig", "ScarletDeblendTask"]

logger = logging.getLogger(__name__)


def _checkBlendConvergence(blend: scl.Blend, f_rel: float) -> bool:
    """Check whether or not a blend has converged"""
    deltaLoss = np.abs(blend.loss[-2] - blend.loss[-1])
    convergence = f_rel * np.abs(blend.loss[-1])
    return deltaLoss < convergence


def isPseudoSource(source: afwTable.source.SourceRecord, pseudoColumns: list[str]) -> bool:
    """Check if a source is a pseudo source.

    This is mostly for skipping sky objects,
    but any other column can also be added to disable
    deblending on a parent or individual source when
    set to `True`.

    Parameters
    ----------
    source :
        The source to check for the pseudo bit.
    pseudoColumns :
        A list of columns to check for pseudo sources.
    """
    isPseudo = False
    for col in pseudoColumns:
        try:
            isPseudo |= source[col]
        except KeyError:
            pass
    return isPseudo


def _getDeconvolvedFootprints(
    mDeconvolved: afwImage.MultibandExposure,
    sources: afwTable.SourceCatalog,
    config: ScarletDeblendConfig,
) -> tuple[list[scl.detect.Footprint], scl.Image]:
    """Detect footprints in the deconvolved image

    Parameters
    ----------
    mDeconvolved :
        The deconvolved multiband exposure to detect footprints in.
    modelPsf :
        The PSF of the model.

    Returns
    -------
    footprints :
        The detected footprints in the deconvolved image.
    footprintImage :
        A footprint image as returned by `scarlet.detect.footprints_to_image`.
    """
    bbox = mDeconvolved.getBBox()
    xmin, ymin = bbox.getMin()
    sigma = np.median(np.sqrt(mDeconvolved.variance.array), axis=(1, 2))
    detect = np.sum(mDeconvolved.image.array/sigma[:, None, None], axis=0)

    # We don't use the variance here because testing in DM-47738
    # has shown that we get better results without it
    # (we're detecting footprints without peaks in the deconvolved,
    # noise reduced image).
    footprints = scl.detect.detect_footprints(
        images=np.array([detect]),
        variance=np.ones((1, detect.shape[0], detect.shape[1]), dtype=detect.dtype),
        scales=1,
        min_area=config.minDeconvolvedArea,
        footprint_thresh=config.footprintSNRThresh,
        find_peaks=False,
        origin=(ymin, xmin)
    )

    # Create an indexed image of the footprints so that the value of a pixel
    # gives the index + 1 of the footprints that contain that pixel.
    scarletBox = utils.bboxToScarletBox(bbox)
    footprintImage = scl.detect.footprints_to_image(footprints, scarletBox)

    # Ensure that there is a minimal footprint for all peaks
    # in the detection catalog.
    footprintArray = footprintImage.data > 0
    for source in sources:
        for peak in source.getFootprint().peaks:
            x, y = peak.getI()
            footprintArray[y - ymin, x - xmin] = True

    # Grow the footprints
    psfMinimalSize = config.growSize * 2 + 1
    detectionArray = ndimage.binary_dilation(
        footprintArray,
        scl.utils.get_circle_mask(psfMinimalSize, bool)
    )

    # Ensure that all of the deconvolved footprints are contained within
    # a single footprint from the input catalog.
    # This should be unecessary, however creating entries in the output
    # catalog will produce unexpected results if a deconvolved footprint
    # is in more than one footprint from the source catalog or has
    # flux outside of its parent footprint.
    sourceImage = footprintsToNumpy(sources, detect.shape, (xmin, ymin))
    detectionArray = detectionArray * sourceImage

    footprints = get_footprints(
        image=detectionArray.astype(np.float32),
        min_separation=0,
        min_area=1,
        peak_thresh=0,
        footprint_thresh=0,
        find_peaks=False,
        y0=ymin,
        x0=xmin,
    )

    footprintImage = scl.detect.footprints_to_image(footprints, scarletBox)

    return footprints, footprintImage


@dataclass(kw_only=True)
class ScarletDeblendContext:
    """Context with parameters and config options for deblending

    Attributes
    ----------
    monotonicity :
        The monotonicity operator.
    observation :
        The observation for the entire coadd.
    deconvolved :
        The deconvolved image.
    residual :
        The residual image
        (observation - deconvolved mode convolved with difference kernel).
    footprints :
        The footprints in the deconvolved image.
    footprintImage :
        An indexed image of the scarlet footprints so that the value
        of a pixel gives the index + 1 of the footprints that
        contain that pixel.
    config :
        The configuration for the deblender.
    """
    monotonicity: scl.operators.Monotonicity
    observation: scl.Observation
    deconvolved: scl.Image
    footprints: list[scl.Footprint]
    footprintImage: scl.Image
    config: ScarletDeblendConfig

    @staticmethod
    def build(
        mExposure: afwImage.MultibandExposure,
        mDeconvolved: afwImage.MultibandExposure,
        catalog: afwTable.SourceCatalog,
        config: ScarletDeblendConfig,
    ) -> ScarletDeblendContext:
        """Build the context from a minimal set of inputs

        Parameters
        ----------
        mExposure :
            The multiband exposure for the entire coadd.
        mDeconvolved :
            The deconvolved multiband exposure for the entire coadd.
        catalog :
            The source catalog for the entire coadd.
        config :
            The configuration for the deblender.
        """
        # The PSF of the model in the deconvolved space
        modelPsf = scl.utils.integrated_circular_gaussian(
            sigma=config.modelPsfSigma
        )
        # Initialize the monotonicity operator with a size of 101 x 101 pixels.
        # Note: If a component is > 101x101 in either axis then the
        # monotonicity operator will resize itself.
        monotonicity = scl.operators.Monotonicity((101, 101))
        # Build the observation for the entire coadd
        observation = utils.buildObservation(
            modelPsf=modelPsf,
            psfCenter=mExposure.getBBox().getCenter(),
            mExposure=mExposure,
            badPixelMasks=config.badMask,
            useWeights=config.useWeights,
            convolutionType=config.convolutionType,
        )

        # Create the deconvolved image
        yx0 = observation.images.yx0
        deconvolved = scl.Image(mDeconvolved.image.array, bands=observation.images.bands, yx0=yx0)

        # Detect footprints in the deconvolved image
        footprints, footprintImage = _getDeconvolvedFootprints(
            mDeconvolved=mDeconvolved,
            sources=catalog,
            config=config,
        )

        return ScarletDeblendContext(
            monotonicity=monotonicity,
            observation=observation,
            deconvolved=deconvolved,
            footprints=footprints,
            footprintImage=footprintImage,
            config=config,
        )


def deblend(
    context: ScarletDeblendContext,
    footprint: afwDet.Footprint,
    config: ScarletDeblendConfig,
    spectrumInit: bool = True,
) -> scl.Blend:
    """Deblend a parent footprint

    Parameters
    ----------
    context :
        Context with parameters and config options for deblending
    footprint :
        The parent footprint to deblend
    config :
        The configuration for the deblender
    spectrumInit :
        Whether or not to initialize the sources with their best-fit spectra

    Returns
    -------
    blend : `scarlet.lite.Blend`
        The blend this is to be deblended
    skippedSources : `list[int]`
        Indices of sources that were skipped due to no flux.
        This usually means that a source was a spurrious detection in one
        band that should not have been included in the merged catalog.
    skippedBands : `list[str]`
        Bands that were skipped because a PSF could not be generated for them.
    """
    # Define the bounding boxes in lsst.geom.Box2I and lsst.scarlet.lite.Box
    footBox = footprint.getBBox()
    bbox = utils.bboxToScarletBox(footBox)

    # Extract the observation that covers the footprint and make
    # a copy so that the changes don't affect the original observation.
    observation = context.observation[:, bbox].copy()
    footprintData = footprint.spans.asArray()

    # Mask the pixels outside of the footprint
    observation.weights.data[:] *= footprintData

    # Convert the peaks into an array
    peaks = [
        np.array([peak.getIy(), peak.getIx()], dtype=int)
        for peak in footprint.peaks
        if not isPseudoSource(peak, config.pseudoColumns)
    ]

    # Initialize the sources
    sources = scl.initialization.FactorizedInitialization(
        observation=observation,
        centers=peaks,
        min_snr=config.minSNR,
        monotonicity=context.monotonicity,
        bg_thresh=config.backgroundThresh,
        initial_bg_thresh=config.initialBackgroundThresh,
    ).sources

    blend = scl.Blend(sources, observation)

    # Initialize each source with its best fit spectrum
    if spectrumInit:
        blend.fit_spectra()

    # Set the optimizer
    if config.optimizer == "adaprox":
        blend.parameterize(
            partial(
                scl.component.default_adaprox_parameterization,
                noise_rms=observation.noise_rms / 10,
            )
        )
    elif config.optimizer == "fista":
        blend.parameterize(scl.component.default_fista_parameterization)
    else:
        raise ValueError("Unrecognized optimizer. Must be either 'adaprox' or 'fista'.")

    if config.maxIter > 0:
        blend.fit(
            max_iter=config.maxIter,
            e_rel=config.relativeError,
            min_iter=config.minIter,
            resize=config.resizeFrequency,
        )
    else:
        loss = (blend.observation.images - blend.get_model(convolve=True)).data
        blend.loss = [np.sum(loss), np.sum(loss)]

    # Attach the peak to all of the initialized sources
    for k, center in enumerate(peaks):
        # This is just to make sure that there isn't a coding bug
        if len(sources[k].components) > 0 and np.any(sources[k].center != center):
            raise ValueError(
                f"Misaligned center, expected {center} but got {sources[k].center}"
            )
        # Store the record for the peak with the appropriate source
        sources[k].detectedPeak = footprint.peaks[k]

    return blend


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
    minIter = pexConfig.Field[int](
        default=5,
        doc="Minimum number of iterations before the optimizer is allowed to stop.",
    )
    maxIter = pexConfig.Field[int](
        default=300,
        doc=("Maximum number of iterations to deblend a single parent"),
    )
    relativeError = pexConfig.Field[float](
        default=1e-3,
        doc=(
            "Change in the loss function between iterations to exit fitter. "
            "Typically this is `1e-3` if measurements will be made on the "
            "flux re-distributed models and `1e-4` when making measurements "
            "on the models themselves."
        ),
    )
    resizeFrequency = pexConfig.Field[int](
        default=3,
        doc="Number of iterations between resizing sources.",
    )

    # Constraints
    morphThresh = pexConfig.Field[float](
        default=1,
        doc="Fraction of background RMS a pixel must have"
        "to be included in the initial morphology",
    )
    # Lite Parameters
    # All of these parameters (except version) are only valid if version='lite'
    version = pexConfig.ChoiceField[str](
        default="lite",
        allowed={
            "lite": "LSST optimized version of scarlet for survey data from a single instrument",
        },
        doc="The version of scarlet to use.",
        deprecated="This field is deprecated since the ony available `version` is `lite` "
                   "and will be removed after v29.0",
    )
    optimizer = pexConfig.ChoiceField[str](
        default="adaprox",
        allowed={
            "adaprox": "Proximal ADAM optimization",
            "fista": "Accelerated proximal gradient method",
        },
        doc="The optimizer to use for fitting parameters.",
    )
    morphImage = pexConfig.ChoiceField[str](
        default="chi2",
        allowed={
            "chi2": "Initialize sources on a chi^2 image made from all available bands",
        },
        doc="The type of image to use for initializing the morphology. "
            "Must be either 'chi2' or 'wavelet'. ",
        deprecated="This field is deprecated since testing has shown that only 'chi2' should be used "
                   "and 'wavelet' has been broken since v27.0. "
                   "This field will be removed in v29.0",
    )
    backgroundThresh = pexConfig.Field[float](
        default=1.0,
        doc="Fraction of background to use for a sparsity threshold. "
        "This prevents sources from growing unrealistically outside "
        "the parent footprint while still modeling flux correctly "
        "for bright sources.",
    )
    initialBackgroundThresh = pexConfig.Field[float](
        default=1.0,
        doc="Same as `backgroundThresh` but used only for source initialization.",
    )
    maxProxIter = pexConfig.Field[int](
        default=1,
        doc="Maximum number of proximal operator iterations inside of each "
        "iteration of the optimizer. "
        "This config field is only used if version='lite' and optimizer='adaprox'.",
    )
    waveletScales = pexConfig.Field[int](
        default=5,
        doc="Number of wavelet scales to use for wavelet initialization. "
        "This field is only used when `version`='lite' and `morphImage`='wavelet'.",
        deprecated="This field is deprecated along with `morphImage` and will be removed in v29.0.",
    )

    # Other scarlet paremeters
    useWeights = pexConfig.Field[bool](
        default=True,
        doc=(
            "Whether or not use use inverse variance weighting."
            "If `useWeights` is `False` then flat weights are used"
        ),
    )
    modelPsfSize = pexConfig.Field[int](
        default=11, doc="Model PSF side length in pixels"
    )
    modelPsfSigma = pexConfig.Field[float](
        default=0.8, doc="Define sigma for the model frame PSF"
    )
    minSNR = pexConfig.Field[float](
        default=50,
        doc="Minimum Signal to noise to accept the source."
        "Sources with lower flux will be initialized with the PSF but updated "
        "like an ordinary ExtendedSource (known in scarlet as a `CompactSource`).",
    )
    saveTemplates = pexConfig.Field[bool](
        default=True, doc="Whether or not to save the SEDs and templates"
    )
    processSingles = pexConfig.Field[bool](
        default=True,
        doc="Whether or not to process isolated sources in the deblender",
    )
    convolutionType = pexConfig.Field[str](
        default="fft",
        doc="Type of convolution to render the model to the observations.\n"
        "- 'fft': perform convolutions in Fourier space\n"
        "- 'real': peform convolutions in real space.",
    )
    sourceModel = pexConfig.Field[str](
        default="double",
        doc=(
            "How to determine which model to use for sources, from\n"
            "- 'single': use a single component for all sources\n"
            "- 'double': use a bulge disk model for all sources\n"
            "- 'compact': use a single component model, initialzed with a point source morphology, "
            " for all sources\n"
            "- 'point': use a point-source model for all sources\n"
            "- 'fit: use a PSF fitting model to determine the number of components (not yet implemented)"
        ),
        deprecated="This field will be deprecated when the default for `version` is changed to `lite`.",
    )
    setSpectra = pexConfig.Field[bool](
        default=True,
        doc="Whether or not to solve for the best-fit spectra during initialization. "
        "This makes initialization slightly longer, as it requires a convolution "
        "to set the optimal spectra, but results in a much better initial log-likelihood "
        "and reduced total runtime, with convergence in fewer iterations."
        "This option is only used when "
        "peaks*area < `maxSpectrumCutoff` will use the improved initialization.",
    )
    footprintSNRThresh = pexConfig.Field[float](
        default=5.0,
        doc="Minimum SNR for a pixel to be detected in a footprint.",
    )
    growSize = pexConfig.Field[int](
        default=2,
        doc="Number of pixels to grow the deconvolved footprints before final detection.",
    )

    # Mask-plane restrictions
    badMask = pexConfig.ListField[str](
        default=utils.defaultBadPixelMasks,
        doc="Whether or not to process isolated sources in the deblender",
    )
    statsMask = pexConfig.ListField[str](
        default=["SAT", "INTRP", "NO_DATA"],
        doc="Mask planes to ignore when performing statistics",
    )
    maskLimits = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        default={},
        doc=(
            "Mask planes with the corresponding limit on the fraction of masked pixels. "
            "Sources violating this limit will not be deblended. "
            "If the fraction is `0` then the limit is a single pixel."
        ),
    )
    minDeconvolvedArea = pexConfig.Field[int](
        default=9,
        doc="Minimum area for a single footprint in the deconvolved image. "
            "Detected footprints smaller than this will not be created.",
    )

    # Size restrictions
    maxNumberOfPeaks = pexConfig.Field[int](
        default=-1,
        doc=(
            "Only deblend the brightest maxNumberOfPeaks peaks in the parent"
            " (<= 0: unlimited)"
        ),
    )
    maxFootprintArea = pexConfig.Field[int](
        default=-1,
        doc=(
            "Maximum area for footprints before they are ignored as large; "
            "non-positive means no threshold applied"
        ),
    )
    maxAreaTimesPeaks = pexConfig.Field[int](
        default=-1,
        doc=(
            "Maximum rectangular footprint area * nPeaks in the footprint. "
            "This was introduced in DM-33690 to prevent fields that are crowded or have a "
            "LSB galaxy that causes memory intensive initialization in scarlet from dominating "
            "the overall runtime and/or causing the task to run out of memory. "
            "(<= 0: unlimited)"
        ),
    )
    maxFootprintSize = pexConfig.Field[int](
        default=0,
        doc=(
            "Maximum linear dimension for footprints before they are ignored "
            "as large; non-positive means no threshold applied"
        ),
    )
    minFootprintAxisRatio = pexConfig.Field[float](
        default=0.0,
        doc=(
            "Minimum axis ratio for footprints before they are ignored "
            "as large; non-positive means no threshold applied"
        ),
    )
    maxSpectrumCutoff = pexConfig.Field[int](
        default=1_000_000,
        doc=(
            "Maximum number of pixels * number of sources in a blend. "
            "This is different than `maxFootprintArea` because this isn't "
            "the footprint area but the area of the bounding box that "
            "contains the footprint, and is also multiplied by the number of"
            "sources in the footprint. This prevents large skinny blends with "
            "a high density of sources from running out of memory. "
            "If `maxSpectrumCutoff == -1` then there is no cutoff."
        ),
    )
    # Failure modes
    fallback = pexConfig.Field[bool](
        default=True,
        doc="Whether or not to fallback to a smaller number of components if a source does not initialize",
    )
    notDeblendedMask = pexConfig.Field[str](
        default="NOT_DEBLENDED",
        optional=True,
        doc="Mask name for footprints not deblended, or None",
    )
    catchFailures = pexConfig.Field[bool](
        default=True,
        doc=(
            "If True, catch exceptions thrown by the deblender, log them, "
            "and set a flag on the parent, instead of letting them propagate up"
        ),
    )

    # Other options
    columnInheritance = pexConfig.DictField(
        keytype=str,
        itemtype=str,
        default={
            "deblend_nChild": "deblend_parentNChild",
            "deblend_nPeaks": "deblend_parentNPeaks",
            "deblend_spectrumInitFlag": "deblend_spectrumInitFlag",
            "deblend_blendConvergenceFailedFlag": "deblend_blendConvergenceFailedFlag",
        },
        doc="Columns to pass from the parent to the child. "
        "The key is the name of the column for the parent record, "
        "the value is the name of the column to use for the child.",
    )
    pseudoColumns = pexConfig.ListField[str](
        default=["merge_peak_sky", "sky_source"],
        doc="Names of flags which should never be deblended.",
    )

    # Testing options
    # Some obs packages and ci packages run the full pipeline on a small
    # subset of data to test that the pipeline is functioning properly.
    # This is not meant as scientific validation, so it can be useful
    # to only run on a small subset of the data that is large enough to
    # test the desired pipeline features but not so long that the deblender
    # is the tall pole in terms of execution times.
    useCiLimits = pexConfig.Field[bool](
        default=False,
        doc="Limit the number of sources deblended for CI to prevent long build times",
    )
    ciDeblendChildRange = pexConfig.ListField[int](
        default=[5, 10],
        doc="Only deblend parent Footprints with a number of peaks in the (inclusive) range indicated."
        "If `useCiLimits==False` then this parameter is ignored.",
    )
    ciNumParentsToDeblend = pexConfig.Field[int](
        default=10,
        doc="Only use the first `ciNumParentsToDeblend` parent footprints with a total peak count "
        "within `ciDebledChildRange`. "
        "If `useCiLimits==False` then this parameter is ignored.",
    )


class ScarletDeblendTask(pipeBase.Task):
    """ScarletDeblendTask

    Split blended sources into individual sources.

    This task has no return value; it only modifies the SourceCatalog in-place.
    """

    ConfigClass = ScarletDeblendConfig
    _DefaultName = "scarletDeblend"

    def __init__(
        self,
        schema: afwTable.Schema,
        peakSchema: afwTable.Schema = None,
        **kwargs
    ):
        """Create the task, adding necessary fields to the given schema.

        Parameters
        ----------
        schema :
            Schema object for measurement fields; will be modified in-place.
        peakSchema :
            Schema of Footprint Peaks that will be passed to the deblender.
            Any fields beyond the PeakTable minimal schema will be transferred
            to the main source Schema.  If None, no fields will be transferred
            from the Peaks.
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
            assert (
                schema == self.peakSchemaMapper.getOutputSchema()
            ), "Logic bug mapping schemas"
        self._addSchemaKeys(schema)
        self.schema = schema
        self.toCopyFromParent = [
            item.key
            for item in self.schema
            if item.field.getName().startswith("merge_footprint")
        ]

    def _addSchemaKeys(self, schema: afwTable.Schema):
        """Add deblender specific keys to the schema"""
        # Parent (blend) fields
        self.runtimeKey = schema.addField(
            "deblend_runtime", type=np.float32, doc="runtime in ms"
        )
        self.iterKey = schema.addField(
            "deblend_iterations", type=np.int32, doc="iterations to converge"
        )
        self.nChildKey = schema.addField(
            "deblend_nChild",
            type=np.int32,
            doc="Number of children this object has (defaults to 0)",
        )
        self.nPeaksKey = schema.addField(
            "deblend_nPeaks",
            type=np.int32,
            doc="Number of initial peaks in the blend. "
            "This includes peaks that may have been culled "
            "during deblending or failed to deblend",
        )
        # Skipped flags
        self.deblendSkippedKey = schema.addField(
            "deblend_skipped", type="Flag", doc="Deblender skipped this source"
        )
        self.isolatedParentKey = schema.addField(
            "deblend_isolatedParent",
            type="Flag",
            doc="The source has only a single peak " "and was not deblended",
        )
        self.pseudoKey = schema.addField(
            "deblend_isPseudo",
            type="Flag",
            doc='The source is identified as a "pseudo" source and '
            "was not deblended",
        )
        self.tooManyPeaksKey = schema.addField(
            "deblend_tooManyPeaks",
            type="Flag",
            doc="Source had too many peaks; " "only the brightest were included",
        )
        self.tooBigKey = schema.addField(
            "deblend_parentTooBig",
            type="Flag",
            doc="Parent footprint covered too many pixels",
        )
        self.maskedKey = schema.addField(
            "deblend_masked",
            type="Flag",
            doc="Parent footprint had too many masked pixels",
        )
        # Convergence flags
        self.blendConvergenceFailedFlagKey = schema.addField(
            "deblend_blendConvergenceFailedFlag",
            type="Flag",
            doc="at least one source in the blend" "failed to converge",
        )
        # Error flags
        self.deblendFailedKey = schema.addField(
            "deblend_failed", type="Flag", doc="Deblending failed on source"
        )
        self.deblendChildFailedKey = schema.addField(
            "deblend_childFailed",
            type="Flag",
            doc="Deblending failed on at least one child blend. "
                " This is set in a parent when at least one of its children "
                "is a blend that failed to deblend.",
        ),
        self.deblendErrorKey = schema.addField(
            "deblend_error",
            type="String",
            size=25,
            doc="Name of error if the blend failed",
        )
        self.incompleteDataKey = schema.addField(
            "deblend_incompleteData",
            type="Flag",
            doc="True when a blend has at least one band "
            "that could not generate a PSF and was "
            "not included in the model.",
        )
        # Deblended source fields
        self.peakCenter = afwTable.Point2IKey.addFields(
            schema,
            name="deblend_peak_center",
            doc="Center used to apply constraints in scarlet",
            unit="pixel",
        )
        self.peakIdKey = schema.addField(
            "deblend_peakId",
            type=np.int32,
            doc="ID of the peak in the parent footprint. "
            "This is not unique, but the combination of 'parent'"
            "and 'peakId' should be for all child sources. "
            "Top level blends with no parents have 'peakId=0'",
        )
        self.modelCenterFlux = schema.addField(
            "deblend_peak_instFlux",
            type=float,
            units="count",
            doc="The instFlux at the peak position of deblended mode",
        )
        self.modelTypeKey = schema.addField(
            "deblend_modelType",
            type="String",
            size=25,
            doc="The type of model used, for example "
            "MultiExtendedSource, SingleExtendedSource, PointSource",
        )
        self.deblendDepthKey = schema.addField(
            "deblend_depth",
            type=np.int32,
            doc="The depth of deblending in the hierarchy."
        )
        self.parentNPeaksKey = schema.addField(
            "deblend_parentNPeaks",
            type=np.int32,
            doc="deblend_nPeaks from this records parent.",
        )
        self.parentNChildKey = schema.addField(
            "deblend_parentNChild",
            type=np.int32,
            doc="deblend_nChild from this records parent.",
        )
        self.scarletFluxKey = schema.addField(
            "deblend_scarletFlux", type=np.float32, doc="Flux measurement from scarlet"
        )
        self.scarletLogLKey = schema.addField(
            "deblend_logL",
            type=np.float32,
            doc="Final logL, used to identify regressions in scarlet.",
        )
        self.scarletChi2Key = schema.addField(
            "deblend_chi2",
            type=np.float32,
            doc="Final reduced chi2 (per pixel), used to identify goodness of fit.",
        )
        self.edgePixelsKey = schema.addField(
            "deblend_edgePixels",
            type="Flag",
            doc="Source had flux on the edge of the parent footprint",
        )
        self.scarletSpectrumInitKey = schema.addField(
            "deblend_spectrumInitFlag",
            type="Flag",
            doc="True when scarlet initializes sources "
            "in the blend with a more accurate spectrum. "
            "The algorithm uses a lot of memory, "
            "so large dense blends will use "
            "a less accurate initialization.",
        )
        self.nComponentsKey = schema.addField(
            "deblend_nComponents",
            type=np.int32,
            doc="Number of components in a ScarletLiteSource. "
            "If `config.version != 'lite'`then "
            "this column is set to zero.",
        )
        self.psfKey = schema.addField(
            "deblend_deblendedAsPsf",
            type="Flag",
            doc="Deblender thought this source looked like a PSF",
        )
        self.coverageKey = schema.addField(
            "deblend_dataCoverage",
            type=np.float32,
            doc="Fraction of pixels with data. "
            "In other words, 1 - fraction of pixels with NO_DATA set.",
        )
        self.zeroFluxKey = schema.addField(
            "deblend_zeroFlux", type="Flag", doc="Source has zero flux."
        )
        # Blendedness/classification metrics
        self.maxOverlapKey = schema.addField(
            "deblend_maxOverlap",
            type=np.float32,
            doc="Maximum overlap with all of the other neighbors flux "
            "combined."
            "This is useful as a metric for determining how blended a "
            "source is because if it only overlaps with other sources "
            "at or below the noise level, it is likely to be a mostly "
            "isolated source in the deconvolved model frame.",
        )
        self.fluxOverlapKey = schema.addField(
            "deblend_fluxOverlap",
            type=np.float32,
            doc="This is the total flux from neighboring objects that "
            "overlaps with this source.",
        )
        self.fluxOverlapFractionKey = schema.addField(
            "deblend_fluxOverlapFraction",
            type=np.float32,
            doc="This is the fraction of "
            "`flux from neighbors/source flux` "
            "for a given source within the source's"
            "footprint.",
        )
        self.blendednessKey = schema.addField(
            "deblend_blendedness",
            type=np.float32,
            doc="The Bosch et al. 2018 metric for 'blendedness.' ",
        )

    @timeMethod
    def run(
        self,
        mExposure: afwImage.MultibandExposure,
        mDeconvolved: afwImage.MultibandExposure,
        mergedSources: afwTable.SourceCatalog,
    ) -> tuple[afwTable.SourceCatalog, scl.io.ScarletModelData]:
        """Get the psf from each exposure and then run deblend().

        Parameters
        ----------
        mExposure :
            The exposures should be co-added images of the same
            shape and region of the sky.
        mDeconvolved :
            The deconvolved images of the same shape and region of the sky.
        mergedSources :
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
        return self.deblend(mExposure, mDeconvolved, mergedSources)

    @timeMethod
    def deblend(
        self,
        mExposure: afwImage.MultibandExposure,
        mDeconvolved: afwImage.MultibandExposure,
        catalog: afwTable.SourceCatalog,
    ) -> tuple[afwTable.SourceCatalog, scl.io.ScarletModelData]:
        """Deblend a data cube of multiband images

        Deblending iterates over sources from the input catalog,
        which are blends of peaks with overlapping PSFs (depth 0 parents).
        In many cases those footprints can be subdived into multiple
        deconvolved footprints, which have an intermediate
        parent record added to the catalog and are be deblended separately.
        All deblended peaks have a source record added to the catalog,
        each of which has a depth one greater than the parent.

        Parameters
        ----------
        mExposure :
            The exposures should be co-added images of the same
            shape and region of the sky.
        mDeconvolved :
            The deconvolved images of the same shape and region of the sky.
        catalog :
            The merged `SourceCatalog` that contains parent footprints
            to (potentially) deblend. The new deblended sources are
            appended to this catalog in place.

        Returns
        -------
        catalog :
            The ``deblendedCatalog`` with parents and child sources.
        modelData :
            The persistable data model for the deblender.
        """

        # Cull footprints if required by ci
        if self.config.useCiLimits:
            self.log.info(
                "Using CI catalog limits, the original number of sources to deblend was %d.",
                len(catalog),
            )
            # Select parents with a number of children in the range
            # config.ciDeblendChildRange
            minChildren, maxChildren = self.config.ciDeblendChildRange
            nPeaks = np.array([len(src.getFootprint().peaks) for src in catalog])
            childrenInRange = np.where(
                (nPeaks >= minChildren) & (nPeaks <= maxChildren)
            )[0]
            if len(childrenInRange) < self.config.ciNumParentsToDeblend:
                raise ValueError(
                    "Fewer than ciNumParentsToDeblend children were contained in the range "
                    "indicated by ciDeblendChildRange. Adjust this range to include more "
                    "parents."
                )
            # Keep all of the isolated parents and the first
            # `ciNumParentsToDeblend` children
            parents = nPeaks == 1
            children = np.zeros((len(catalog),), dtype=bool)
            children[childrenInRange[: self.config.ciNumParentsToDeblend]] = True
            catalog = catalog[parents | children]
            # We need to update the IdFactory, otherwise the the source ids
            # will not be sequential
            idFactory = catalog.getIdFactory()
            maxId = np.max(catalog["id"])
            idFactory.notify(maxId)

        self.log.info(
            "Deblending %d sources in %d exposure bands", len(catalog), len(mExposure),
        )
        periodicLog = PeriodicLogger(self.log)

        # Add the NOT_DEBLENDED mask to the mask plane in each band
        if self.config.notDeblendedMask:
            for mask in mExposure.mask:
                mask.addMaskPlane(self.config.notDeblendedMask)

        # Create the context for the entire coadd
        context = ScarletDeblendContext.build(
            mExposure=mExposure,
            mDeconvolved=mDeconvolved,
            config=self.config,
            catalog=catalog,
        )

        # Initialize the persistable ScarletModelData object
        modelData = scl.io.ScarletModelData(context.observation.model_psf[0])

        # Subdivide the psf blended parents into deconvolved parents
        # using the deconvolved footprints.
        self.mExposure = mExposure
        nPsfBlendedParents = len(catalog)
        convolvedParents, parentHierarchy, skippedParents = self._buildParentHierarchy(catalog, context)

        self.log.info(
            "Subdivided the top level parents to create %d deconvolved parents.",
            np.sum((catalog[self.deblendDepthKey] == 1) & (catalog[self.nPeaksKey] > 1))
        )

        # Attach full image objects to the task to simplify the API
        # and use for debugging.
        self.catalog = catalog
        self.context = context
        self.parentHierarchy = parentHierarchy
        self.convolvedParents = convolvedParents
        self.modelData = modelData

        skippedParents = []
        nBands = len(context.observation.bands)

        # Deblend sources
        for index, (convolvedParentId, children) in enumerate(parentHierarchy.items()):
            # Log a message if it has been a while since the last log.
            periodicLog.log(
                "Deblended %d PSF parents out of %d PSF parents",
                index,
                nPsfBlendedParents,
            )

            convolvedParent = catalog.find(convolvedParentId)

            if convolvedParentId in convolvedParents and convolvedParentId not in skippedParents:
                self.log.trace("Deblending parent %d directly", convolvedParent.getId())
                # The deconvolved footprint had all of the same peaks,
                # so there is no deconvolved parent record.
                blendModel = self._deblendParent(
                    blendRecord=convolvedParent,
                    children=children,
                    footprint=convolvedParents[convolvedParentId],
                )
                if blendModel is None:
                    skippedParents.append(convolvedParent.getId())

            else:
                self.log.trace(
                    "Split parent %d into %d deconvolved parents",
                    convolvedParent.getId(),
                    len(children),
                )
                # Create an image to keep track of the cumulative model
                # for all sub blends in the parent footprint.
                convolvedParentFoot = convolvedParent.getFootprint()
                bbox = convolvedParentFoot.getBBox()
                width, height = bbox.getDimensions()
                x0, y0 = bbox.getMin()
                emptyModel = np.zeros(
                    (nBands, height, width),
                    dtype=mExposure.image.array.dtype,
                )
                convolvedParentModel = scl.Image(
                    emptyModel,
                    bands=context.observation.images.bands,
                    yx0=(y0, x0),
                )

                for deconvolvedParentId in children:
                    deconvolvedParent = catalog.find(deconvolvedParentId)
                    blendModel = self._deblendParent(
                        blendRecord=deconvolvedParent,
                        children=children[deconvolvedParentId],
                    )
                    if blendModel is None:
                        skippedParents.append(deconvolvedParentId)
                        convolvedParent.set(self.deblendChildFailedKey, True)
                        continue
                    # Update the parent model
                    convolvedParentModel.insert(blendModel)

                # Calculate the reduced chi2 for the PSF parent
                parentFootprintImage = convolvedParentModel.data > 0
                chi2 = utils.calcChi2(convolvedParentModel, context.observation, parentFootprintImage)

                # Update the PSF parent record with the deblending results
                deconvolvedParents = [
                    catalog.find(childId) for childId in children.keys()
                ]
                self._updateBlendRecord(
                    blendRecord=convolvedParent,
                    nPeaks=len(convolvedParentFoot.peaks),
                    nChild=np.sum(deconvolvedParents[self.nChildKey]),
                    nComponents=np.sum(deconvolvedParents[self.nComponentsKey]),
                    runtime=np.sum(deconvolvedParents[self.runtimeKey]),
                    iterations=np.sum(deconvolvedParents[self.iterKey]),
                    logL=np.nan,
                    chi2=np.sum(chi2.data)/np.sum(parentFootprintImage),
                    spectrumInit=np.all([
                        child[self.scarletSpectrumInitKey]
                        for child in deconvolvedParents
                    ]),  # type: ignore
                    converged=np.all([
                        child[self.blendConvergenceFailedFlagKey]
                        for child in deconvolvedParents
                    ]),  # type: ignore
                )

        # Update the mExposure mask with the footprint of skipped parents
        for sourceIndex in skippedParents:
            children = catalog[catalog["parent"] == sourceIndex]
            # Delete the children of skipped parents that have been added
            # to the catalog, otherwise there will be empty records that
            # will cause the measurement code to fail.
            for child in children:
                del child
            fp = catalog.find(sourceIndex).getFootprint()

            if self.config.notDeblendedMask:
                for mask in mExposure.mask:
                    fp.spans.setMask(
                        mask, mask.getPlaneBitMask(self.config.notDeblendedMask)
                    )

        nDeconvolvedParents = np.sum((catalog[self.deblendDepthKey] == 1) & (catalog[self.nPeaksKey] > 1))
        nDeblendedSources = np.sum((catalog[self.deblendDepthKey] > 0) & (catalog[self.nPeaksKey] == 1))
        self.log.info(
            "Deblender results: %d parent sources were "
            "split into %d deconvovled parents,"
            "resulting in %d deblended sources, "
            "for a total catalog size of %d sources",
            nPsfBlendedParents,
            nDeconvolvedParents,
            nDeblendedSources,
            len(catalog),
        )

        table = afwTable.SourceTable.make(self.schema)
        sortedCatalog = afwTable.SourceCatalog(table)
        parents = catalog[catalog["parent"] == 0]
        sortedCatalog.extend(parents, deep=True)
        parentIds = np.unique(catalog["parent"])
        for parentId in parentIds[parentIds != 0]:
            parent = catalog.find(parentId)
            if not parent.get(self.deblendSkippedKey):
                children = catalog[catalog["parent"] == parentId]
                sortedCatalog.extend(children, deep=True)

        return sortedCatalog, modelData

    def _deblendParent(
        self,
        blendRecord: afwTable.SourceRecord,
        children: dict[int, afwTable.SourceRecord],
        footprint: afwDet.Footprint | None = None,
    ) -> scl.Image | None:
        """Deblend a parent source record

        Parameters
        ----------
        blendRecord :
            The parent source record to deblend.
        catalog :
            The `SourceCatalog` to which the deblended sources will be added.
        children :
            The dict from peak IDs to source records for the children.
        footprint :
            The `Footprint` of the parent source. If None, it will be
            extracted from the `blendRecord`.

        Returns
        -------
        blendModel :
            The `scl.Image` model of the blend.
        """
        if footprint is None:
            footprint = blendRecord.getFootprint()
        assert footprint is not None
        bbox = footprint.getBBox()
        peaks = footprint.getPeaks()

        # Since we use the first peak for the parent object, we should
        # propagate its flags to the parent source.
        blendRecord.assign(peaks[0], self.peakSchemaMapper)

        # Skip the source if it meets the skipping criteria
        if self._checkSkipped(blendRecord, self.mExposure) is not None:
            raise RuntimeError("Skipped parents should be handled when building the parent hierarchy")

        self.log.trace(
            "Parent %d: deblending {%d} peaks",
            blendRecord.getId(),
            len(peaks),
        )
        # Run the deblender
        blendError = None

        # Choose whether or not to use improved spectral initialization.
        # This significantly cuts down on the number of iterations
        # that the optimizer needs and usually results in a better
        # fit.
        # But using least squares on a very large blend causes memory
        # issues, so it is not done for large blends
        if self.config.setSpectra:
            if self.config.maxSpectrumCutoff <= 0:
                spectrumInit = True
            else:
                spectrumInit = (
                    len(footprint.peaks) * bbox.getArea() < self.config.maxSpectrumCutoff
                )
        else:
            spectrumInit = False

        try:
            t0 = time.monotonic()
            # Build the parameter lists with the same ordering
            if self.config.version == "lite":
                blend = deblend(self.context, footprint, self.config, spectrumInit)
            else:
                msg = f"The only currently supported version is 'lite', received {self.config.version}"
                raise NotImplementedError(msg)
            tf = time.monotonic()
            runtime = (tf - t0) * 1000
            converged = _checkBlendConvergence(blend, self.config.relativeError)
            # Store the number of components in the blend
            nComponents = len(blend.components)
            nChild = len(blend.sources)
        # Catch all errors and filter out the ones that we know about
        except Exception as e:
            blendError = type(e).__name__
            if self.config.catchFailures:
                # Make it easy to find UnknownErrors in the log file
                self.log.warn("UnknownError")
                import traceback

                traceback.print_exc()
            else:
                raise

            self._skipParent(
                blendRecord=blendRecord,
                skipKey=self.deblendFailedKey,
                logMessage=f"Unable to deblend source {blendRecord.getId}: {blendError}",
            )
            blendRecord.set(self.deblendErrorKey, blendError)
            return None

        # Calculate the reduced chi2
        blendModel = blend.get_model(convolve=False)
        blendFootprintImage = blendModel.data > 0
        chi2 = utils.calcChi2(blendModel, self.context.observation, blendFootprintImage)

        # Update the blend record with the deblending results
        self._updateBlendRecord(
            blendRecord=blendRecord,
            nPeaks=len(peaks),
            nChild=nChild,
            nComponents=nComponents,
            runtime=runtime,
            iterations=len(blend.loss),
            logL=blend.loss[-1],
            chi2=np.sum(chi2.data)/np.sum(blendFootprintImage),
            spectrumInit=spectrumInit,
            converged=converged,
        )

        # Add each deblended source to the catalog
        for scarletSource in blend.sources:
            # Add all fields except the HeavyFootprint to the
            # source record
            scarletSource.peak_id = scarletSource.detectedPeak.getId()
            sourceRecord = self._updateDeblendedSource(
                parent=blendRecord,
                src=children[scarletSource.peak_id],
                scarletSource=scarletSource,
                chi2=chi2,
            )
            scarletSource.record_id = sourceRecord.getId()

        # Store the blend information so that it can be persisted
        if self.config.version == "lite":
            psfCenter = self.mExposure.getBBox().getCenter()
            blendData = scl.io.ScarletBlendData.from_blend(blend, (psfCenter.y, psfCenter.x))
        else:
            # We keep this here in case other versions are introduced
            raise NotImplementedError(
                "Only the 'lite' version of scarlet is currently supported"
            )

        self.modelData.blends[blendRecord.getId()] = blendData

        return blendModel

    def _isLargeFootprint(self, footprint: afwDet.Footprint) -> bool:
        """Returns whether a Footprint is large

        'Large' is defined by thresholds on the area, size and axis ratio,
        and total area of the bounding box multiplied by
        the number of children.
        These may be disabled independently by configuring them to be
        non-positive.
        """
        if (
            self.config.maxFootprintArea > 0
            and footprint.getArea() > self.config.maxFootprintArea
        ):
            return True
        if self.config.maxFootprintSize > 0:
            bbox = footprint.getBBox()
            if max(bbox.getWidth(), bbox.getHeight()) > self.config.maxFootprintSize:
                return True
        if self.config.minFootprintAxisRatio > 0:
            axes = afwEll.Axes(footprint.getShape())
            if axes.getB() < self.config.minFootprintAxisRatio * axes.getA():
                return True
        if self.config.maxAreaTimesPeaks > 0:
            if (
                footprint.getBBox().getArea() * len(footprint.peaks)
                > self.config.maxAreaTimesPeaks
            ):
                return True
        return False

    def _isMasked(self, footprint: afwDet.Footprint, mExposure: afwImage.MultibandExposure) -> bool:
        """Returns whether the footprint violates the mask limits

        Parameters
        ----------
        footprint :
            The footprint to check for masked pixels
        mExposure :
            The multiband exposure to check for masked pixels.

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
            if (maskedSpan.getArea()) / size > limit:
                return True
        return False

    def _skipParent(
        self,
        blendRecord: afwTable.SourceRecord,
        skipKey: bool,
        logMessage: str | None,
    ):
        """Update a parent record that is not being deblended.

        This is a fairly trivial function but is implemented to ensure
        that a skipped parent updates the appropriate columns
        consistently, and always has a flag to mark the reason that
        it is being skipped.

        Parameters
        ----------
        parent :
            The parent record to flag as skipped.
        skipKey :
            The name of the flag to mark the reason for skipping.
        logMessage :
            The message to display in a log.trace when a source
            is skipped.
        """
        if logMessage is not None:
            self.log.trace(logMessage)
        self._updateBlendRecord(
            blendRecord=blendRecord,
            nPeaks=len(blendRecord.getFootprint().peaks),
            nChild=0,
            nComponents=0,
            runtime=np.nan,
            iterations=0,
            logL=np.nan,
            chi2=np.nan,
            spectrumInit=False,
            converged=False,
        )

        # Mark the source as skipped by the deblender and
        # flag the reason why.
        blendRecord.set(self.deblendSkippedKey, True)
        blendRecord.set(skipKey, True)

    def _checkSkipped(self, parent: afwTable.SourceRecord, mExposure: afwImage.MultibandExposure) -> bool:
        """Update a parent record that is not being deblended.

        This is a fairly trivial function but is implemented to ensure
        that a skipped parent updates the appropriate columns
        consistently, and always has a flag to mark the reason that
        it is being skipped.

        Parameters
        ----------
        parent :
            The parent record to flag as skipped.
        mExposure :
            The exposures should be co-added images of the same
            shape and region of the sky.

        Returns
        -------
        skip :
            `True` if the deblender will skip the parent
        """
        skipKey = None
        skipMessage = None
        footprint = parent.getFootprint()
        if (len(footprint.peaks) < 2
                and not self.config.processSingles
                and parent.get(self.deblendDepthKey) == 0):
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
        elif (
            self.config.maxNumberOfPeaks > 0
            and len(footprint.peaks) > self.config.maxNumberOfPeaks
        ):
            # Unlike meas_deblender, in scarlet we skip the entire blend
            # if the number of peaks exceeds max peaks, since neglecting
            # to model any peaks often results in catastrophic failure
            # of scarlet to generate models for the brighter sources.
            skipKey = self.tooManyPeaksKey
            skipMessage = f"Parent {parent.getId()}: skipping blend with too many peaks"
        if skipKey is not None:
            return (skipKey, skipMessage)
        return None

    def _updateBlendRecord(
        self,
        blendRecord: afwTable.SourceRecord,
        nPeaks: int,
        nChild: int,
        nComponents: int,
        runtime: float,
        iterations: int,
        logL: float,
        chi2: float,
        spectrumInit: bool,
        converged: bool,
    ):
        """Update a parent record in all of the single band catalogs.

        Ensure that all locations that update a blend record,
        whether it is skipped or updated after deblending,
        update all of the appropriate columns.

        Parameters
        ----------
        blendRecord :
            The catalog record to update.
        nPeaks :
            Number of peaks in the parent footprint.
        nChild :
            Number of children deblended from the parent.
            This may differ from `nPeaks` if some of the peaks
            were culled and have no deblended model.
        nComponents :
            Total number of components in the parent.
            This is usually different than the number of children,
            since it is common for a single source to have multiple
            components.
        runtime :
            Total runtime for deblending.
        iterations :
            Total number of iterations in scarlet before convergence.
        logL :
            Final log likelihood of the blend.
        chi2 :
            Final reduced chi2 of the blend.
        spectrumInit :
            True when scarlet used `set_spectra` to initialize all
            sources with better initial intensities.
        converged :
            True when the optimizer reached convergence before
            reaching the maximum number of iterations.
        """
        blendRecord.set(self.nPeaksKey, nPeaks)
        blendRecord.set(self.nChildKey, nChild)
        blendRecord.set(self.nComponentsKey, nComponents)
        blendRecord.set(self.runtimeKey, runtime)
        blendRecord.set(self.iterKey, iterations)
        blendRecord.set(self.scarletLogLKey, logL)
        blendRecord.set(self.scarletSpectrumInitKey, spectrumInit)
        blendRecord.set(self.blendConvergenceFailedFlagKey, converged)
        blendRecord.set(self.scarletChi2Key, chi2)

    def _buildParentHierarchy(
        self,
        catalog: afwTable.SourceCatalog,
        context: ScarletDeblendContext,
    ) -> tuple[dict[int, list[afwDet.Footprint]], dict[int, Any]]:
        """Add sub-parents to the catalog

        Each parent may be subdivided into multiple blends that are
        isolated in deconvolved space but still blended in the image.

        The measurement tasks require the catalog to be ordered by parent ID,
        so we add records for all of the deblended children and deconvolved
        parents here. Many of the fields will be updated after deblending,
        but the static fields are set when they are added.

        Parameters
        ----------
        catalog :
            The merged `SourceCatalog` that contains parent footprints.
        context :
            The context for the entire coadd.

        Returns
        -------
        deconvolvedParents :
            A dictionary where the keys are the parent ids and the values
            are lists of deconvolved parents.
        """
        nParents = len(catalog)
        convolvedParents = {}
        deconvolvedParents = {}
        hierarchy = {}
        skipped = []

        def _addChildren(parent: afwTable.SourceRecord):
            """Add a child record for every peak in the parent footprint"""
            children = {}
            footprint = parent.getFootprint()
            for peak in footprint.peaks:
                child = self._addDeblendedSource(
                    parent=parent,
                    peak=peak,
                    catalog=catalog,
                )
                children[peak.getId()] = child
            return children

        for n in range(nParents):
            convolvedParent = catalog[n]
            parentFoot = convolvedParent.getFootprint()
            # Since we use the first peak for the parent object, we should
            # propagate its flags to the parent source.
            # For example, this propagates `merge_peak_sky` to the parent
            convolvedParent.assign(convolvedParent.getFootprint().peaks[0], self.peakSchemaMapper)

            if (skipArgs := self._checkSkipped(convolvedParent, self.mExposure)) is not None:
                self._skipParent(convolvedParent, *skipArgs)
                # Do not add sources for skipped parents
                skipped.append(convolvedParent.getId())
                continue
            parentId = convolvedParent.getId()
            deconvolvedFootprints = self._getIntersectingFootprints(
                parentFoot,
                context.footprints,
                context.footprintImage
            )
            if len(deconvolvedFootprints) == 1:
                # There is only one deconvolved footprint, so there is no
                # need to make a child record. However, we do want to keep
                # the deconvolved footprint in order to keep the initial
                # and final model compact.
                convolvedParents[parentId] = deconvolvedFootprints[0]
                hierarchy[parentId] = _addChildren(convolvedParent)
            else:
                # Add a source record for the deconvolved child and
                # add it to the dictionary of deconvolved parents.
                deconvolvedParents[parentId] = self._addDeconvolvedParents(
                    convolvedParent,
                    catalog,
                    deconvolvedFootprints
                )

        # Add children for all of the deconvolved parents
        for parentId, sources in deconvolvedParents.items():
            hierarchy[parentId] = {}
            for sourceRecord in sources:
                if (skipArgs := self._checkSkipped(sourceRecord, self.mExposure)) is not None:
                    self._skipParent(sourceRecord, *skipArgs)
                    skipped.append(sourceRecord.getId())
                    continue
                hierarchy[parentId][sourceRecord.getId()] = _addChildren(sourceRecord)

        return convolvedParents, hierarchy, skipped

    def _getIntersectingFootprints(
        self,
        afwFootprint: afwDet.Footprint,
        sclFootprints: list[scl.detect.Footprint],
        footprintImage: scl.Image,
    ) -> list[afwDet.Footprint]:
        """Get the intersection of two footprints

        Parameters
        ----------
        afwFootprint :
            The afw footprint
        sclFootprint :
            The scarlet footprint
        footprintImage :
            An indexed image of the scarlet footprints so that the value
            of a pixel gives the index + 1 of the footprints that
            contain that pixel.

        Returns
        -------
        intersection :
            The intersection of the two footprints
        """
        footprintIndices = set()
        ymin, xmin = footprintImage.bbox.origin

        for peak in afwFootprint.peaks:
            x = peak["i_x"] - xmin
            y = peak["i_y"] - ymin
            try:
                footprintIndex = footprintImage.data[y, x] - 1
            except:  # noqa: E722
                continue
            if footprintIndex >= 0:
                footprintIndices.add(footprintIndex)
            else:
                raise RuntimeError(f"no footprint at ({y}, {x})")

        footprints = []
        for index in footprintIndices:
            _sclFootprint = scarletFootprintToAfw(sclFootprints[index])
            intersection = getFootprintIntersection(afwFootprint, _sclFootprint, copyFromFirst=True)
            if len(intersection.peaks) > 0:
                footprints.append(intersection)

        return footprints

    def _addDeconvolvedParents(
        self,
        parent: afwTable.SourceRecord,
        catalog: afwTable.SourceCatalog,
        footprints: afwDet.Footprint,
    ) -> list[afwTable.SourceRecord]:
        """Add sub-parents to the catalog

        Each parent may be subdivided into multiple blends that are
        isolated in deconvolved space but still blended in the image.
        This function adds the sub-parents to the catalog.

        Parameters
        ----------
        parent :
            The parent of the sub-parents.
        footprints :
            The footprints of the sub-parents

        Results
        -------
        deconvolvedParents :
            A list of the deconvolved parents.
        """
        deconvolvedParents = []
        for footprint in footprints:
            deconvolvedParent = catalog.addNew()
            deconvolvedParents.append(deconvolvedParent)
            deconvolvedParent.setParent(parent.getId())
            deconvolvedParent.setFootprint(footprint)
            deconvolvedParent.set(self.deblendDepthKey, parent[self.deblendDepthKey] + 1)
            self._propagateToChild(parent, deconvolvedParent)

        return deconvolvedParents

    def _addDeblendedSource(
        self,
        parent: afwTable.SourceRecord,
        peak: afwDet.PeakRecord,
        catalog: afwTable.SourceCatalog,
    ):
        """Add a deblended source to a catalog.

        This creates a new child in the source catalog,
        assigning it a parent id, and adding all columns
        that are independent across all filter bands and not
        updated after deblending.

        Parameters
        ----------
        parent :
            The parent of the new child record.
        peak :
            The peak record for the peak from the parent peak catalog.
        catalog :
            The source catalog that the child is added to.

        Returns
        -------
        src :
            The new child source record.
        """
        src = catalog.addNew()
        # The peak catalog is the same for all bands,
        # so we just use the first peak catalog
        src.assign(peak, self.peakSchemaMapper)
        src.setParent(parent.getId())
        src.set(self.nPeaksKey, 1)
        src.set(self.nChildKey, 0)
        # We set the runtime to zero so that summing up the
        # runtime column will give the total time spent
        # running the deblender for the catalog.
        src.set(self.runtimeKey, 0)

        # Set the position of the peak from the parent footprint
        # This will make it easier to match the same source across
        # deblenders and across observations, where the peak
        # position is unlikely to change unless enough time passes
        # for a source to move on the sky.
        src.set(self.peakCenter, geom.Point2I(peak["i_x"], peak["i_y"]))
        src.set(self.peakIdKey, peak["id"])

        # Set the deblend depth
        src.set(self.deblendDepthKey, parent[self.deblendDepthKey] + 1)

        return src

    def _updateDeblendedSource(
        self,
        parent: afwTable.SourceRecord,
        src: afwTable.SourceRecord,
        scarletSource: scl.Source,
        chi2: scl.Image,
    ):
        """Add a deblended source to a catalog.

        This creates a new child in the source catalog,
        assigning it a parent id, and adding all columns
        that are independent across all filter bands.

        Parameters
        ----------
        parent :
            The parent of the new child record.
        src :
            The peak record for the peak from the parent peak catalog.
        scarletSource :
            The scarlet model for the new source record.
        chi2 :
            The chi2 for each pixel.
        """
        # Set the psf key based on whether or not the source was
        # deblended using the PointSource model.
        # This key is not that useful anymore since we now keep track of
        # `modelType`, but we continue to propagate it in case code downstream
        # is expecting it.
        src.set(self.psfKey, scarletSource.__class__.__name__ == "PointSource")
        src.set(self.modelTypeKey, scarletSource.__class__.__name__)

        # Store the number of components for the source
        src.set(self.nComponentsKey, len(scarletSource.components))

        # Calculate the reduced chi2 for the source
        area = np.sum(scarletSource.get_model().data > 0)
        src.set(self.scarletChi2Key, np.sum(chi2[:, scarletSource.bbox].data/area))

        # Propagate columns from the parent to the child
        self._propagateToChild(parent, src)
        return src

    def _propagateToChild(self, parent: afwTable.SourceRecord, child: afwTable.SourceRecord):
        """Propagate columns from the parent to the child.

        Parameters
        ----------
        parent :
            The parent source record.
        child :
            The child source record.
        """
        for key in self.toCopyFromParent:
            child.set(key, parent.get(key))
        for parentColumn, childColumn in self.config.columnInheritance.items():
            child.set(childColumn, parent.get(parentColumn))
