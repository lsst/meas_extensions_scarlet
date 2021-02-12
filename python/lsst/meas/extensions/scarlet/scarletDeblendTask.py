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
import numpy as np
import scarlet
from scarlet.psf import ImagePSF, GaussianPSF
from scarlet import Blend, Frame, Observation
from scarlet.renderer import ConvolutionRenderer
from scarlet.initialization import init_all_sources

import lsst.log
import lsst.pex.config as pexConfig
from lsst.pex.exceptions import InvalidParameterError
import lsst.pipe.base as pipeBase
from lsst.geom import Point2I, Box2I, Point2D
import lsst.afw.geom.ellipses as afwEll
import lsst.afw.image.utils
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.table as afwTable

from .source import modelToHeavy

# scarlet initialization allows the user to specify the maximum number
# of components for a source but will fall back to fewer components or
# an initial PSF morphology depending on the S/N. If either of those happen
# then scarlet currently warnings that the type of source created by the
# user was modified. This is not ideal behavior, as it creates a lot of
# unnecessary warnings for expected behavior and the information is
# already persisted due to the change in source type.
# So we silence all of the initialization warnings here to prevent
# polluting the log files.
scarletInitLogger = logging.getLogger("scarlet.initialisation")
scarletSourceLogger = logging.getLogger("scarlet.source")
scarletInitLogger.setLevel(logging.ERROR)
scarletSourceLogger.setLevel(logging.ERROR)

__all__ = ["deblend", "ScarletDeblendConfig", "ScarletDeblendTask"]

logger = lsst.log.Log.getLogger("meas.deblender.deblend")


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


def _getPsfFwhm(psf):
    """Calculate the FWHM of the `psf`
    """
    return psf.computeShape().getDeterminantRadius() * 2.35


def _computePsfImage(self, position=None):
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
        Coordinates to evaluate the PSF. If `position` is `None`
        then `Psf.getAveragePosition()` is used.
    Returns
    -------
    self._psfImage: array
        The multiband PSF image.
    """
    psfs = []
    # Make the coordinates into a Point2D (if necessary)
    if not isinstance(position, Point2D) and position is not None:
        position = Point2D(position[0], position[1])

    for bidx, single in enumerate(self.singles):
        try:
            if position is None:
                psf = single.getPsf().computeImage()
                psfs.append(psf)
            else:
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
    centers = [np.array([peak.getIy()-ymin, peak.getIx()-xmin], dtype=int) for peak in footprint.peaks]

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
        set_spectra=config.setSpectra,
    )

    # Attach the peak to all of the initialized sources
    srcIndex = 0
    for k, center in enumerate(centers):
        if k not in skipped:
            # This is just to make sure that there isn't a coding bug
            assert np.all(sources[srcIndex].center == center)
            # Store the record for the peak with the appropriate source
            sources[srcIndex].detectedPeak = footprint.peaks[k]
            # Turn off box resizing to prevent runaway box sizes
            if not config.resizeBoxes:
                sources[srcIndex]._resize = False
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

    return blend, skipped


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
    maxIter = pexConfig.Field(dtype=int, default=300,
                              doc=("Maximum number of iterations to deblend a single parent"))
    relativeError = pexConfig.Field(dtype=float, default=1e-4,
                                    doc=("Change in the loss function between"
                                         "iterations to exit fitter"))

    # Constraints
    morphThresh = pexConfig.Field(dtype=float, default=1,
                                  doc="Fraction of background RMS a pixel must have"
                                      "to be included in the initial morphology")
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
             "- 'fit: use a PSF fitting model to determine the number of components (not yet implemented)")
    )
    setSpectra = pexConfig.Field(
        dtype=bool, default=True,
        doc="Whether or not to solve for the best-fit spectra during initialization. "
            "This makes initialization slightly longer, as it requires a convolution "
            "to set the optimal spectra, but results in a much better initial log-likelihood "
            "and reduced total runtime, with convergence in fewer iterations.")
    resizeBoxes = pexConfig.Field(
        dtype=bool, default=False,
        doc="Whether or not to resize boxes based on the gradient updates for each source. "
            "This is the preferred behavior, as it will allow bright sources to grow and "
            "avoid truncating flux in the wings and also (in theory) make the boxes for "
            "faint sources smaller. "
            "However, in HSC RC2 reprocessing we found that the boxes are growing too large "
            "so until we have better control over this algorithm the default is to NOT "
            "resize the boxes (see DM-28805).")

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
             "Sources violating this limit will not be deblended."),
    )

    # Size restrictions
    maxNumberOfPeaks = pexConfig.Field(
        dtype=int, default=0,
        doc=("Only deblend the brightest maxNumberOfPeaks peaks in the parent"
             " (<= 0: unlimited)"))
    maxFootprintArea = pexConfig.Field(
        dtype=int, default=1000000,
        doc=("Maximum area for footprints before they are ignored as large; "
             "non-positive means no threshold applied"))
    maxFootprintSize = pexConfig.Field(
        dtype=int, default=0,
        doc=("Maximum linear dimension for footprints before they are ignored "
             "as large; non-positive means no threshold applied"))
    minFootprintAxisRatio = pexConfig.Field(
        dtype=float, default=0.0,
        doc=("Minimum axis ratio for footprints before they are ignored "
             "as large; non-positive means no threshold applied"))

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
        self.runtimeKey = schema.addField('deblend_runtime', type=np.float32, doc='runtime in ms')

        self.iterKey = schema.addField('deblend_iterations', type=np.int32, doc='iterations to converge')

        self.nChildKey = schema.addField('deblend_nChild', type=np.int32,
                                         doc='Number of children this object has (defaults to 0)')
        self.psfKey = schema.addField('deblend_deblendedAsPsf', type='Flag',
                                      doc='Deblender thought this source looked like a PSF')
        self.tooManyPeaksKey = schema.addField('deblend_tooManyPeaks', type='Flag',
                                               doc='Source had too many peaks; '
                                               'only the brightest were included')
        self.tooBigKey = schema.addField('deblend_parentTooBig', type='Flag',
                                         doc='Parent footprint covered too many pixels')
        self.maskedKey = schema.addField('deblend_masked', type='Flag',
                                         doc='Parent footprint was predominantly masked')
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
        self.edgePixelsKey = schema.addField('deblend_edgePixels', type='Flag',
                                             doc='Source had flux on the edge of the parent footprint')
        self.deblendFailedKey = schema.addField('deblend_failed', type='Flag',
                                                doc="Deblending failed on source")
        self.deblendErrorKey = schema.addField('deblend_error', type="String", size=25,
                                               doc='Name of error if the blend failed')
        self.deblendSkippedKey = schema.addField('deblend_skipped', type='Flag',
                                                 doc="Deblender skipped this source")
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
        self.nPeaksKey = schema.addField("deblend_nPeaks", type=np.int32,
                                         doc="Number of initial peaks in the blend. "
                                             "This includes peaks that may have been culled "
                                             "during deblending or failed to deblend")
        self.parentNPeaksKey = schema.addField("deblend_parentNPeaks", type=np.int32,
                                               doc="Same as deblend_n_peaks, but the number of peaks "
                                                   "in the parent footprint")
        self.scarletFluxKey = schema.addField("deblend_scarletFlux", type=np.float32,
                                              doc="Flux measurement from scarlet")
        self.scarletLogLKey = schema.addField("deblend_logL", type=np.float32,
                                              doc="Final logL, used to identify regressions in scarlet.")

        # self.log.trace('Added keys to schema: %s', ", ".join(str(x) for x in
        #               (self.nChildKey, self.tooManyPeaksKey, self.tooBigKey))
        #               )

    @pipeBase.timeMethod
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

    @pipeBase.timeMethod
    def deblend(self, mExposure, sources):
        """Deblend a data cube of multiband images

        Parameters
        ----------
        mExposure : `MultibandExposure`
            The exposures should be co-added images of the same
            shape and region of the sky.
        sources : `SourceCatalog`
            The merged `SourceCatalog` that contains parent footprints
            to (potentially) deblend.

        Returns
        -------
        templateCatalogs : dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are catalogs with heavy footprints that are the templates
            created by the multiband templates.
        """
        import time

        filters = mExposure.filters
        self.log.info("Deblending {0} sources in {1} exposure bands".format(len(sources), len(mExposure)))

        # Create the output catalogs
        templateCatalogs = {}
        # This must be returned but is not calculated right now, setting it to
        # None to be consistent with doc string
        for f in filters:
            _catalog = afwTable.SourceCatalog(sources.table.clone())
            _catalog.extend(sources)
            templateCatalogs[f] = _catalog

        n0 = len(sources)
        nparents = 0
        for pk, src in enumerate(sources):
            foot = src.getFootprint()
            bbox = foot.getBBox()
            peaks = foot.getPeaks()

            # Since we use the first peak for the parent object, we should
            # propagate its flags to the parent source.
            src.assign(peaks[0], self.peakSchemaMapper)

            # Block of Skipping conditions
            if len(peaks) < 2 and not self.config.processSingles:
                for f in filters:
                    templateCatalogs[f][pk].set(self.runtimeKey, 0)
                continue
            if self._isLargeFootprint(foot):
                src.set(self.tooBigKey, True)
                self._skipParent(src, mExposure.mask)
                self.log.trace('Parent %i: skipping large footprint', int(src.getId()))
                continue
            if self._isMasked(foot, mExposure):
                src.set(self.maskedKey, True)
                mask = np.bitwise_or.reduce(mExposure.mask[:, bbox].array, axis=0)
                mask = afwImage.MaskX(mask, xy0=bbox.getMin())
                self._skipParent(src, mask)
                self.log.trace('Parent %i: skipping masked footprint', int(src.getId()))
                continue
            if self.config.maxNumberOfPeaks > 0 and len(peaks) > self.config.maxNumberOfPeaks:
                src.set(self.tooManyPeaksKey, True)
                self._skipParent(src, mExposure.mask)
                msg = 'Parent {0}: Too many peaks, skipping blend'
                self.log.trace(msg.format(int(src.getId())))
                # Unlike meas_deblender, in scarlet we skip the entire blend
                # if the number of peaks exceeds max peaks, since neglecting
                # to model any peaks often results in catastrophic failure
                # of scarlet to generate models for the brighter sources.
                continue

            nparents += 1
            self.log.trace('Parent %i: deblending %i peaks', int(src.getId()), len(peaks))
            # Run the deblender
            blendError = None
            try:
                t0 = time.time()
                # Build the parameter lists with the same ordering
                blend, skipped = deblend(mExposure, foot, self.config)
                tf = time.time()
                runtime = (tf-t0)*1000
                src.set(self.deblendFailedKey, False)
                src.set(self.runtimeKey, runtime)
                converged = _checkBlendConvergence(blend, self.config.relativeError)
                src.set(self.blendConvergenceFailedFlagKey, converged)
                sources = [src for src in blend.sources]
                # Re-insert place holders for skipped sources
                # to propagate them in the catalog so
                # that the peaks stay consistent
                for k in skipped:
                    sources.insert(k, None)
            # Catch all errors and filter out the ones that we know about
            except Exception as e:
                blendError = type(e).__name__
                if isinstance(e, ScarletGradientError):
                    src.set(self.iterKey, e.iterations)
                elif not isinstance(e, IncompleteDataError):
                    blendError = "UnknownError"
                    self._skipParent(src, mExposure.mask)
                    if self.config.catchFailures:
                        # Make it easy to find UnknownErrors in the log file
                        self.log.warn("UnknownError")
                        import traceback
                        traceback.print_exc()
                    else:
                        raise

                self.log.warn("Unable to deblend source %d: %s" % (src.getId(), blendError))
                src.set(self.deblendFailedKey, True)
                src.set(self.deblendErrorKey, blendError)
                self._skipParent(src, mExposure.mask)
                continue

            # Add the merged source as a parent in the catalog for each band
            templateParents = {}
            parentId = src.getId()
            for f in filters:
                templateParents[f] = templateCatalogs[f][pk]
                templateParents[f].set(self.nPeaksKey, len(foot.peaks))
                templateParents[f].set(self.runtimeKey, runtime)
                templateParents[f].set(self.iterKey, len(blend.loss))
                logL = blend.loss[-1]-blend.observations[0].log_norm
                templateParents[f].set(self.scarletLogLKey, logL)

            # Add each source to the catalogs in each band
            nchild = 0
            for k, source in enumerate(sources):
                # Skip any sources with no flux or that scarlet skipped because
                # it could not initialize
                if k in skipped:
                    # No need to propagate anything
                    continue
                else:
                    src.set(self.deblendSkippedKey, False)
                    models = modelToHeavy(source, filters, xy0=bbox.getMin(),
                                          observation=blend.observations[0])

                flux = scarlet.measure.flux(source)
                for fidx, f in enumerate(filters):
                    if len(models[f].getPeaks()) != 1:
                        err = "Heavy footprint should have a single peak, got {0}"
                        raise ValueError(err.format(len(models[f].peaks)))
                    cat = templateCatalogs[f]
                    child = self._addChild(src, cat, models[f], source, converged,
                                           xy0=bbox.getMin(), flux=flux[fidx])
                    if parentId == 0:
                        child.setId(src.getId())
                        child.set(self.runtimeKey, runtime)
                nchild += 1

            # Set the number of children for each parent
            for f in filters:
                templateParents[f].set(self.nChildKey, nchild)

        K = len(list(templateCatalogs.values())[0])
        self.log.info('Deblended: of %i sources, %i were deblended, creating %i children, total %i sources'
                      % (n0, nparents, K-n0, K))
        return templateCatalogs

    def _isLargeFootprint(self, footprint):
        """Returns whether a Footprint is large

        'Large' is defined by thresholds on the area, size and axis ratio.
        These may be disabled independently by configuring them to be
        non-positive.

        This is principally intended to get rid of satellite streaks, which the
        deblender or other downstream processing can have trouble dealing with
        (e.g., multiple large HeavyFootprints can chew up memory).
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
        return False

    def _isMasked(self, footprint, mExposure):
        """Returns whether the footprint violates the mask limits"""
        bbox = footprint.getBBox()
        mask = np.bitwise_or.reduce(mExposure.mask[:, bbox].array, axis=0)
        size = float(footprint.getArea())
        for maskName, limit in self.config.maskLimits.items():
            maskVal = mExposure.mask.getPlaneBitMask(maskName)
            _mask = afwImage.MaskX(mask & maskVal, xy0=bbox.getMin())
            unmaskedSpan = footprint.spans.intersectNot(_mask)  # spanset of unmasked pixels
            if (size - unmaskedSpan.getArea())/size > limit:
                return True
        return False

    def _skipParent(self, source, masks):
        """Indicate that the parent source is not being deblended

        We set the appropriate flags and masks for each exposure.

        Parameters
        ----------
        source : `lsst.afw.table.source.source.SourceRecord`
            The source to flag as skipped
        masks : list of `lsst.afw.image.MaskX`
            The mask in each band to update with the non-detection
        """
        fp = source.getFootprint()
        source.set(self.deblendSkippedKey, True)
        if self.config.notDeblendedMask:
            for mask in masks:
                mask.addMaskPlane(self.config.notDeblendedMask)
                fp.spans.setMask(mask, mask.getPlaneBitMask(self.config.notDeblendedMask))
        # The deblender didn't run on this source, so it has zero runtime
        source.set(self.runtimeKey, 0)
        # Set the center of the parent
        bbox = fp.getBBox()
        centerX = int(bbox.getMinX()+bbox.getWidth()/2)
        centerY = int(bbox.getMinY()+bbox.getHeight()/2)
        source.set(self.peakCenter, Point2I(centerX, centerY))
        # There are no deblended children, so nChild = 0
        source.set(self.nChildKey, 0)
        # But we also want to know how many peaks that we would have
        # deblended if the parent wasn't skipped.
        source.set(self.nPeaksKey, len(fp.peaks))
        # The blend was skipped, so it didn't take any iterations
        source.set(self.iterKey, 0)
        # Top level parents are not a detected peak, so they have no peakId
        source.set(self.peakIdKey, 0)
        # Top level parents also have no parentNPeaks
        source.set(self.parentNPeaksKey, 0)

    def _addChild(self, parent, sources, heavy, scarletSource, blend_converged, xy0, flux):
        """Add a child to a catalog

        This creates a new child in the source catalog,
        assigning it a parent id, adding a footprint,
        and setting all appropriate flags based on the
        deblender result.
        """
        assert len(heavy.getPeaks()) == 1
        src = sources.addNew()
        for key in self.toCopyFromParent:
            src.set(key, parent.get(key))
        src.assign(heavy.getPeaks()[0], self.peakSchemaMapper)
        src.setParent(parent.getId())
        src.setFootprint(heavy)
        # Set the psf key based on whether or not the source was
        # deblended using the PointSource model.
        # This key is not that useful anymore since we now keep track of
        # `modelType`, but we continue to propagate it in case code downstream
        # is expecting it.
        src.set(self.psfKey, scarletSource.__class__.__name__ == "PointSource")
        src.set(self.runtimeKey, 0)
        src.set(self.blendConvergenceFailedFlagKey, not blend_converged)

        # Set the position of the peak from the parent footprint
        # This will make it easier to match the same source across
        # deblenders and across observations, where the peak
        # position is unlikely to change unless enough time passes
        # for a source to move on the sky.
        peak = scarletSource.detectedPeak
        src.set(self.peakCenter, Point2I(peak["i_x"], peak["i_y"]))
        src.set(self.peakIdKey, peak["id"])

        # The children have a single peak
        src.set(self.nPeaksKey, 1)

        # Store the flux at the center of the model and the total
        # scarlet flux measurement.
        morph = afwDet.multiband.heavyFootprintToImage(heavy).image.array

        # Set the flux at the center of the model (for SNR)
        try:
            cy, cx = scarletSource.center
            cy = np.max([np.min([int(np.round(cy)), morph.shape[0]-1]), 0])
            cx = np.max([np.min([int(np.round(cx)), morph.shape[1]-1]), 0])
            src.set(self.modelCenterFlux, morph[cy, cx])
        except AttributeError:
            msg = "Did not recognize coordinates for source type of `{0}`, "
            msg += "could not write coordinates or center flux. "
            msg += "Add `{0}` to meas_extensions_scarlet to properly persist this information."
            logger.warning(msg.format(type(scarletSource)))

        src.set(self.modelTypeKey, scarletSource.__class__.__name__)
        # Include the source flux in the model space in the catalog.
        # This uses the narrower model PSF, which ensures that all sources
        # not located on an edge have all of their flux included in the
        # measurement.
        src.set(self.scarletFluxKey, flux)
        return src
