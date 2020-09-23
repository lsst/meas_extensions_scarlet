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

from functools import partial

import numpy as np
import scarlet
from scarlet.psf import PSF, gaussian
from scarlet import PointSource, ExtendedSource, MultiComponentSource, Blend, Frame, Observation

import lsst.log
import lsst.pex.config as pexConfig
from lsst.pex.exceptions import InvalidParameterError
import lsst.pipe.base as pipeBase
from lsst.geom import Point2I, Box2I, Point2D
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as afwEll
import lsst.afw.image.utils
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.table as afwTable

from .source import initSource, modelToHeavy

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


def _estimateRMS(exposure, statsMask):
    """Estimate the standard dev. of an image

    Calculate the RMS of the `exposure`.
    """
    mi = exposure.getMaskedImage()
    statsCtrl = afwMath.StatisticsControl()
    statsCtrl.setAndMask(mi.getMask().getPlaneBitMask(statsMask))
    stats = afwMath.makeStatistics(mi.variance, mi.mask, afwMath.STDEV | afwMath.MEAN, statsCtrl)
    rms = np.sqrt(stats.getValue(afwMath.MEAN)**2 + stats.getValue(afwMath.STDEV)**2)
    return rms


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

    for band, single in enumerate(self.singles):
        try:
            if position is None:
                psf = single.getPsf().computeImage()
                psfs.append(psf)
            else:
                psf = single.getPsf().computeImage(position)
                psfs.append(psf)
        except InvalidParameterError:
            # This band failed to compute the PSF due to incomplete data
            # at that location. This is unlikely to be a problem for Rubin,
            # however the edges of some HSC COSMOS fields contain incomplete
            # data in some bands, so we track this error to distinguish it
            # from unknown errors.
            msg = "Failed to compute PSF at {} in band {}"
            raise IncompleteDataError(msg.format(position, self.filters[band]))

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

    # Mask out the pixels outside the footprint
    mask = getFootprintMask(footprint, mExposure)
    weights *= ~mask

    psfs = _computePsfImage(mExposure, footprint.getCentroid()).array.astype(np.float32)

    psfShape = (config.modelPsfSize, config.modelPsfSize)
    model_psf = PSF(partial(gaussian, sigma=config.modelPsfSigma), shape=(None,)+psfShape)

    frame = Frame(images.shape, psfs=model_psf, channels=mExposure.filters)
    observation = Observation(images, psfs=psfs, weights=weights, channels=mExposure.filters)
    observation.match(frame)

    assert(config.sourceModel in ["single", "double", "point", "fit"])

    # Set the appropriate number of components
    if config.sourceModel == "single":
        maxComponents = 1
    elif config.sourceModel == "double":
        maxComponents = 2
    elif config.sourceModel == "point":
        maxComponents = 0
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
    sources = []
    skipped = []
    for k, center in enumerate(centers):
        source = initSource(
            frame=frame,
            center=center,
            observation=observation,
            symmetric=config.symmetric,
            monotonic=config.monotonic,
            thresh=config.morphThresh,
            maxComponents=maxComponents,
            edgeDistance=config.edgeDistance,
            shifting=False,
            downgrade=config.downgrade,
            fallback=config.fallback,
        )
        if source is not None:
            source.detectedPeak = footprint.peaks[k]
            sources.append(source)
        else:
            skipped.append(k)

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

    # Blend Configuration options
    edgeDistance = pexConfig.Field(dtype=int, default=1,
                                   doc="All sources with flux within `edgeDistance` from the edge "
                                       "will be considered edge sources.")

    # Constraints
    morphThresh = pexConfig.Field(dtype=float, default=1,
                                  doc="Fraction of background RMS a pixel must have"
                                      "to be included in the initial morphology")
    monotonic = pexConfig.Field(dtype=bool, default=True, doc="Make models monotonic")
    symmetric = pexConfig.Field(dtype=bool, default=False, doc="Make models symmetric")

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
    saveTemplates = pexConfig.Field(
        dtype=bool, default=True,
        doc="Whether or not to save the SEDs and templates")
    processSingles = pexConfig.Field(
        dtype=bool, default=False,
        doc="Whether or not to process isolated sources in the deblender")
    sourceModel = pexConfig.Field(
        dtype=str, default="single",
        doc=("How to determine which model to use for sources, from\n"
             "- 'single': use a single component for all sources\n"
             "- 'double': use a bulge disk model for all sources\n"
             "- 'point: use a point-source model for all sources\n"
             "- 'fit: use a PSF fitting model to determine the number of components (not yet implemented)")
    )
    downgrade = pexConfig.Field(
        dtype=bool, default=False,
        doc="Whether or not to downgrade the number of components for sources in small bounding boxes"
    )

    # Mask-plane restrictions
    badMask = pexConfig.ListField(
        dtype=str, default=["BAD", "CR", "NO_DATA", "SAT", "SUSPECT"],
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
        dtype=bool, default=False,
        doc=("If True, catch exceptions thrown by the deblender, log them, "
             "and set a flag on the parent, instead of letting them propagate up"))
    propagateAllPeaks = pexConfig.Field(dtype=bool, default=False,
                                        doc=('Guarantee that all peaks produce a child source.'))


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
        self.modelCenter = afwTable.Point2DKey.addFields(schema, name="deblend_peak_center",
                                                         doc="Center used to apply constraints in scarlet",
                                                         unit="pixel")
        self.modelCenterFlux = schema.addField('deblend_peak_instFlux', type=float, units='count',
                                               doc="The instFlux at the peak position of deblended mode")
        self.modelTypeKey = schema.addField("deblend_modelType", type="String", size=20,
                                            doc="The type of model used, for example "
                                                "MultiComponentSource, ExtendedSource, PointSource")
        self.edgeFluxFlagKey = schema.addField("deblend_edgeFluxFlag", type="Flag",
                                               doc="Source has flux on the edge of the image")
        self.scarletFluxKey = schema.addField("deblend_scarletFlux", type=np.float32,
                                              doc="Flux measurement from scarlet")
        self.nPeaksKey = schema.addField("deblend_nPeaks", type=np.int32,
                                         doc="Number of initial peaks in the blend. "
                                             "This includes peaks that may have been culled "
                                             "during deblending or failed to deblend")
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
        fluxCatalogs: dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are the flux-conserved catalogs with heavy footprints with
            the image data weighted by the multiband templates.
            If `self.config.conserveFlux` is `False`, then this item will be
            None
        templateCatalogs: dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are catalogs with heavy footprints that are the templates
            created by the multiband templates.
            If `self.config.saveTemplates` is `False`, then this item will be
            None
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
        fluxCatalogs : dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are the flux-conserved catalogs with heavy footprints with
            the image data weighted by the multiband templates.
            If `self.config.conserveFlux` is `False`, then this item will be
            None
        templateCatalogs : dict or None
            Keys are the names of the filters and the values are
            `lsst.afw.table.source.source.SourceCatalog`'s.
            These are catalogs with heavy footprints that are the templates
            created by the multiband templates.
            If `self.config.saveTemplates` is `False`, then this item will be
            None
        """
        import time

        filters = mExposure.filters
        self.log.info("Deblending {0} sources in {1} exposure bands".format(len(sources), len(mExposure)))

        # Create the output catalogs
        templateCatalogs = {}
        # This must be returned but is not calculated right now, setting it to
        # None to be consistent with doc string
        fluxCatalogs = None
        for f in filters:
            _catalog = afwTable.SourceCatalog(sources.table.clone())
            _catalog.extend(sources)
            templateCatalogs[f] = _catalog

        n0 = len(sources)
        nparents = 0
        for pk, src in enumerate(sources):
            foot = src.getFootprint()
            bbox = foot.getBBox()
            logger.info("id: {0}".format(src["id"]))
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
            if len(peaks) > self.config.maxNumberOfPeaks:
                src.set(self.tooManyPeaksKey, True)
                msg = 'Parent {0}: Too many peaks, using the first {1} peaks'
                self.log.trace(msg.format(int(src.getId()), self.config.maxNumberOfPeaks))

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
            # Catch the errors that we know about
            except IncompleteDataError:
                blendError = "IncompleteDataError"
            except ScarletGradientError as e:
                blendError = "ScarletGradientError"
                src.set(self.iterKey, e.iterations)
            # Catch all unknown errors to prevent the taks from crashing,
            # but store the result
            except Exception as e:
                if self.config.catchFailures:
                    blendError = "UnknownError: " + str(e)
                    import traceback
                    traceback.print_exc()
                else:
                    raise
            # Common actions for all failed blends
            if blendError is not None:
                self.log.warn("Unable to deblend source %d: %s" % (src.getId(), blendError))
                src.set(self.deblendFailedKey, True)
                src.set(self.runtimeKey, 0)
                src.set(self.deblendErrorKey, blendError)
                bbox = foot.getBBox()
                src.set(self.modelCenter, Point2D(bbox.getMinX(), bbox.getMinY()))
                # We want to store the total number of initial peaks,
                # even if some of them fail
                src.set(self.nPeaksKey, len(foot.peaks))
                continue

            # Add the merged source as a parent in the catalog for each band
            templateParents = {}
            parentId = src.getId()
            for f in filters:
                templateParents[f] = templateCatalogs[f][pk]
                templateParents[f].set(self.nPeaksKey, len(foot.peaks))
                templateParents[f].set(self.runtimeKey, runtime)
                templateParents[f].set(self.iterKey, len(blend.loss))
                # TODO: When DM-26603 is merged observation has a "log_norm"
                # property that performs the following calculation,
                # so this code block can be removed
                observation = blend.observations[0]
                _weights = observation.weights
                _images = observation.images
                log_sigma = np.zeros(_weights.shape, dtype=_weights.dtype)
                cuts = _weights > 0
                log_sigma[cuts] = np.log(1/_weights[cuts])
                log_norm = np.prod(_images.shape)/2 * np.log(2*np.pi)+np.sum(log_sigma)/2
                # end temporary code block
                logL = blend.loss[-1]-log_norm
                templateParents[f].set(self.scarletLogLKey, logL)

            # Add each source to the catalogs in each band
            templateSpans = {f: afwGeom.SpanSet() for f in filters}
            nchild = 0
            for k, source in enumerate(sources):
                # Skip any sources with no flux or that scarlet skipped because
                # it could not initialize
                if k in skipped:
                    if not self.config.propagateAllPeaks:
                        # We don't care
                        continue
                    # We need to preserve the peak: make sure we have enough
                    # info to create a minimal child src
                    msg = "Peak at {0} failed deblending.  Using minimal default info for child."
                    self.log.trace(msg.format(src.getFootprint().peaks[k]))
                    # copy the full footprint and strip out extra peaks
                    foot = afwDet.Footprint(src.getFootprint())
                    peakList = foot.getPeaks()
                    peakList.clear()
                    peakList.append(src.peaks[k])
                    zeroMimg = afwImage.MaskedImageF(foot.getBBox())
                    heavy = afwDet.makeHeavyFootprint(foot, zeroMimg)
                    models = afwDet.MultibandFootprint(mExposure.filters, [heavy]*len(mExposure.filters))
                else:
                    src.set(self.deblendSkippedKey, False)
                    models = modelToHeavy(source, filters, xy0=bbox.getMin(),
                                          observation=blend.observations[0])
                # TODO: We should eventually write the morphology and SED to
                # the catalog
                # morph = source.morphToHeavy(xy0=bbox.getMin())
                # sed = source.sed / source.sed.sum()

                flux = scarlet.measure.flux(source)
                for fidx, f in enumerate(filters):
                    if len(models[f].getPeaks()) != 1:
                        err = "Heavy footprint should have a single peak, got {0}"
                        raise ValueError(err.format(len(models[f].peaks)))
                    cat = templateCatalogs[f]
                    child = self._addChild(parentId, cat, models[f], source, converged,
                                           xy0=bbox.getMin(), flux=flux[fidx])
                    if parentId == 0:
                        child.setId(src.getId())
                        child.set(self.runtimeKey, runtime)
                    else:
                        templateSpans[f] = templateSpans[f].union(models[f].getSpans())
                nchild += 1

            # Child footprints may extend beyond the full extent of their
            # parent's which results in a failure of the replace-by-noise code
            # to reinstate these pixels to their original values.  The
            # following updates the parent footprint in-place to ensure it
            # contains the full union of itself and all of its
            # children's footprints.
            for f in filters:
                templateParents[f].set(self.nChildKey, nchild)
                templateParents[f].getFootprint().setSpans(templateSpans[f])

        K = len(list(templateCatalogs.values())[0])
        self.log.info('Deblended: of %i sources, %i were deblended, creating %i children, total %i sources'
                      % (n0, nparents, K-n0, K))
        return fluxCatalogs, templateCatalogs

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
        source.set(self.nChildKey, len(fp.getPeaks()))  # It would have this many if we deblended them all
        if self.config.notDeblendedMask:
            for mask in masks:
                mask.addMaskPlane(self.config.notDeblendedMask)
                fp.spans.setMask(mask, mask.getPlaneBitMask(self.config.notDeblendedMask))

    def _addChild(self, parentId, sources, heavy, scarlet_source, blend_converged, xy0, flux):
        """Add a child to a catalog

        This creates a new child in the source catalog,
        assigning it a parent id, adding a footprint,
        and setting all appropriate flags based on the
        deblender result.
        """
        assert len(heavy.getPeaks()) == 1
        src = sources.addNew()
        src.assign(heavy.getPeaks()[0], self.peakSchemaMapper)
        src.setParent(parentId)
        src.setFootprint(heavy)
        src.set(self.psfKey, False)
        src.set(self.runtimeKey, 0)
        src.set(self.blendConvergenceFailedFlagKey, not blend_converged)
        if isinstance(scarlet_source, ExtendedSource):
            cy, cx = scarlet_source.pixel_center
            morph = scarlet_source.morph
        elif isinstance(scarlet_source, MultiComponentSource):
            cy, cx = scarlet_source.components[0].pixel_center
            morph = scarlet_source.components[0].morph
        elif isinstance(scarlet_source, PointSource):
            cy, cx = scarlet_source.parameters[1]
            morph = scarlet_source.morph
        else:
            msg = "Did not recognize source type of `{0}`, could not write coordinates or center flux. "
            msg += "Add `{0}` to meas_extensions_scarlet to properly persist this information."
            logger.warning(msg.format(type(scarlet_source)))
            return src
        xmin, ymin = xy0
        src.set(self.modelCenter, Point2D(cx+xmin, cy+ymin))
        cy = np.max([np.min([int(np.round(cy)), morph.shape[0]-1]), 0])
        cx = np.max([np.min([int(np.round(cx)), morph.shape[1]-1]), 0])
        src.set(self.modelCenterFlux, morph[cy, cx])
        src.set(self.modelTypeKey, scarlet_source.__class__.__name__)
        src.set(self.edgeFluxFlagKey, scarlet_source.isEdge)
        # Include the source flux in the model space in the catalog.
        # This uses the narrower model PSF, which ensures that all sources
        # not located on an edge have all of their flux included in the
        # measurement.
        src.set(self.scarletFluxKey, flux)
        return src
