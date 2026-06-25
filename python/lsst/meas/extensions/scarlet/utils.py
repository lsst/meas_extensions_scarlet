import logging
import warnings

import lsst.geom as geom
import lsst.scarlet.lite as scl
import numpy as np
from scipy.signal import convolve
from lsst.afw.detection import InvalidPsfError, Footprint as afwFootprint
from lsst.afw.image import (
    IncompleteDataError,
    MultibandExposure,
    MultibandImage,
    Exposure,
)
from lsst.afw.image.utils import projectImage
from lsst.afw.table import SourceCatalog
from lsst.cell_coadds import StitchedPsf
from lsst.geom import Box2I, Point2D, Point2I
from lsst.pipe.base import NoWorkFound

from .stitched_psf import ScarletStitchedPsf

logger = logging.getLogger(__name__)

__all__ = [
    "defaultBadPixelMasks",
    "scarletBoxToBBox",
    "bboxToScarletBox",
    "nonzeroBandSupport",
    "multiband_convolve",
    "computePsfKernelImage",
    "computeNearestPsf",
    "computeNearestPsfMultiBand",
    "buildObservation",
    "calcChi2",
]

defaultBadPixelMasks = ["BAD", "NO_DATA", "SAT", "SUSPECT", "EDGE", "INEXACT_PSF", "REJECTED", "INTRP"]


def scarletBoxToBBox(box: scl.Box, xy0: geom.Point2I = geom.Point2I()) -> geom.Box2I:
    """Convert a scarlet_lite Box into a Box2I.

    Parameters
    ----------
    box:
        The scarlet bounding box to convert.
    xy0:
        An additional offset to add to the scarlet box.
        This is common since scarlet sources have an origin of
        `(0,0)` at the lower left corner of the blend while
        the blend itself is likely to have an offset in the
        `Exposure`.

    Returns
    -------
    bbox:
        The converted bounding box.
    """
    xy0 = geom.Point2I(box.origin[-1] + xy0.x, box.origin[-2] + xy0.y)
    extent = geom.Extent2I(box.shape[-1], box.shape[-2])
    return geom.Box2I(xy0, extent)


def bboxToScarletBox(bbox: geom.Box2I, xy0: geom.Point2I = geom.Point2I()) -> scl.Box:
    """Convert a Box2I into a scarlet_lite Box.

    Parameters
    ----------
    bbox:
        The Box2I to convert into a scarlet `Box`.
    xy0:
        An overall offset to subtract from the `Box2I`.
        This is common in blends, where `xy0` is the minimum pixel
        location of the blend and `bbox` is the box containing
        a source in the blend.

    Returns
    -------
    box:
        A scarlet `Box` that is more useful for slicing image data
        as a numpy array.
    """
    origin = (bbox.getMinY() - xy0.y, bbox.getMinX() - xy0.x)
    return scl.Box((bbox.getHeight(), bbox.getWidth()), origin)


def nonzeroBandSupport(data: np.ndarray) -> np.ndarray:
    """Return the per-pixel support of a multi-band model.

    A pixel is in the support whenever any band's value is non-zero.
    This is the canonical "spatial extent of a model across bands"
    test; the alternative idioms ``data > 0`` and
    ``np.max(data, axis=0) != 0`` either exclude negative-valued
    pixels outright or exclude pixels whose largest band value is
    exactly zero, both of which under-count the true spatial extent.

    Parameters
    ----------
    data :
        A ``(bands, height, width)`` array of model values.

    Returns
    -------
    support :
        A ``(height, width)`` boolean mask, ``True`` at pixels where
        at least one band is non-zero.
    """
    return np.any(data != 0, axis=0)


def multiband_convolve(images: np.ndarray, psfs: np.ndarray) -> np.ndarray:
    """Convolve a multi-band image with the PSF in each band.

    `images` and `psfs` should have dimensions `(bands, height, width)`.

    Parameters
    ----------
    images :
        The multi-band images to convolve.
    psfs :
        The PSF for each band.

    Returns
    -------
    result :
        The convolved images.
    """
    result = np.zeros(images.shape, dtype=images.dtype)
    for bidx, (image, psf) in enumerate(zip(images, psfs, strict=True)):
        result[bidx] = convolve(image, psf, mode="same")
    return result


def computePsfKernelImage(mExposure, psfCenter, catalog=None):
    """Compute the PSF kernel image and update the multiband exposure
    if not all of the PSF images could be computed.

    Parameters
    ----------
    psfCenter : `tuple` or `Point2I` or `Point2D`
        The location `(x, y)` used as the center of the PSF.
    catalog :
        Deprecated and ignored. Retained for signature stability; will
        be removed after v31. Passing a non-``None`` value emits a
        ``FutureWarning``. For nearest-PSF fallback at a different
        location, call ``computeNearestPsfMultiBand`` instead.

    Returns
    -------
    psfModels : `np.ndarray`
        The multiband PSF image
    mExposure : `MultibandExposure`
        The exposure, updated to only use bands that
        successfully generated a PSF image.
    """
    if catalog is not None:
        warnings.warn(
            "The `catalog` parameter to `computePsfKernelImage` is "
            "deprecated and ignored; it will be removed after v31. "
            "For nearest-PSF fallback, use `computeNearestPsfMultiBand`.",
            FutureWarning, stacklevel=2,
        )
    if not isinstance(psfCenter, geom.Point2D):
        psfCenter = geom.Point2D(*psfCenter)

    try:
        psfModels = mExposure.computePsfKernelImage(psfCenter)
    except IncompleteDataError as e:
        psfModels = e.partialPsf
        if psfModels is None:
            return None, None
        # Use only the bands that successfully generated a PSF image.
        bands = psfModels.bands
        mExposure = mExposure[bands,]
        if len(bands) == 1:
            # Only a single band generated a PSF, so the MultibandExposure
            # became a single band ExposureF.
            # Convert the result back into a MultibandExposure.
            mExposure = MultibandExposure.fromExposures(bands, [mExposure])
    return psfModels.array, mExposure


def computeNearestPsf(
    calexp: Exposure,
    catalog: SourceCatalog,
    band: str | None = None,
    psfCenter: Point2D | None = None,
) -> tuple[np.ndarray, Point2D, float] | tuple[None, None, None]:
    """Create a PSF image at the nearest valid location

    Sometimes not all locations in an image can generate a PSF image so the
    source catalog is used to find the nearest valid location.

    Parameters
    ----------
    calexp :
        The exposure.
    catalog :
        The catalog.
    band :
        The band of the exposure used to filter the catalog by only
        selecting sources that have a
        If band is ``None`` then the full catalog is used.
    psfCenter :
        The location of the PSF image.
        If no location is provided, the center of the exposure is used.

    Returns
    -------
    psf :
        The PSF image.
    location :
        The location of the PSF image.
    diff :
        The difference between the requested location and the
        nearest valid location.
    """
    if psfCenter is None:
        psfCenter = calexp.getBBox().getCenter()

    if not isinstance(psfCenter, geom.Point2D):
        psfCenter = geom.Point2D(*psfCenter)

    try:
        psf = calexp.getPsf().computeKernelImage(psfCenter)
        return psf, psfCenter, 0
    except InvalidPsfError:
        pass

    xc, yc = psfCenter

    # Only select records that have detections in this band
    if band is not None:
        sources = catalog[catalog[f'merge_footprint_{band}']]
    else:
        sources = catalog

    # Get the peaks of all of the sources
    x = []
    y = []
    for src in sources:
        for peak in src.getFootprint().peaks:
            if band is None or peak[f'merge_peak_{band}']:
                x.append(peak['i_x'])
                y.append(peak['i_y'])
    x = np.array(x)
    y = np.array(y)

    # Sort the peaks based on their distance to the location
    diff_x = x - xc
    diff_y = y - yc
    sorted_indices = np.argsort(diff_x**2 + diff_y**2)

    # Iterate over sources until a location is found that can generate a PSF
    psf = None
    for ref_index in sorted_indices:
        try:
            psf = calexp.getPsf().computeKernelImage(Point2D(x[ref_index], y[ref_index]))
            break
        except InvalidPsfError:
            pass
    if psf is None:
        return None, None, None
    newLocation = Point2D(x[ref_index], y[ref_index])
    diff = np.sqrt(diff_x[ref_index]**2 + diff_y[ref_index]**2)

    return psf, newLocation, diff


def _sortedCatalogPositions(
    catalog: SourceCatalog | None,
    psfCenter: Point2D,
) -> list[Point2D]:
    """Catalog peak positions, sorted by distance from ``psfCenter``."""
    if catalog is None:
        return []
    xs: list[float] = []
    ys: list[float] = []
    for src in catalog:
        for peak in src.getFootprint().peaks:
            xs.append(peak["i_x"])
            ys.append(peak["i_y"])
    if not xs:
        return []
    xs_a = np.asarray(xs)
    ys_a = np.asarray(ys)
    xc, yc = psfCenter
    order = np.argsort((xs_a - xc) ** 2 + (ys_a - yc) ** 2)
    return [Point2D(float(xs_a[i]), float(ys_a[i])) for i in order]


def computeNearestPsfMultiBand(
    mExposure: MultibandExposure,
    psfCenter: tuple[int, int] | geom.Point2I | geom.Point2D,
    catalog: SourceCatalog | None,
) -> tuple[MultibandImage | None, MultibandExposure | None]:
    """Compute a multiband PSF kernel image at or near the requested center.

    The PSF kernel is computed at ``psfCenter`` in every band where the
    PSF model is valid there — this is the hot path and the only work
    done when no fallback is needed. For any band whose PSF is invalid
    at ``psfCenter``, the function walks ``catalog`` peak positions
    sorted by distance from the requested center and accepts the first
    position that works in every failing band as a common fallback.
    If that fallback location also works in the bands that already
    succeeded at the center, the function "upgrades" by sampling every
    band at that one location, so the multiband PSF is genuinely at a
    single sky point; otherwise the successful bands keep their center
    PSF and only the failing bands use the common fallback, and a
    warning is logged. When no single fallback works for every failing
    band, each failing band falls back to its own nearest valid
    catalog position (per-band fallback), and a warning is logged.
    Bands with no valid PSF anywhere are dropped from the returned
    multiband exposure, matching the historical incomplete-PSF
    behavior.

    PSF computations are not duplicated: each ``(band, position)`` pair
    is evaluated at most once, and at function return the kept PSFs are
    held one per band.

    Parameters
    ----------
    mExposure :
        The multi-band exposure.
    psfCenter :
        The location ``(x, y)`` used as the center of the PSF.
    catalog :
        Source catalog whose peak positions are candidate fallback
        locations. If `None`, no fallback search is performed and a
        band whose PSF is invalid at ``psfCenter`` is dropped.

    Returns
    -------
    mPsf :
        The multiband PSF kernel image, or `None` if no band produced a
        valid PSF.
    mExposure :
        The input exposure restricted to bands that produced a valid
        PSF.
    """
    if not isinstance(psfCenter, Point2D):
        psfCenter = Point2D(*psfCenter)

    bands = tuple(mExposure.bands)

    # Stage 1: try every band at the requested center.
    psfs: dict = {}
    locations: dict[str, Point2D] = {}
    failingBands: list[str] = []
    for band in bands:
        try:
            psfs[band] = mExposure[band,].getPsf().computeKernelImage(psfCenter)
            locations[band] = psfCenter
        except InvalidPsfError:
            failingBands.append(band)

    if failingBands:
        # Stage 2: walk catalog candidates sorted by distance from the
        # requested center and accept the first that works for every
        # failing band. The bands that already succeeded at the center
        # are not part of the search — their PSFs are already at a
        # strictly better position than any fallback could provide.
        candidates = _sortedCatalogPositions(catalog, psfCenter)
        commonLocation: Point2D | None = None
        commonPsfs: dict = {}
        # The closest valid PSF found so far for each failing band,
        # populated as we walk the candidate list. If no single
        # candidate works for every failing band, these are the per-
        # band fallbacks. Each ``(band, position)`` is evaluated at
        # most once across the walk.
        closestPsfs: dict = {}
        closestLocations: dict = {}
        for pos in candidates:
            tentative: dict = {}
            allOk = True
            for band in failingBands:
                try:
                    psf = (
                        mExposure[band,].getPsf().computeKernelImage(pos)
                    )
                except InvalidPsfError:
                    # Mark the candidate as not common, but keep trying
                    # the remaining bands at this candidate so a band
                    # that *is* valid here still gets the chance to be
                    # cached at its true closest position.
                    allOk = False
                    continue
                tentative[band] = psf
                if band not in closestPsfs:
                    closestPsfs[band] = psf
                    closestLocations[band] = pos
            if allOk:
                commonLocation = pos
                commonPsfs = tentative
                break

        if commonLocation is not None:
            # Stage 2 upgrade: if the common fallback is also valid in
            # the bands that succeeded at the center, switch every band
            # to it so the multiband PSF is sampled at a single sky
            # location. Otherwise, keep center PSFs for the successful
            # bands and use the fallback only for the failing ones.
            upgradePsfs: dict = {}
            canUpgrade = True
            for band in psfs:
                try:
                    upgradePsfs[band] = (
                        mExposure[band,].getPsf().computeKernelImage(commonLocation)
                    )
                except InvalidPsfError:
                    canUpgrade = False
                    break

            if canUpgrade:
                psfs = {**upgradePsfs, **commonPsfs}
            else:
                logger.warning(
                    "Multiband PSF falls back at two locations: bands %s at "
                    "the requested center %s; bands %s at %s.",
                    list(psfs), psfCenter, failingBands, commonLocation,
                )
                for band in failingBands:
                    psfs[band] = commonPsfs[band]
        else:
            # No common fallback location: use the per-band closest
            # PSFs that the candidate walk already collected. Bands
            # without any valid PSF in the walk are absent from
            # closestPsfs and will be dropped below.
            psfs.update(closestPsfs)
            locations.update(closestLocations)
            if any(b in psfs for b in failingBands):
                logger.warning(
                    "Multiband PSF: no single fallback location works for "
                    "every band; per-band fallback locations %s.",
                    {b: locations[b] for b in bands if b in psfs},
                )

    if len(psfs) == 0:
        return None, None

    # Project each kept band's PSF onto the union bbox so the multiband
    # image has consistent shape across bands.
    left = np.min([psf.getBBox().getMinX() for psf in psfs.values()])
    bottom = np.min([psf.getBBox().getMinY() for psf in psfs.values()])
    right = np.max([psf.getBBox().getMaxX() for psf in psfs.values()])
    top = np.max([psf.getBBox().getMaxY() for psf in psfs.values()])
    bbox = Box2I(Point2I(left, bottom), Point2I(right, top))

    # Ensure that the returned multiband PSF and exposure only contain
    # the bands for which a valid PSF was found, in the same order as the
    # input exposure's bands.
    bandsKept = tuple(b for b in bands if b in psfs)
    psf_images = [projectImage(psfs[b], bbox) for b in bandsKept]
    mPsf = MultibandImage.fromImages(bandsKept, psf_images)

    if len(bandsKept) < len(bands):
        mExposure = mExposure[bandsKept,]
        if len(bandsKept) == 1:
            # ``mExposure[(band,),]`` for a single band returns an
            # ExposureF rather than a MultibandExposure; wrap it back.
            mExposure = MultibandExposure.fromExposures(bandsKept, [mExposure])

    return mPsf.array, mExposure


def buildObservation(
    modelPsf: np.ndarray,
    psfCenter: tuple[int, int] | geom.Point2I | geom.Point2D,
    mExposure: MultibandExposure,
    badPixelMasks: list[str] | None = None,
    footprint: afwFootprint = None,
    useWeights: bool = True,
    convolutionType: str = "real",
    catalog: SourceCatalog | None = None,
    useStitchedPsf: bool = True,
) -> scl.Observation:
    """Generate an Observation from a set of arguments.

    Make the generation and reconstruction of a scarlet model consistent
    by building an `Observation` from a set of arguments.

    Parameters
    ----------
    modelPsf :
        The 2D model of the PSF in the partially deconvolved space.
    psfCenter :
        The location `(x, y)` used as the center of the PSF.
    mExposure :
        The multi-band exposure that the model represents.
        If `mExposure` is `None` then no image, variance, or weights are
        attached to the observation.
    footprint :
        The footprint that is being fit.
        If `footprint` is `None` then the weights are not updated to mask
        out pixels not contained in the footprint.
    badPixelMasks :
        The keys from the bit mask plane used to mask out pixels
        during the fit.
        If `badPixelMasks` is `None` then the default values from
        `ScarletDeblendConfig.badMask` are used.
    useWeights :
        Whether or not fitting should use inverse variance weights to
        calculate the log-likelihood.
    convolutionType :
        The type of convolution to use (either "real" or "fft").
        When reconstructing an image it is advised to use "real" to avoid
        polluting the footprint with artifacts from the fft.
    catalog :
        A source catalog to use for PSFs that cannot be determined at
        the center of the image.
    useStitchedPsf :
        When the per-band coadd PSFs are cell-coadd ``StitchedPsf`` objects,
        whether to build a spatially-varying `ScarletStitchedPsf` (more
        accurate, but far slower since each cell is convolved with its own
        FFT). If `False`, a single PSF kernel at ``psfCenter`` is used (an
        `~lsst.scarlet.lite.ImagePsf`) even for cell coadds. Ignored for
        non-cell coadds, which are always flat.

    Returns
    -------
    observation:
        The observation constructed from the input parameters.
    """
    # Initialize the observed PSFs
    if not isinstance(psfCenter, geom.Point2D):
        psfCenter = geom.Point2D(*psfCenter)

    bandPsfs = {band: mExposure[band,].getPsf() for band in mExposure.bands}
    if useStitchedPsf and all(isinstance(psf, StitchedPsf) for psf in bandPsfs.values()):
        # Cell-based coadd: the PSF is genuinely discontinuous across cells,
        # so build a spatially-varying ScarletStitchedPsf over the cell grid
        # rather than one kernel image per band. A StitchedPsf is valid
        # everywhere within the coadd, so no band is dropped and the
        # nearest-PSF fallback used by the flat path is unnecessary here.
        observedPsf: scl.Psf = ScarletStitchedPsf.from_stitched_psf(bandPsfs)
    else:
        if catalog is None:
            psfModels, mExposure = computePsfKernelImage(mExposure, psfCenter)
        else:
            psfModels, mExposure = computeNearestPsfMultiBand(mExposure, psfCenter, catalog)

        if psfModels is None:
            raise NoWorkFound("No valid PSF could be obtained for building the observation")
        observedPsf = scl.ImagePsf(psfModels, bands=tuple(mExposure.bands))

    # Use the inverse variance as the weights
    if useWeights:
        # Zero/NaN variance produces inf/NaN weights here; the next line
        # zeros them deliberately. Silence the spurious RuntimeWarnings
        # the division would otherwise emit on those pixels.
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = 1 / mExposure.variance.array
        weights[~np.isfinite(weights)] = 0
    else:
        weights = np.ones_like(mExposure.image.array)

    # Mask out bad pixels
    if badPixelMasks is None:
        badPixelMasks = defaultBadPixelMasks
    badPixels = mExposure.mask.getPlaneBitMask(badPixelMasks)
    mask = mExposure.mask.array & badPixels
    weights[mask > 0] = 0

    if footprint is not None:
        # Mask out the pixels outside the footprint
        weights *= footprint.spans.asArray()

    # Mask out non-finite pixels
    image = mExposure.image.array.copy()
    weights[~np.isfinite(image)] = 0
    image[~np.isfinite(image)] = 0

    return scl.Observation(
        images=image,
        variance=mExposure.variance.array,
        weights=weights,
        psf=observedPsf,
        model_psf=scl.ImagePsf(modelPsf[None, :, :]),
        convolution_mode=convolutionType,
        bands=mExposure.bands,
        bbox=bboxToScarletBox(mExposure.getBBox()),
    )


def calcChi2(
    model: scl.Image,
    observation: scl.Observation,
    footprint: np.ndarray | None = None,
    doConvolve: bool = True,
) -> scl.Image:
    """Calculate the chi2 image for a model.

    Parameters
    ----------
    model :
        The model used to calculate the chi2.
    observation :
        The observation used to calculate the chi2.
    footprint :
        The footprint to use when calculating the chi2.
        If `footprint` is `None` then the footprint is calculated
        to be the pixels where the model is greater than 0.
    doConvolve :
        Whether or not to convolve the model with the PSF.

    Returns
    -------
    chi2 :
        The chi2/pixel image for the model.
    """
    if doConvolve:
        model = observation.convolve(model)
    if footprint is None:
        footprint = model.data > 0
    bbox = model.bbox
    nBands = len(observation.images.bands)
    residual = (observation.images[:, bbox].data - model.data) * footprint
    cuts = observation.variance[:, bbox].data != 0
    chi2Data = np.zeros(residual.shape, dtype=residual.dtype)
    chi2Data[cuts] = residual[cuts]**2 / observation.variance[:, bbox].data[cuts] / nBands
    chi2 = scl.Image(
        chi2Data,
        bands=model.bands,
        yx0=model.yx0,
    )
    return chi2
