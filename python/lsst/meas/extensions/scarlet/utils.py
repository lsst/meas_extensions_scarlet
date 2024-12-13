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
from lsst.geom import Box2I, Point2D, Point2I

defaultBadPixelMasks = ["BAD", "NO_DATA", "SAT", "SUSPECT", "EDGE"]


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

    Returns
    -------
    psfModels : `np.ndarray`
        The multiband PSF image
    mExposure : `MultibandExposure`
        The exposure, updated to only use bands that
        successfully generated a PSF image.
    """
    if not isinstance(psfCenter, geom.Point2D):
        psfCenter = geom.Point2D(*psfCenter)

    try:
        psfModels = mExposure.computePsfKernelImage(psfCenter)
    except IncompleteDataError as e:
        psfModels = e.partialPsf
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
    band: str,
    psfCenter: Point2D,
) -> tuple[np.ndarray, Point2I, float]:
    """Create a PSF image at the neareset valid location

    Sometimes not all locations in an image can generate a PSF image so the
    source catalog is used to find the nearest valid location.

    Parameters
    ----------
    calexp :
        The exposure.
    catalog :
        The catalog.
    band :
        The band.
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
    sources = catalog[catalog[f'merge_footprint_{band}']]

    # Get the peaks of all of the sources
    x = []
    y = []
    for src in sources:
        for peak in src.getFootprint().peaks:
            if peak[f'merge_peak_{band}']:
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
    newLocation = Point2I(x[ref_index], y[ref_index])
    diff = np.sqrt(diff_x[ref_index]**2 + diff_y[ref_index]**2)

    return psf, newLocation, diff


def computeNearestMultiBandPsf(
    mExposure: MultibandExposure,
    psfCenter: tuple[int, int] | geom.Point2I | geom.Point2D,
    catalog: SourceCatalog,
) -> tuple[np.ndarray, MultibandExposure]:
    """Compute the image in each band at the location nearest to the PSF Center

    Parameters
    ----------
    mExposure :
        The multi-band exposure.
    psfCenter :
        The location `(x, y)` used as the center of the PSF.
    catalog :
        The source catalog.
    """
    psfs = {}
    incomplete = False
    for band in mExposure.filters:
        psf, psfCenter, diff = computeNearestPsf(
            mExposure[band,],
            psfCenter,
            catalog,
            band,
        )
        if psf is None:
            incomplete = True
        else:
            psfs[band] = psf

    left = np.min([psf.getBBox().getMinX() for psf in psfs.values()])
    bottom = np.min([psf.getBBox().getMinY() for psf in psfs.values()])
    right = np.max([psf.getBBox().getMaxX() for psf in psfs.values()])
    top = np.max([psf.getBBox().getMaxY() for psf in psfs.values()])
    bbox = Box2I(Point2I(left, bottom), Point2I(right, top))

    psf_images = [projectImage(psf, bbox) for psf in psfs.values()]

    mPsf = MultibandImage.fromImages(list(psfs.keys()), psf_images)

    if incomplete:
        bands = mPsf.filters
        mExposure = mExposure[bands,]

        if len(bands) == 1:
            # Only a single band generated a PSF, so the MultibandExposure
            # became a single band ExposureF.
            # Convert the result back into a MultibandExposure.
            mExposure = MultibandExposure.fromExposures(bands, [mExposure])

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

    Returns
    -------
    observation:
        The observation constructed from the input parameters.
    """
    # Initialize the observed PSFs
    if not isinstance(psfCenter, geom.Point2D):
        psfCenter = geom.Point2D(*psfCenter)
    if catalog is None:
        psfModels, mExposure = computePsfKernelImage(mExposure, psfCenter)
    else:
        psfModels, mExposure = computeNearestMultiBandPsf(mExposure, psfCenter, catalog)

    # Use the inverse variance as the weights
    if useWeights:
        weights = 1 / mExposure.variance.array
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

    return scl.Observation(
        images=mExposure.image.array,
        variance=mExposure.variance.array,
        weights=weights,
        psfs=psfModels,
        model_psf=modelPsf[None, :, :],
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
