from contextlib import contextmanager
from typing import Sequence

import lsst.geom as geom
import lsst.scarlet.lite as scl
import numpy as np
from scipy.signal import convolve
from lsst.afw.detection import Footprint as afwFootprint
from lsst.afw.detection import HeavyFootprintF, PeakCatalog, makeHeavyFootprint
from lsst.afw.detection.multiband import MultibandFootprint
from lsst.afw.geom import SpanSet
from lsst.afw.image import Image as afwImage
from lsst.afw.image import (
    IncompleteDataError,
    Mask,
    MaskedImage,
    MultibandExposure,
    MultibandImage,
)
from lsst.afw.table import SourceCatalog
from lsst.scarlet.lite.detect_pybind11 import Peak

defaultBadPixelMasks = ["BAD", "NO_DATA", "SAT", "SUSPECT", "EDGE"]


def footprintsToNumpy(
    catalog: SourceCatalog,
    shape: tuple[int, int],
    xy0: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert all of the footprints in a catalog into a boolean array.

    Parameters
    ----------
    catalog:
        The source catalog containing the footprints.
        This is typically a mergeDet catalog, or a full source catalog
        with the parents removed.
    shape:
        The final shape of the output array.
    xy0:
        The lower-left corner of the array that will contain the spans.

    Returns
    -------
    result:
        The array with pixels contained in `spans` marked as `True`.
    """
    if xy0 is None:
        offset = (0, 0)
    else:
        offset = (-xy0[0], -xy0[1])

    result = np.zeros(shape, dtype=bool)
    for src in catalog:
        spans = src.getFootprint().spans
        yidx, xidx = spans.shiftedBy(*offset).indices()
        result[yidx, xidx] = 1
    return result


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


def afwFootprintToScarlet(footprint: afwFootprint, copyPeaks: bool = True):
    """Convert an afw Footprint into a scarlet lite Footprint.

    Parameters
    ----------
    footprint:
        The afw Footprint to convert.
    copyPeaks:
        Whether or not to copy the peaks from the afw Footprint.

    Returns
    -------
    scarletFootprint:
        The converted scarlet Footprint.
    """
    data = footprint.spans.asArray()
    afwBox = footprint.getBBox()
    bbox = bboxToScarletBox(afwBox)
    peaks = []
    if copyPeaks:
        for peak in footprint.peaks:
            newPeak = Peak(peak.getIy(), peak.getIx(), peak.getPeakValue())
            peaks.append(newPeak)
    bounds = scl.detect.bbox_to_bounds(bbox)
    return scl.detect.Footprint(data, peaks, bounds)


def scarletFootprintToAfw(footprint: scl.detect.Footprint, copyPeaks: bool = True) -> afwFootprint:
    """Convert a scarlet lite Footprint into an afw Footprint.

    Parameters
    ----------
    footprint:
        The scarlet Footprint to convert.
    copyPeaks:
        Whether or not to copy the peaks from the scarlet Footprint.

    Returns
    -------
    newFootprint:
        The converted afw Footprint.
    """
    xy0 = geom.Point2I(footprint.bbox.origin[1], footprint.bbox.origin[0])
    data = Mask(footprint.data.astype(np.int32), xy0=xy0)
    spans = SpanSet.fromMask(data)
    newFootprint = afwFootprint(spans)

    if copyPeaks:
        for peak in footprint.peaks:
            newFootprint.addPeak(peak.x, peak.y, peak.flux)
    return newFootprint


def getFootprintIntersection(
    footprint1: afwFootprint,
    footprint2: afwFootprint,
    copyMethod: str | None = 'left'
) -> afwFootprint:
    """Calculate the intersection of two Footprints.

    Parameters
    ----------
    footprint1:
        The first afw Footprint.
    footprint2:
        The second afw Footprint.
    copyMethod:
        The method to use when copying peaks into the new Footprint
        that are contained in the intersection.
        If ``None`` then no peaks are copied, 'left' copies the
        peaks from `footprint`, and 'right' copies the peaks from
        `footprint2`.

    Returns
    -------
    result:
        The Footprint containing the intersection of the two footprints.
    """
    if copyMethod is not None and copyMethod not in ['left', 'right']:
        raise ValueError(f'copyMethod should be "left", "right", or None, got {copyMethod}')

    # Create the intersecting footprint
    spans = footprint1.spans.intersect(footprint2.spans)
    result = afwFootprint(spans)

    peaks = []
    if copyMethod is not None:
        # Copy peaks into the new Footprint
        if copyMethod == 'left':
            peaks = footprint1.peaks
        else:
            peaks = footprint2.peaks

        for peak in peaks:
            if spans.contains(geom.Point2I(peak.getIx(), peak.getIy())):
                newPeak = result.addPeak(peak.getIx(), peak.getIy(), peak.getPeakValue())
                # Ensure the peak has the same ID as the original peak
                newPeak.setId(peak.getId())
    return result


def multiband_convolve(images: np.ndarray, psfs: np.ndarray) -> np.ndarray:
    """Convolve a multi-band images with the PSF in each band.

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
    for bidx, image in enumerate(images):
        result[bidx] = convolve(image, psfs[bidx], mode="same")
    return result


def computePsfKernelImage(mExposure, psfCenter):
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
        bands = psfModels.filters
        mExposure = mExposure[bands,]
        if len(bands) == 1:
            # Only a single band generated a PSF, so the MultibandExposure
            # became a single band ExposureF.
            # Convert the result back into a MultibandExposure.
            mExposure = MultibandExposure.fromExposures(bands, [mExposure])
    return psfModels.array, mExposure


def buildObservation(
    modelPsf: np.ndarray,
    psfCenter: tuple[int, int] | geom.Point2I | geom.Point2D,
    mExposure: MultibandExposure,
    badPixelMasks: list[str] | None = None,
    footprint: afwFootprint = None,
    useWeights: bool = True,
    convolutionType: str = "real",
) -> scl.Observation:
    """Generate an Observation from a set of arguments.

    Make the generation and reconstruction of a scarlet model consistent
    by building an `Observation` from a set of arguments.

    Parameters
    ----------
    modelPsf:
        The 2D model of the PSF in the partially deconvolved space.
    psfCenter:
        The location `(x, y)` used as the center of the PSF.
    mExposure:
        The multi-band exposure that the model represents.
        If `mExposure` is `None` then no image, variance, or weights are
        attached to the observation.
    footprint:
        The footprint that is being fit.
        If `footprint` is `None` then the weights are not updated to mask
        out pixels not contained in the footprint.
    badPixelMasks:
        The keys from the bit mask plane used to mask out pixels
        during the fit.
        If `badPixelMasks` is `None` then the default values from
        `ScarletDeblendConfig.badMask` are used.
    useWeights:
        Whether or not fitting should use inverse variance weights to
        calculate the log-likelihood.
    convolutionType:
        The type of convolution to use (either "real" or "fft").
        When reconstructing an image it is advised to use "real" to avoid
        polluting the footprint with artifacts from the fft.

    Returns
    -------
    observation:
        The observation constructed from the input parameters.
    """
    # Initialize the observed PSFs
    if not isinstance(psfCenter, geom.Point2D):
        psfCenter = geom.Point2D(*psfCenter)
    psfModels, mExposure = computePsfKernelImage(mExposure, psfCenter)

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
        bands=mExposure.filters,
        bbox=bboxToScarletBox(mExposure.getBBox()),
    )


def scarletModelToHeavy(
    source: scl.Source,
    blend: scl.Blend,
    useFlux=False,
) -> HeavyFootprintF | MultibandFootprint:
    """Convert a scarlet_lite model to a `HeavyFootprintF`
    or `MultibandFootprint`.

    Parameters
    ----------
    source:
        The source to convert to a `HeavyFootprint`.
    blend:
        The `Blend` object that contains information about
        the observation, PSF, etc, used to convolve the
        scarlet model to the observed seeing in each band.
    useFlux:
        Whether or not to re-distribute the flux from the image
        to conserve flux.

    Returns
    -------
    heavy:
        The footprint (possibly multiband) containing the model for the source.
    """
    # We want to convolve the model with the observed PSF,
    # which means we need to grow the model box by the PSF to
    # account for all of the flux after convolution.

    # Get the PSF size and radii to grow the box
    py, px = blend.observation.psfs.shape[1:]
    dh = py // 2
    dw = px // 2

    if useFlux:
        bbox = source.flux_weighted_image.bbox
    else:
        bbox = source.bbox.grow((dh, dw))
    # Only use the portion of the convolved model that fits in the image
    overlap = bbox & blend.observation.bbox
    # Load the full multiband model in the larger box
    if useFlux:
        # The flux weighted model is already convolved, so we just load it
        model = source.get_model(use_flux=True).project(bbox=overlap)
    else:
        model = source.get_model().project(bbox=overlap)
        # Convolve the model with the PSF in each band
        # Always use a real space convolution to limit artifacts
        model = blend.observation.convolve(model, mode="real")

    # Update xy0 with the origin of the sources box
    xy0 = geom.Point2I(model.yx0[-1], model.yx0[-2])
    # Create the spans for the footprint
    valid = np.max(model.data, axis=0) != 0
    valid = Mask(valid.astype(np.int32), xy0=xy0)
    spans = SpanSet.fromMask(valid)

    # Add the location of the source to the peak catalog
    peakCat = PeakCatalog(source.detectedPeak.table)
    peakCat.append(source.detectedPeak)

    # Create the MultibandHeavyFootprint
    foot = afwFootprint(spans)
    foot.setPeakCatalog(peakCat)
    if model.n_bands == 1:
        image = afwImage(
            array=model.data[0], xy0=valid.getBBox().getMin(), dtype=model.dtype
        )
        maskedImage = MaskedImage(image, dtype=model.dtype)
        heavy = makeHeavyFootprint(foot, maskedImage)
    else:
        model = MultibandImage(blend.bands, model.data, valid.getBBox())
        heavy = MultibandFootprint.fromImages(blend.bands, model, footprint=foot)
    return heavy


def scarletFootprintsToPeakCatalog(
    footprints: Sequence[scl.detect.Footprint],
) -> PeakCatalog:
    """Create a PeakCatalog from a list of scarlet footprints.

    This creates a dummy Footprint to add the peaks to,
    then extracts the peaks from the Footprint.
    It seems like there should be a better way to do this but
    I couldn't find one.

    Parameters
    ----------
    footprints:
        A list of scarlet footprints.

    Returns
    -------
    peaks:
        A PeakCatalog containing all of the peaks in the footprints.
    """
    tempFootprint = afwFootprint()
    for footprint in footprints:
        for peak in footprint.peaks:
            tempFootprint.addPeak(peak.x, peak.y, peak.flux)
    return tempFootprint.peaks


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