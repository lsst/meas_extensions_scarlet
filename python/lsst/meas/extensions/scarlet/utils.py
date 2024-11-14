from typing import Sequence

import lsst.geom as geom
import lsst.scarlet.lite as scl
import numpy as np
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

defaultBadPixelMasks = ["BAD", "CR", "NO_DATA", "SAT", "SUSPECT", "EDGE"]


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


def removeHighFrequencySignal(
    image: np.ndarray,
    waveletGeneration: int = 2,
) -> np.ndarray:
    """Remove high frequency signal from an image using a starlet transform.

    Parameters
    ----------
    image:
        The image to remove high frequency signal from.
    waveletScales:
        The number of wavelet scales to use.
    waveletGeneration:
        The generation of the starlet transform to use.

    Returns
    -------
    lowFreqImage:
        The image with high frequency signal removed.
    """
    wavelets = scl.wavelet.multiband_starlet_transform(
        image,
        scales=1,
        generation=waveletGeneration,
    )
    wavelets[0] = 0
    return scl.wavelet.multiband_starlet_reconstruction(
        wavelets,
        generation=waveletGeneration,
    )


def buildObservation(
    modelPsf: np.ndarray,
    psfCenter: tuple[int, int] | geom.Point2I | geom.Point2D,
    mExposure: MultibandExposure,
    badPixelMasks: list[str] | None = None,
    footprint: afwFootprint = None,
    useWeights: bool = True,
    convolutionType: str = "real",
    useWavelets: bool = False,
    waveletGeneration: int = 2,
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
    useWavelets:
        Whether or to remove high frequency signal from the image.
    waveletGeneration:
        The generation of the starlet transform to use.

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

    if useWavelets:
        # Remove high frequency signal from the image
        # (this is sometimes used for detection)
        images = removeHighFrequencySignal(mExposure.image.array, waveletGeneration)
        psfModels = removeHighFrequencySignal(psfModels, waveletGeneration)
    else:
        images = mExposure.image.array

    return scl.Observation(
        images=images,
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
