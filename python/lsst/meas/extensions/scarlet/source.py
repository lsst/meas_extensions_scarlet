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

import numpy as np
from scarlet.source import PointSource, ExtendedSource, MultiComponentSource
from scarlet.bbox import Box

import lsst.afw.image as afwImage
from lsst.afw.geom import SpanSet
from lsst.geom import Point2I
import lsst.log
import lsst.afw.detection as afwDet

__all__ = ["initSource", "morphToHeavy", "modelToHeavy"]

logger = lsst.log.Log.getLogger("meas.deblender.deblend")


def hasEdgeFlux(source, edgeDistance=1):
    """hasEdgeFlux

    Determine whether or not a source has flux within `edgeDistance`
    of the edge.

    Parameters
    ----------
    source : `scarlet.Component`
        The source to check for edge flux
    edgeDistance : int
        The distance from the edge of the image to consider
        a source an edge source. For example if `edgeDistance=3`
        then any source within 3 pixels of the edge will be
        considered to have edge flux.
        If `edgeDistance` is `None` then the edge check is ignored.

    Returns
    -------
    isEdge: `bool`
        Whether or not the source has flux on the edge.
    """
    if edgeDistance is None:
        return False

    assert edgeDistance > 0

    # Use the first band that has a non-zero SED
    if hasattr(source, "sed"):
        band = np.min(np.where(source.sed > 0)[0])
    else:
        band = np.min(np.where(source.components[0].sed > 0)[0])
    model = source.get_model()[band]
    for edge in range(edgeDistance):
        if (
            np.any(model[edge-1] > 0)
            or np.any(model[-edge] > 0)
            or np.any(model[:, edge-1] > 0)
            or np.any(model[:, -edge] > 0)
        ):
            return True
    return False


def initAllSources(frame, centers, observation,
                   symmetric=False, monotonic=True,
                   thresh=1, maxComponents=1, edgeDistance=1, shifting=False,
                   downgrade=False, fallback=True):
    """Initialize all sources in a blend

    Any sources which cannot be initialized are returned as a `skipped`
    index, the index needed to reinsert them into a catalog to preserve
    their index in the output catalog.

    See `~initSources` for a description of the parameters

    Parameters
    ----------
    centers : list of tuples
        `(y, x)` center location for each source

    Returns
    -------
    sources: list
        List of intialized sources, where each source derives from the
        `~scarlet.Component` class.
    """
    # Only deblend sources that can be initialized
    sources = []
    skipped = []
    for k, center in enumerate(centers):
        source = initSource(
            frame, center, observation,
            symmetric, monotonic,
            thresh, maxComponents, edgeDistance, shifting,
            downgrade, fallback)
        if source is not None:
            sources.append(source)
        else:
            skipped.append(k)
    return sources, skipped


def initSource(frame, center, observation,
               symmetric=False, monotonic=True,
               thresh=1, maxComponents=1, edgeDistance=1, shifting=False,
               downgrade=False, fallback=True):
    """Initialize a Source

    The user can specify the number of desired components
    for the modeled source. If scarlet cannot initialize a
    model with the desired number of components it continues
    to attempt initialization of one fewer component until
    it finds a model that can be initialized.
    It is possible that scarlet will be unable to initialize a
    source with the desired number of components, for example
    a two component source might have degenerate components,
    a single component source might not have enough signal in
    the joint coadd (all bands combined together into
    single signal-to-noise weighted image for initialization)
    to initialize, and a true spurious detection will not have
    enough signal to initialize as a point source.
    If all of the models fail, including a `PointSource` model,
    then this source is skipped.

    Parameters
    ----------
    frame : `LsstFrame`
        The model frame for the scene
    center : `tuple` of `float``
        `(y, x)` location for the center of the source.
    observation : `~scarlet.Observation`
        The `Observation` that contains the images, weights, and PSF
        used to generate the model.
    symmetric : `bool`
        Whether or not the object is symmetric
    monotonic : `bool`
        Whether or not the object has flux monotonically
        decreasing from its center
    thresh : `float`
        Fraction of the background to use as a threshold for
        each pixel in the initialization
    maxComponents : int
        The maximum number of components in a source.
        If `fallback` is `True` then when
        a source fails to initialize with `maxComponents` it
        will continue to subtract one from the number of components
        until it reaches zero (which fits a point source).
        If a point source cannot be fit then the source is skipped.
    edgeDistance : int
        The distance from the edge of the image to consider
        a source an edge source. For example if `edgeDistance=3`
        then any source within 3 pixels of the edge will be
        considered to have edge flux.
        If `edgeDistance` is `None` then the edge check is ignored.
    shifting : bool
        Whether or not to fit the position of a source.
        This is an expensive operation and is typically only used when
        a source is on the edge of the detector.
    downgrade : bool
        Whether or not to decrease the number of components for sources
        with small bounding boxes. For example, a source with no flux
        outside of its 16x16 box is unlikely to be resolved enough
        for multiple components, so a single source can be used.
    fallback : bool
        Whether to reduce the number of components
        if the model cannot be initialized with `maxComponents`.
        This is unlikely to be used in production
        but can be useful for troubleshooting when an error can cause
        a particular source class to fail every time.
    """
    while maxComponents > 1:
        try:
            source = MultiComponentSource(frame, center, observation, symmetric=symmetric,
                                          monotonic=monotonic, thresh=thresh, shifting=shifting)
            if (np.any([np.any(np.isnan(c.sed)) for c in source.components])
                    or np.any([np.all(c.sed <= 0) for c in source.components])
                    or np.any([np.any(~np.isfinite(c.morph)) for c in source.components])):
                msg = "Could not initialize source at {} with {} components".format(center, maxComponents)
                logger.warning(msg)
                raise ValueError(msg)

            if downgrade and np.all([np.all(np.array(c.bbox.shape[1:]) <= 8) for c in source.components]):
                # the source is in a small box so it must be a point source
                maxComponents = 0
            elif downgrade and np.all([np.all(np.array(c.bbox.shape[1:]) <= 16) for c in source.components]):
                # if the source is in a slightly larger box
                # it is not big enough to model with 2 components
                maxComponents = 1
            elif hasEdgeFlux(source, edgeDistance):
                source.shifting = True

            break
        except Exception as e:
            if not fallback:
                raise e
            # If the MultiComponentSource failed to initialize
            # try an ExtendedSource
            maxComponents -= 1

    if maxComponents == 1:
        try:
            source = ExtendedSource(frame, center, observation, thresh=thresh,
                                    symmetric=symmetric, monotonic=monotonic, shifting=shifting)
            if np.any(np.isnan(source.sed)) or np.all(source.sed <= 0) or np.sum(source.morph) == 0:
                msg = "Could not initlialize source at {} with 1 component".format(center)
                logger.warning(msg)
                raise ValueError(msg)

            if downgrade and np.all(np.array(source.bbox.shape[1:]) <= 16):
                # the source is in a small box so it must be a point source
                maxComponents = 0
            elif hasEdgeFlux(source, edgeDistance):
                source.shifting = True
        except Exception as e:
            if not fallback:
                raise e
            # If the source is too faint for background detection,
            # initialize it as a PointSource
            maxComponents -= 1

    if maxComponents == 0:
        try:
            source = PointSource(frame, center, observation)
        except Exception:
            # None of the models worked to initialize the source,
            # so skip this source
            return None

    if hasEdgeFlux(source, edgeDistance):
        # The detection algorithm implemented in meas_algorithms
        # does not place sources within the edge mask
        # (roughly 5 pixels from the edge). This results in poor
        # deblending of the edge source, which for bright sources
        # may ruin an entire blend. So we reinitialize edge sources
        # to allow for shifting and return the result.
        if not isinstance(source, PointSource) and not shifting:
            return initSource(frame, center, observation,
                              symmetric, monotonic, thresh, maxComponents,
                              edgeDistance, shifting=True)
        source.isEdge = True
    else:
        source.isEdge = False

    return source


def morphToHeavy(source, peakSchema, xy0=Point2I()):
    """Convert the morphology to a `HeavyFootprint`

    Parameters
    ----------
    source : `scarlet.Component`
        The scarlet source with a morphology to convert to
        a `HeavyFootprint`.
    peakSchema : `lsst.daf.butler.Schema`
        The schema for the `PeakCatalog` of the `HeavyFootprint`.
    xy0 : `tuple`
        `(x,y)` coordinates of the bounding box containing the
        `HeavyFootprint`.

    Returns
    -------
    heavy : `lsst.afw.detection.HeavyFootprint`
    """
    mask = afwImage.MaskX(np.array(source.morph > 0, dtype=np.int32), xy0=xy0)
    ss = SpanSet.fromMask(mask)

    if len(ss) == 0:
        return None

    tfoot = afwDet.Footprint(ss, peakSchema=peakSchema)
    cy, cx = source.pixel_center
    xmin, ymin = xy0
    # HeavyFootprints are not defined for 64 bit floats
    morph = source.morph.astype(np.float32)
    peakFlux = morph[cy, cx]
    tfoot.addPeak(cx+xmin, cy+ymin, peakFlux)
    timg = afwImage.ImageF(morph, xy0=xy0)
    timg = timg[tfoot.getBBox()]
    heavy = afwDet.makeHeavyFootprint(tfoot, afwImage.MaskedImageF(timg))
    return heavy


def modelToHeavy(source, filters, xy0=Point2I(), observation=None, dtype=np.float32):
    """Convert the model to a `MultibandFootprint`

    Parameters
    ----------
    source : `scarlet.Component`
        The source to convert to a `HeavyFootprint`.
    filters : `iterable`
        A "list" of names for each filter.
    xy0 : `lsst.geom.Point2I`
        `(x,y)` coordinates of the bounding box containing the
        `HeavyFootprint`. If `observation` is not `None` then
        this parameter is updated with the position of the new model
    observation : `scarlet.Observation`
        The scarlet observation, used to convolve the image with
        the origin PSF. If `observation`` is `None` then the
        `HeavyFootprint` will exist in the model frame.
    dtype : `numpy.dtype`
        The data type for the returned `HeavyFootprint`.

    Returns
    -------
    mHeavy : `lsst.detection.MultibandFootprint`
        The multi-band footprint containing the model for the source.
    """
    if observation is not None:
        # We want to convolve the model with the observed PSF,
        # which means we need to grow the model box by the PSF to
        # account for all of the flux after convolution.
        # FYI: The `scarlet.Box` class implements the `&` operator
        # to take the intersection of two boxes.

        # Get the PSF size and radii to grow the box
        py, px = observation.frame.psf.shape[1:]
        dh = py // 2
        dw = px // 2
        shape = (source.bbox.shape[0], source.bbox.shape[1] + py, source.bbox.shape[2] + px)
        origin = (source.bbox.origin[0], source.bbox.origin[1] - dh, source.bbox.origin[2] - dw)
        # Create the larger box to fit the model + PSf
        bbox = Box(shape, origin=origin)
        # Only use the portion of the convolved model that fits in the image
        overlap = bbox & source.model_frame
        # Load the full multiband model in the larger box
        model = source.model_to_frame(overlap)
        # Convolve the model with the PSF in each band
        # Always use a real space convolution to limit artifacts
        model = observation.convolve(model, convolution_type="real").astype(dtype)
        # Update xy0 with the origin of the sources box
        xy0 = Point2I(overlap.origin[-1] + xy0.x, overlap.origin[-2] + xy0.y)
    else:
        model = source.get_model().astype(dtype)
    mHeavy = afwDet.MultibandFootprint.fromArrays(filters, model, xy0=xy0)
    peakCat = afwDet.PeakCatalog(source.detectedPeak.table)
    peakCat.append(source.detectedPeak)
    for footprint in mHeavy:
        footprint.setPeakCatalog(peakCat)
    return mHeavy
