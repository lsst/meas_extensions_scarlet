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

import lsst.afw.image as afwImage
from lsst.afw.geom import SpanSet
from lsst.geom import Point2I
import lsst.log
import lsst.afw.detection as afwDet

__all__ = ["initSource", "morphToHeavy", "modelToHeavy"]

logger = lsst.log.Log.getLogger("meas.deblender.deblend")


def initSource(frame, peak, observation, bbox,
                symmetric=False, monotonic=True,
                thresh=5, components=1):
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
    peak : `PeakRecord`
        Record for a peak in the parent `PeakCatalog`
    observation : `LsstObservation`
        The images, psfs, etc, of the observed data.
    bbox : `lsst.geom.Box2I`
        The bounding box of the parent footprint.
    symmetric : `bool`
        Whether or not the object is symmetric
    monotonic : `bool`
        Whether or not the object has flux monotonically
        decreasing from its center
    thresh : `float`
        Fraction of the background to use as a threshold for
        each pixel in the initialization
    components : int
        The number of components for the source.
        If `components=0` then a `PointSource` model is used.
    """
    assert components <= 2
    xmin = bbox.getMinX()
    ymin = bbox.getMinY()
    center = np.array([peak.getIy()-ymin, peak.getIx()-xmin], dtype=int)

    while components > 1:
        try:
            source = MultiComponentSource(frame, center, observation, symmetric=symmetric,
                                          monotonic=monotonic, thresh=thresh)
            if (np.any([np.any(np.isnan(c.sed)) for c in source.components]) or
                    np.any([np.all(c.sed <= 0) for c in source.components])):
                logger.warning("Could not initialize")
                raise ValueError("Could not initialize source")
            break
        except Exception:
            # If the MultiComponentSource failed to initialize
            # try an ExtendedSource
            components -= 1

    if components == 1:
        try:
            source = ExtendedSource(frame, center, observation, thresh=thresh,
                                    symmetric=symmetric, monotonic=monotonic)
            if np.any(np.isnan(source.sed)) or np.all(source.sed <= 0) or np.sum(source.morph) == 0:
                logger.warning("Could not initialize")
                raise ValueError("Could not initialize source")
        except Exception:
            # If the source is too faint for background detection,
            # initialize it as a PointSource
            components -= 1

    if components == 0:
        try:
            source = PointSource(frame, center, observation)
        except Exception:
            # None of the models worked to initialize the source,
            # so skip this source
            return None

    source.detectedPeak = peak
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
        `HeavyFootprint`.
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
        model = observation.render(source.get_model()).astype(dtype)
    else:
        model = source.get_model().astype(dtype)
    mHeavy = afwDet.MultibandFootprint.fromArrays(filters, model, xy0=xy0)
    peakCat = afwDet.PeakCatalog(source.detectedPeak.table)
    peakCat.append(source.detectedPeak)
    for footprint in mHeavy:
        footprint.setPeakCatalog(peakCat)
    return mHeavy
