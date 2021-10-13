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

<<<<<<< HEAD
from lsst.geom import Point2I
import lsst.afw.detection as afwDet
=======
from lsst.geom import Point2I, Box2I, Extent2I
from lsst.afw.geom import SpanSet, Stencil
from lsst.afw.detection import Footprint, PeakCatalog
from lsst.afw.detection.multiband import MultibandFootprint
from lsst.afw.image import Mask, MultibandImage
import lsst.log
>>>>>>> 2d40ab5 (Fix HeavyFootprints generated from scarlet models)

__all__ = ["modelToHeavy"]

logger = logging.getLogger(__name__)


def scarletBoxToBBox(box, xy0=Point2I()):
    """Convert a scarlet.Box to a Box2I

    Parameters
    ----------
    box : `scarlet.Box`
        The scarlet bounding box to convert
    xy0 : `Point2I`
        An additional offset to apply to the scarlet box.
        This is common since scarlet sources have an origin of
        `(0,0)` at the lower left corner of the blend while
        the blend itself is likely to have an offset in the
        `Exposure`.

    Returns
    -------
    bbox : `lsst.geom.box2I`
        The converted bounding box
    """
    xy0 = Point2I(box.origin[-1]+xy0.x, box.origin[-2]+xy0.y)
    extent = Extent2I(box.shape[-1], box.shape[-2])
    return Box2I(xy0, extent)


def modelToHeavy(source, mExposure, blend, xy0=Point2I(), dtype=np.float32):
    """Convert a scarlet model to a `MultibandFootprint`.

    Parameters
    ----------
    source : `scarlet.Component`
        The source to convert to a `HeavyFootprint`.
    mExposure : `lsst.image.MultibandExposure`
        The multiband exposure containing the image,
        mask, and variance data.
    blend : `scarlet.Blend`
        The `Blend` object that contains information about
        the observation, PSF, etc, used to convolve the
        scarlet model to the observed seeing in each band.
    xy0 : `lsst.geom.Point2I`
        `(x,y)` coordinates of the lower-left pixel of the
        entire blend.
    dtype : `numpy.dtype`
        The data type for the returned `HeavyFootprint`.

    Returns
    -------
    mHeavy : `lsst.detection.MultibandFootprint`
        The multi-band footprint containing the model for the source.
    """
    # Get the SpanSet that contains all of the model pixels
    model = np.array(source.get_model()).astype(np.float32)
    bbox = scarletBoxToBBox(source.bbox, xy0)
    mask = np.max(np.array(model), axis=0) > 0
    mask = Mask(mask.astype(np.int32), xy0=bbox.getMin())
    spans = SpanSet.fromMask(mask)

    # Grow the spanset to allow the model to grow by the PSF
    observation = blend.observations[0]
    py, px = observation.psf.get_model().shape[1:]
    psfSize = np.max([py, px])
    dilation = psfSize // 2
    spans = spans.dilated(dilation, stencil=Stencil.BOX)

    # convolve the model
    dilatedBox = spans.getBBox()
    shape = (len(mExposure.filters), dilatedBox.getHeight(), dilatedBox.getWidth())
    emptyModel = np.zeros(shape, dtype=dtype)
    convolved = MultibandImage(mExposure.filters, emptyModel, dilatedBox)
    convolved[:, bbox].array = model
    convolved.array[:] = observation.renderer.convolve(
        convolved.array, convolution_type="real").astype(dtype)

    # Create the MultibandHeavyFootprint
    foot = Footprint(spans)
    # Clip to the Exposure, just in case the edge sources
    # have convolved flux outside the image
    foot.clipTo(mExposure.getBBox())
    mHeavy = MultibandFootprint.fromImages(mExposure.filters, convolved, footprint=foot)
    # Add the location of the source to the peak catalog
    peakCat = PeakCatalog(source.detectedPeak.table)
    peakCat.append(source.detectedPeak)
    for footprint in mHeavy:
        footprint.setPeakCatalog(peakCat)
    return mHeavy
