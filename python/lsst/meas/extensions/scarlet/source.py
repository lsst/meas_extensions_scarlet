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
from scarlet.bbox import Box

from lsst.geom import Point2I, Box2I, Extent2I
from lsst.afw.geom import SpanSet
from lsst.afw.detection import Footprint, PeakCatalog
from lsst.afw.detection.multiband import MultibandFootprint
from lsst.afw.image import Mask, MultibandImage
import lsst.log

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


def bboxToScarletBox(nBands, bbox, xy0=Point2I()):
    """Convert a Box2I to a scarlet bounding Box

    Parameters
    ----------
    nBands : `int`
        Number of bands in the full image.
    bbox : `lsst.geom.Box2I`
        The Box2I to convert into a scarlet `Box`.
    xy0 : `lsst.geom.Point2I`.
        An overall offset to subtract from the `Box2I`.
        This is common in blends, where `xy0` is the minimum pixel
        location of the blend and `bbox` is the box containing
        a source in the blend.

    Returns
    -------
    box : `scarlet.Box`
        A scarlet `Box` that is more useful for slicing image data
        as a numpy array.
    """
    origin = (0, bbox.getMinY()-xy0.y, bbox.getMinX()-xy0.x)
    return Box((nBands, bbox.getHeight(), bbox.getWidth()), origin)


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
    # We want to convolve the model with the observed PSF,
    # which means we need to grow the model box by the PSF to
    # account for all of the flux after convolution.
    # FYI: The `scarlet.Box` class implements the `&` operator
    # to take the intersection of two boxes.

    # Get the PSF size and radii to grow the box
    py, px = blend.observations[0].psf.get_model().shape[1:]
    dh = py // 2
    dw = px // 2
    shape = (source.bbox.shape[0], source.bbox.shape[1] + py, source.bbox.shape[2] + px)
    origin = (source.bbox.origin[0], source.bbox.origin[1] - dh, source.bbox.origin[2] - dw)
    # Create the larger box to fit the model + PSf
    bbox = Box(shape, origin=origin)
    # Only use the portion of the convolved model that fits in the image
    overlap = bbox & source.frame.bbox
    # Load the full multiband model in the larger box
    model = source.model_to_box(overlap)
    # Convolve the model with the PSF in each band
    # Always use a real space convolution to limit artifacts
    model = blend.observations[0].renderer.convolve(model, convolution_type="real").astype(dtype)
    # Update xy0 with the origin of the sources box
    xy0 = Point2I(overlap.origin[-1] + xy0.x, overlap.origin[-2] + xy0.y)
    # Create the spans for the footprint
    valid = np.max(np.array(model), axis=0) != 0
    valid = Mask(valid.astype(np.int32), xy0=xy0)
    spans = SpanSet.fromMask(valid)

    # Add the location of the source to the peak catalog
    peakCat = PeakCatalog(source.detectedPeak.table)
    peakCat.append(source.detectedPeak)
    # Create the MultibandHeavyFootprint
    foot = Footprint(spans)
    foot.setPeakCatalog(peakCat)
    model = MultibandImage(mExposure.filters, model, valid.getBBox())
    mHeavy = MultibandFootprint.fromImages(mExposure.filters, model, footprint=foot)
    return mHeavy
