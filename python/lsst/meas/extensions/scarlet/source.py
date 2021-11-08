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
from scarlet.bbox import Box

from lsst.geom import Point2I
import lsst.log
import lsst.afw.detection as afwDet

__all__ = ["modelToHeavy"]

logger = lsst.log.Log.getLogger("lsst.meas.extensions.scarlet.source")


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
        py, px = observation.psf.get_model().shape[1:]
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
        model = observation.renderer.convolve(model, convolution_type="real").astype(dtype)
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
