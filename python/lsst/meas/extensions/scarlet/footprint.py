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

"""Functions for converting between afw and scarlet footprints."""

from typing import Sequence

import lsst.geom as geom
import lsst.scarlet.lite as scl
import numpy as np
from lsst.afw.detection import Footprint as afwFootprint
from lsst.afw.detection import HeavyFootprintF, PeakCatalog, makeHeavyFootprint
from lsst.afw.detection.multiband import MultibandFootprint
from lsst.afw.geom import SpanSet
from lsst.afw.image import Image as afwImage
from lsst.afw.image import Mask, MaskedImage, MultibandImage
from lsst.scarlet.lite.detect_pybind11 import Peak

from .utils import bboxToScarletBox


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

    # Create the MultibandHeavyFootprint and
    # add the location of the source to the peak catalog.
    foot = afwFootprint(spans)
    foot.addPeak(source.center[1], source.center[0], np.max(model.data))
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
    It would be better to create a PeakCatalog directly,
    but currently that is not supported in afw.

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
