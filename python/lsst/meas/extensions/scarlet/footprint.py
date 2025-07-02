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
from lsst.afw.table import SourceCatalog


def footprintsToNumpy(
    catalog: SourceCatalog,
    shape: tuple[int, int],
    xy0: tuple[int, int] | None = None,
    asBool: bool = True,
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
        result[yidx, xidx] = src.getId()
    if asBool:
        result = result != 0
    return result


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
    copyFromFirst: bool | None = True,
) -> afwFootprint:
    """Calculate the intersection of two Footprints.

    Parameters
    ----------
    footprint1:
        The first afw Footprint.
    footprint2:
        The second afw Footprint.
    copyFromFirst:
        Whether or not to copy the peaks from the first Footprint.

    Returns
    -------
    result:
        The Footprint containing the intersection of the two footprints.
    """
    # Create the intersecting footprint
    spans = footprint1.spans.intersect(footprint2.spans)
    result = afwFootprint(spans)
    result.setPeakSchema(footprint1.peaks.getSchema())

    # Only copy peaks that are contained in the intersection
    for peak in (footprint1 if copyFromFirst else footprint2).peaks:
        if spans.contains(geom.Point2I(peak.getIx(), peak.getIy())):
            # Add a copy of the entire peak record to preserve the ID
            result.peaks.append(peak)
    return result


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
