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

__all__ = [
    "DeblenderMetrics",
    "setDeblenderMetrics",
]

from dataclasses import dataclass

import numpy as np
from lsst.scarlet.lite import Blend


@dataclass
class DeblenderMetrics:
    """Metrics and measurements made on single sources.

    Store deblender metrics to be added as attributes to a scarlet source
    before it is converted into a `SourceRecord`.
    TODO: When DM-34414 is finished this class will be eliminated and the
    metrics will be added to the schema using a pipeline task that calculates
    them from the stored deconvolved models.

    All of the parameters are one dimensional numpy arrays,
    with an element for each band in the observed images.

    `maxOverlap` is useful as a metric for determining how blended a source
    is because if it only overlaps with other sources at or below
    the noise level, it is likely to be a mostly isolated source
    in the deconvolved model frame.

    `fluxOverlapFraction` is potentially more useful than the canonical
    "blendedness" (or purity) metric because it accounts for potential
    biases created during deblending by not weighting the overlapping
    flux with the flux of this sources model.

    Attributes
    ----------
    maxOverlap:
        The maximum overlap that the source has with its neighbors in
        a single pixel.
    fluxOverlap:
        The total flux from neighbors overlapping with the current source.
    fluxOverlapFraction:
        The fraction of `flux from neighbors/source flux` for a
        given source within the source's footprint.
    blendedness:
        The metric for determining how blended a source is using the
        Bosch et al. 2018 metric for "blendedness." Note that some
        surveys use the term "purity," which is `1-blendedness`.
    """

    maxOverlap: np.array
    fluxOverlap: np.array
    fluxOverlapFraction: np.array
    blendedness: np.array


def setDeblenderMetrics(blend: Blend):
    """Set metrics that can be used to evalute the deblender accuracy

    This function calculates the `DeblenderMetrics` for each source in the
    blend, and assigns it to that sources `metrics` property in place.

    Parameters
    ----------
    blend:
        The blend containing the sources to measure.
    """
    # Store the full model of the scene for comparison
    blendModel = blend.get_model().data
    for k, src in enumerate(blend.sources):
        # Extract the source model in the full bounding box
        model = src.get_model().project(bbox=blend.bbox).data
        # The footprint is the 2D array of non-zero pixels in each band
        footprint = np.bitwise_or.reduce(model > 0, axis=0)
        # Calculate the metrics.
        # See `DeblenderMetrics` for a description of each metric.
        neighborOverlap = (blendModel - model) * footprint[None, :, :]
        maxOverlap = np.max(neighborOverlap, axis=(1, 2))
        fluxOverlap = np.sum(neighborOverlap, axis=(1, 2))
        fluxModel = np.sum(model, axis=(1, 2))
        fluxOverlapFraction = np.zeros((len(model),), dtype=float)
        isFinite = fluxModel > 0
        fluxOverlapFraction[isFinite] = fluxOverlap[isFinite] / fluxModel[isFinite]
        blendedness = 1 - np.sum(model * model, axis=(1, 2)) / np.sum(
            blendModel * model, axis=(1, 2)
        )
        src.metrics = DeblenderMetrics(
            maxOverlap, fluxOverlap, fluxOverlapFraction, blendedness
        )
