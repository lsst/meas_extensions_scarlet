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

from functools import partial

import numpy as np
from scarlet.source import PointSource, MultiComponentSource, init_extended_source
from scarlet.component import FactorizedComponent
from scarlet.parameter import Parameter, relative_step
from scarlet.constraint import (PositivityConstraint, MonotonicityConstraint,
                                SymmetryConstraint, ConstraintChain)

import lsst.afw.image as afwImage
from lsst.afw.geom import SpanSet
from lsst.geom import Point2I
import lsst.log
import lsst.afw.detection as afwDet

__all__ = ["init_source", "morphToHeavy", "modelToHeavy"]

logger = lsst.log.Log.getLogger("meas.deblender.deblend")


class ExtendedSource(FactorizedComponent):
    def __init__(
        self,
        frame,
        sky_coord,
        observations,
        obs_idx=0,
        thresh=1.0,
        symmetric=False,
        monotonic=True,
        shifting=False,
        normalization="max"
    ):
        """Extended source initialized to match a set of observations
        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.observation.Observation`
            Observation(s) to initialize this source.
        obs_idx: int
            Index of the observation in `observations` to
            initialize the morphology.
        thresh: `float`
            Multiple of the background RMS used as a
            flux cutoff for morphology initialization.
        symmetric: `bool`
            Whether or not to enforce symmetry.
        monotonic: `bool`
            Whether or not to make the object monotonically decrease
            in flux from the center.
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        """
        assert normalization.lower() in ["none", "max", "sum"]
        self.symmetric = symmetric
        self.monotonic = monotonic
        center = np.array(frame.get_pixel(sky_coord), dtype="float")
        self.pixel_center = tuple(np.round(center).astype("int"))

        if shifting:
            shift = Parameter(center - self.pixel_center, name="shift", step=1e-1)
        else:
            shift = None

        # initialize from observation
        sed, morph, bbox = init_extended_source(
            sky_coord,
            frame,
            observations,
            obs_idx=obs_idx,
            thresh=thresh,
            symmetric=True,
            monotonic=True,
        )

        sed = Parameter(
            sed,
            name="sed",
            step=partial(relative_step, factor=1e-2),
            constraint=PositivityConstraint(),
        )

        constraints = []
        if monotonic:
            # most astronomical sources are monotonically decreasing
            # from their center
            constraints.append(MonotonicityConstraint())
        if symmetric:
            # have 2-fold rotation symmetry around their center ...
            constraints.append(SymmetryConstraint())

        # ... and are positive emitters
        constraints.append(PositivityConstraint())

        # break degeneracies between sed and morphology
        if normalization.lower() != "none":
            constraints.append(NormalizationConstraint(normalization))
        morph_constraint = ConstraintChain(*constraints)

        morph = Parameter(morph, name="morph", step=1e-2, constraint=morph_constraint)

        super().__init__(frame, sed, morph, bbox=bbox, shift=shift)

    @property
    def center(self):
        if len(self.parameters) == 3:
            return self.pixel_center + self.shift
        else:
            return self.pixel_center


def init_source(frame, peak, observation, bbox,
                symmetric=False, monotonic=True,
                thresh=5, components=1, normalization='max'):
    """Initialize a Source

    The user can specify the number of desired components
    for the modeled source. If scarlet cannot initialize a
    model with the desired number of components it continues
    to attempt initialization of one fewer component until
    it finds a model that can be initialized. If all of the
    models fail, including a `PointSource` model, then skip
    the source.

    Parameters
    ----------
    frame: `LsstFrame`
        The model frame for the scene
    peak: `PeakRecord`
        Record for a peak in the parent `PeakCatalog`
    observation: `LsstObservation`
        The images, psfs, etc, of the observed data.
    bbox: `Rect`
        The bounding box of the parent footprint.
    symmetric: `bool`
        Whether or not the object is symmetric
    monotonic: `bool`
        Whether or not the object has flux monotonically
        decreasing from its center
    thresh: `float`
        Fraction of the background to use as a threshold for
        each pixel in the initialization
    components: int
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
            if (np.any([np.isnan(c.sed) for c in components]) or
                    np.all([c.sed <= 0 for c in source.components])):
                logger.warning("Could not initialize")
                raise ValueError("Could not initialize source")
        except Exception:
            # If the MultiComponentSource failed to initialize
            # try an ExtendedSource
            components -= 1

    if components == 1:
        try:
            source = ExtendedSource(frame, center, observation, thresh=thresh,
                                    symmetric=symmetric, monotonic=monotonic, normalization=normalization)
            if np.any(np.isnan(source.sed)) or np.all(source.sed <= 0):
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


def checkConvergence(source):
    """Check that a source converged
    """
    converged = 0
    if hasattr(source, "components"):
        for component in source.components:
            converged = converged & checkConvergence(component)
    else:
        for p, parameter in enumerate(source.parameters):
            if not parameter.converged:
                converged += 2 << p
    return converged


def morphToHeavy(source, peakSchema, xy0=Point2I()):
    """Convert the morphology to a `HeavyFootprint`
    """
    mask = afwImage.MaskX(np.array(source.morph > 0, dtype=np.int32), xy0=xy0)
    ss = SpanSet.fromMask(mask)

    if len(ss) == 0:
        return None

    tfoot = afwDet.Footprint(ss, peakSchema=peakSchema)
    cy, cx = source.pixel_center
    xmin, ymin = xy0
    peakFlux = source.morph[cy, cx]
    tfoot.addPeak(cx+xmin, cy+ymin, peakFlux)
    timg = afwImage.ImageF(source.morph, xy0=xy0)
    timg = timg[tfoot.getBBox()]
    heavy = afwDet.makeHeavyFootprint(tfoot, afwImage.MaskedImageF(timg))
    return heavy


def modelToHeavy(source, filters, xy0=Point2I(), observation=None, dtype=np.float32):
    """Convert the model to a `MultibandFootprint`
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
