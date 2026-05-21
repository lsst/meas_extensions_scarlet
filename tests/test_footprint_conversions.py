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

"""Round-trip tests for the afw ↔ scarlet footprint conversions.

The footprints are built from circular masks produced by
``scl.utils.get_circle_mask`` rather than rectangles. A non-rectangular
span set turns the round-trip "spans match" assertion into a real
check — a rectangle would survive almost any broken conversion.
"""

import unittest

import lsst.geom as geom
import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np
from lsst.afw.detection import Footprint as afwFootprint
from lsst.afw.geom import SpanSet
from lsst.afw.image import Mask
from lsst.scarlet.lite.detect_pybind11 import Peak


def _circle_mask(diameter):
    """Return a ``(diameter, diameter)`` int32 mask of a circle."""
    return scl.utils.get_circle_mask(diameter, dtype=np.int32)


def _afw_circle_footprint(min_point, diameter, peaks):
    """Build a circular afw Footprint with the given peaks.

    Parameters
    ----------
    min_point : `tuple` [`int`]
        ``(x, y)`` corner of the enclosing bbox.
    diameter : `int`
        Diameter of the circle (also the side length of the bbox).
    peaks : `list` [`tuple` [`int`, `int`, `float`]]
        ``(x, y, peakValue)`` triples, in the order afw's ``addPeak``
        expects.
    """
    afw_mask = Mask(_circle_mask(diameter), xy0=geom.Point2I(*min_point))
    fp = afwFootprint(SpanSet.fromMask(afw_mask))
    for x, y, v in peaks:
        fp.addPeak(x, y, v)
    return fp


def _scarlet_circle_footprint(origin, diameter, peaks):
    """Build a circular scarlet Footprint with the given peaks.

    Parameters
    ----------
    origin : `tuple` [`int`]
        ``(y_min, x_min)`` of the enclosing bbox (scarlet's `(y, x)`
        order).
    diameter : `int`
        Diameter of the circle (also the side length of the bbox).
    peaks : `list` [`tuple` [`int`, `int`, `float`]]
        ``(y, x, flux)`` triples, in the order scarlet's ``Peak`` ctor
        expects.
    """
    bounds = scl.detect.bbox_to_bounds(scl.Box((diameter, diameter), origin))
    scl_peaks = [Peak(y, x, v) for y, x, v in peaks]
    return scl.detect.Footprint(_circle_mask(diameter), scl_peaks, bounds)


class TestFootprintConversions(lsst.utils.tests.TestCase):
    """Round-trip tests for ``afwFootprintToScarlet``,
    ``scarletFootprintToAfw``, and ``scarletFootprintsToPeakCatalog``
    in ``lsst.meas.extensions.scarlet.footprint``.

    Peaks live in afw as ``(getIx(), getIy(), getPeakValue())`` and in
    scarlet as ``(peak.x, peak.y, peak.flux)``. The conversion functions
    cross between those conventions and also between afw's
    separate-axis ``Box2I`` and scarlet's ``(y, x)``-ordered ``Box``.
    """

    def test_afwFootprintToScarlet_peak_order(self):
        """afw → scarlet maps ``getIy() → peak.y`` and ``getIx() → peak.x``."""
        peaks_in = [
            # Peak at the centre of a diameter-7 circle.
            (3, 3, 100.0),
            # Off-centre asymmetric (x, y) — a silent x/y flip in the
            # conversion would land at (2, 1) instead.
            (1, 2, 50.0),
            # Opposite quadrant; flux < 1 confirms peakValue is kept
            # as float, not int.
            (5, 4, 0.5),
        ]
        fp = _afw_circle_footprint((0, 0), diameter=7, peaks=peaks_in)
        sf = mes.footprint.afwFootprintToScarlet(fp)
        self.assertEqual(len(sf.peaks), len(peaks_in))
        for (x, y, v), p in zip(peaks_in, sf.peaks):
            self.assertEqual(p.y, y)
            self.assertEqual(p.x, x)
            self.assertEqual(p.flux, v)
        # The span data crosses over too — the scarlet footprint's mask
        # is the bbox-sized array of the afw spans.
        np.testing.assert_array_equal(sf.data, _circle_mask(7))

    def test_scarletFootprintToAfw_peak_order(self):
        """scarlet → afw maps ``peak.x → getIx()`` and ``peak.y → getIy()``."""
        peaks_in = [
            # Same three positions as the afw → scarlet test but with
            # the role of x and y swapped in the input tuple; a silent
            # flip would surface here.
            (3, 3, 100.0),
            (2, 1, 50.0),
            (4, 5, 0.5),
        ]
        sf = _scarlet_circle_footprint((0, 0), diameter=7, peaks=peaks_in)
        fp = mes.footprint.scarletFootprintToAfw(sf)
        self.assertEqual(len(fp.peaks), len(peaks_in))
        for (y, x, v), p in zip(peaks_in, fp.peaks):
            self.assertEqual(p.getIy(), y)
            self.assertEqual(p.getIx(), x)
            self.assertEqual(p.getPeakValue(), v)
        np.testing.assert_array_equal(fp.spans.asArray(), _circle_mask(7))

    def test_roundtrip_footprint_with_negative_origin(self):
        """afw → scarlet → afw preserves a circle whose bbox corner is
        below ``(0, 0)``.
        """
        fp = _afw_circle_footprint(
            min_point=(-5, -3),
            diameter=7,
            # (-2, 0) lands on the circle centre when the bbox origin
            # is (-5, -3): (x - (-5), y - (-3)) == (3, 3).
            peaks=[(-2, 0, 7.0)],
        )
        sf = mes.footprint.afwFootprintToScarlet(fp)
        fp_back = mes.footprint.scarletFootprintToAfw(sf)
        self.assertEqual(fp_back.getBBox(), fp.getBBox())
        np.testing.assert_array_equal(
            fp_back.spans.asArray(), fp.spans.asArray()
        )
        self.assertEqual(len(fp_back.peaks), 1)
        peak = fp_back.peaks[0]
        self.assertEqual(peak.getIx(), -2)
        self.assertEqual(peak.getIy(), 0)
        self.assertEqual(peak.getPeakValue(), 7.0)

    def test_roundtrip_footprint_empty_spans(self):
        """A footprint with no spans round-trips without error and the
        scarlet form is a 0×0 box with no peaks.
        """
        # Empty isn't really a "circle" — the diameter that would
        # describe it is zero, so we build the empty Footprint directly.
        fp = afwFootprint(SpanSet())
        sf = mes.footprint.afwFootprintToScarlet(fp)
        self.assertEqual(tuple(int(s) for s in sf.bbox.shape), (0, 0))
        self.assertEqual(len(sf.peaks), 0)
        fp_back = mes.footprint.scarletFootprintToAfw(sf)
        self.assertEqual(len(fp_back.peaks), 0)
        # afw represents a degenerate footprint as min=(0,0),
        # max=(-1,-1); both round-trip endpoints agree on that.
        self.assertEqual(fp_back.getBBox(), fp.getBBox())

    def test_roundtrip_footprint_edge_pixels(self):
        """A circle with its bbox rooted at ``(0, 0)`` round-trips —
        pins the edge case where the conversion's signed offsets are zero.
        """
        peaks_in = [
            # (3, 0) is the apex of a diameter-7 circle's top row,
            # where mask[0, 3] == 1 — exercises the y == 0 edge.
            (3, 0, 100.0),
            # (0, 3) is the leftmost pixel of the circle's middle row,
            # where mask[3, 0] == 1 — exercises the x == 0 edge.
            (0, 3, 50.0),
        ]
        fp = _afw_circle_footprint((0, 0), diameter=7, peaks=peaks_in)
        sf = mes.footprint.afwFootprintToScarlet(fp)
        fp_back = mes.footprint.scarletFootprintToAfw(sf)
        self.assertEqual(fp_back.getBBox(), fp.getBBox())
        np.testing.assert_array_equal(
            fp_back.spans.asArray(), _circle_mask(7)
        )
        self.assertEqual(len(fp_back.peaks), len(peaks_in))
        for (x, y, v), p in zip(peaks_in, fp_back.peaks):
            self.assertEqual(p.getIx(), x)
            self.assertEqual(p.getIy(), y)
            self.assertEqual(p.getPeakValue(), v)

    def test_scarletFootprintsToPeakCatalog_schema(self):
        """The returned ``PeakCatalog`` has the default ``PeakTable``
        schema and one row per input peak with matching values.
        """
        # Two circular scarlet footprints, three peaks total, with one
        # in negative-coordinate territory to confirm the conversion
        # doesn't clamp.
        sf1 = _scarlet_circle_footprint(
            origin=(0, 0),
            diameter=7,
            peaks=[(3, 3, 100.0), (3, 0, 50.0)],
        )
        sf2 = _scarlet_circle_footprint(
            origin=(-3, -2),
            diameter=7,
            # (0, 1) lands on the circle centre: (y - (-3), x - (-2))
            # == (3, 3).
            peaks=[(0, 1, 25.0)],
        )
        catalog = mes.footprint.scarletFootprintsToPeakCatalog([sf1, sf2])

        # Default PeakTable schema columns; the function uses a "dummy
        # Footprint" internally and so picks up the default schema.
        self.assertEqual(
            set(catalog.schema.getNames()),
            {"id", "i_x", "i_y", "f_x", "f_y", "peakValue"},
        )
        # Catalog rows are emitted in input order; each peak.x/.y/.flux
        # maps to getIx()/getIy()/getPeakValue().
        expected = [(3, 3, 100.0), (0, 3, 50.0), (1, 0, 25.0)]
        self.assertEqual(len(catalog), len(expected))
        for (x, y, v), row in zip(expected, catalog):
            self.assertEqual(row.getIx(), x)
            self.assertEqual(row.getIy(), y)
            self.assertEqual(row.getPeakValue(), v)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
