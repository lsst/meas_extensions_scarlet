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

"""Round-trip tests for ``scarletBoxToBBox`` / ``bboxToScarletBox``."""

import unittest

import lsst.geom as geom
import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
import lsst.utils.tests


class TestBoxConversions(lsst.utils.tests.TestCase):
    """Round-trip tests for the box conversions in
    ``lsst.meas.extensions.scarlet.utils``.

    Scarlet stores boxes as ``(..., y, x)``-ordered shape and origin
    tuples; afw uses ``Box2I`` with separate `(x, y)` accessors. The
    conversion functions also accept an ``xy0`` offset so the same call
    can hop between a blend-local frame and an exposure-global frame.
    """

    def test_scarletBoxToBBox_roundtrip(self):
        """``bboxToScarletBox(scarletBoxToBBox(box)) == box`` over a
        sweep of origins and shapes.
        """
        cases = [
            # Zero origin — the most common parent-blend case.
            ((5, 8), (0, 0)),
            # Small positive origin with an asymmetric (y, x) shape.
            ((25, 30), (17, 5)),
            # Degenerate 1×1 box far from the origin.
            ((1, 1), (100, 200)),
            # Negative origin — exercises sub-image offsets where the
            # blend extends below (0, 0).
            ((50, 50), (-10, -20)),
            # Origin pinned to a single axis (y == 0, x large).
            ((3, 17), (0, 99)),
        ]
        for shape, origin in cases:
            with self.subTest(shape=shape, origin=origin):
                box = scl.Box(shape, origin)
                bbox = mes.utils.scarletBoxToBBox(box)
                roundtripped = mes.utils.bboxToScarletBox(bbox)
                self.assertTupleEqual(roundtripped.origin, origin)
                self.assertTupleEqual(roundtripped.shape, shape)

    def test_bboxToScarletBox_roundtrip(self):
        """``scarletBoxToBBox(bboxToScarletBox(bbox)) == bbox`` over a
        sweep of corners and extents.
        """
        cases = [
            # Zero-origin Box2I — the trivial case.
            (geom.Point2I(0, 0), geom.Extent2I(5, 8)),
            # Positive corner; mirrors the scarlet test above with
            # axes swapped to confirm the (x, y) / (y, x) ordering
            # convention is consistent in both directions.
            (geom.Point2I(17, 5), geom.Extent2I(30, 25)),
            # Negative corner — afw allows it; this pins that
            # ``bboxToScarletBox`` doesn't clamp it to zero.
            (geom.Point2I(-20, -10), geom.Extent2I(50, 50)),
            # Single-axis offset, asymmetric extent.
            (geom.Point2I(99, 0), geom.Extent2I(17, 3)),
        ]
        for minPoint, extent in cases:
            # ``subTest`` kwargs are serialized through ``execnet`` when
            # the suite runs under ``pytest-xdist`` (e.g. on Jenkins),
            # and ``execnet`` can't dump pybind11 objects. Label the
            # subtest with plain int tuples instead.
            with self.subTest(
                minPoint=(minPoint.getX(), minPoint.getY()),
                extent=(extent.getX(), extent.getY()),
            ):
                bbox = geom.Box2I(minPoint, extent)
                box = mes.utils.bboxToScarletBox(bbox)
                roundtripped = mes.utils.scarletBoxToBBox(box)
                self.assertEqual(roundtripped, bbox)

    def test_box_negative_origin(self):
        """Both conversions handle origins below ``(0, 0)``."""
        box = scl.Box((10, 8), (-5, -3))
        bbox = mes.utils.scarletBoxToBBox(box)
        self.assertEqual(bbox.getMinY(), -5)
        self.assertEqual(bbox.getMinX(), -3)
        self.assertEqual(bbox.getHeight(), 10)
        self.assertEqual(bbox.getWidth(), 8)
        roundtripped = mes.utils.bboxToScarletBox(bbox)
        self.assertTupleEqual(roundtripped.origin, box.origin)
        self.assertTupleEqual(roundtripped.shape, box.shape)

    def test_box_with_xy0_offset(self):
        """A non-zero ``xy0`` shifts forward and back symmetrically."""
        box = scl.Box((25, 30), (17, 5))
        xy0 = geom.Point2I(100, 200)
        bbox = mes.utils.scarletBoxToBBox(box, xy0)
        self.assertEqual(bbox.getMinX(), 5 + 100)
        self.assertEqual(bbox.getMinY(), 17 + 200)
        self.assertEqual(bbox.getWidth(), 30)
        self.assertEqual(bbox.getHeight(), 25)
        roundtripped = mes.utils.bboxToScarletBox(bbox, xy0)
        self.assertTupleEqual(roundtripped.origin, box.origin)
        self.assertTupleEqual(roundtripped.shape, box.shape)

    def test_box_transforms(self):
        """Backward-continuity lift of ``TestUtils.test_box_transforms``
        from the pre-refactor ``test_deblend.py``.
        """
        box = scl.Box((25, 30), (17, 5))
        bbox = mes.utils.scarletBoxToBBox(box)
        x0, y0 = bbox.getMin()
        width = bbox.getWidth()
        height = bbox.getHeight()
        self.assertTupleEqual((y0, x0), box.origin)
        self.assertTupleEqual((height, width), box.shape)

        newBox = mes.utils.bboxToScarletBox(bbox)
        self.assertTupleEqual(newBox.origin, box.origin)
        self.assertTupleEqual(newBox.shape, box.shape)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
