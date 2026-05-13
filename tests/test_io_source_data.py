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

"""Round-trip tests for ``IsolatedSourceData`` serialization."""

import unittest

import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np
from lsst.meas.extensions.scarlet.io.source_data import IsolatedSourceData


class TestIsolatedSourceData(lsst.utils.tests.TestCase):
    """Tests for ``IsolatedSourceData.as_dict`` ↔ ``from_dict`` in
    ``lsst.meas.extensions.scarlet.io.source_data``.

    ``IsolatedSourceData`` is a dataclass that backs the on-disk JSON
    representation of every isolated-source row in a deblend catalog.
    Any serialization round-trip that drops or reorders a field would
    silently corrupt those catalogs at the next read.
    """

    def test_isolated_source_data_roundtrip(self):
        """All fields of ``IsolatedSourceData`` survive
        ``as_dict`` → ``from_dict``.
        """
        # Non-trivial span_array (a circle) is sufficient to check
        # shape and dtype round-trip; the dedicated span-array test
        # below pins bit-exact recovery for a larger mask.
        span = scl.utils.get_circle_mask(5, dtype=np.float32)
        original = IsolatedSourceData(
            span_array=span,
            origin=(5, 10),
            peak=(7, 12),
            metadata={"id": 42, "note": "test"},
        )

        roundtripped = IsolatedSourceData.from_dict(original.as_dict())

        np.testing.assert_array_equal(
            roundtripped.span_array, original.span_array
        )
        self.assertEqual(roundtripped.origin, original.origin)
        self.assertEqual(roundtripped.peak, original.peak)
        self.assertEqual(roundtripped.metadata, original.metadata)
        self.assertEqual(roundtripped.source_type, original.source_type)
        self.assertEqual(roundtripped.version, original.version)

    def test_isolated_source_data_roundtrip_no_metadata(self):
        """``metadata=None`` round-trips as ``None`` (the field is
        omitted from the dict entirely).
        """
        original = IsolatedSourceData(
            span_array=np.ones((3, 3), dtype=np.float32),
            origin=(0, 0),
            peak=(1, 1),
        )
        encoded = original.as_dict()
        # The dict shape should not carry a "metadata" key when the
        # source has no metadata — otherwise downstream JSON would
        # have a stray null that callers might mishandle.
        self.assertNotIn("metadata", encoded)

        roundtripped = IsolatedSourceData.from_dict(encoded)
        self.assertIsNone(roundtripped.metadata)

    def test_span_array_roundtrip(self):
        """A non-trivial ``span_array`` survives the round-trip
        bit-for-bit.

        The mask is a diameter-15 circle (significantly larger than the
        scalar-field test) and the assertion is exact element-wise
        equality, so any reshape or stride bug would show up.
        """
        span = scl.utils.get_circle_mask(15, dtype=np.float32)
        # Sanity check that the test fixture itself is "non-trivial":
        # the circle covers most but not all of the bounding box.
        self.assertGreater(span.sum(), 0)
        self.assertLess(span.sum(), span.size)

        original = IsolatedSourceData(
            span_array=span,
            origin=(-7, 3),
            peak=(0, 10),
        )
        roundtripped = IsolatedSourceData.from_dict(original.as_dict())

        np.testing.assert_array_equal(
            roundtripped.span_array, original.span_array
        )
        self.assertEqual(roundtripped.span_array.shape, span.shape)
        self.assertEqual(roundtripped.span_array.dtype, span.dtype)


if __name__ == "__main__":
    unittest.main()
