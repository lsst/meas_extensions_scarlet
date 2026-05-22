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

"""Tests for the helpers in ``lsst.meas.extensions.scarlet.io.utils``."""

import unittest

import lsst.meas.extensions.scarlet as mes
import lsst.utils.tests
import numpy as np
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.meas.extensions.scarlet.io.model_data import LsstScarletModelData
from lsst.meas.extensions.scarlet.io.source_data import IsolatedSourceData
from lsst.pipe.base import NoWorkFound


class TestUpdateCatalogFootprints(lsst.utils.tests.TestCase):
    """Tests for the empty-input guard in
    ``lsst.meas.extensions.scarlet.io.updateCatalogFootprints``.
    """

    @staticmethod
    def _empty_catalog():
        # The empty-input guard returns/raises before the catalog is
        # touched, so a bare minimal-schema catalog is enough.
        return SourceCatalog(SourceTable.make(SourceTable.makeMinimalSchema()))

    def test_update_catalog_footprints_empty_raises(self):
        """``updateCatalogFootprints`` raises ``NoWorkFound`` when the
        model data has neither blends nor isolated sources.

        ``NoWorkFound`` is a ``lsst.pipe.base`` control-flow exception:
        raising it short-circuits the quantum on empty input. The guard
        previously *returned* the exception instead of raising it, so the
        empty-input case slipped past every caller. Regression test for
        finding C-2 of the ``audits/audit-2026-05-05.md`` audit.
        """
        modelData = LsstScarletModelData()
        self.assertEqual(len(modelData.blends), 0)
        self.assertEqual(len(modelData.isolated), 0)

        with self.assertRaises(NoWorkFound):
            mes.io.updateCatalogFootprints(
                modelData, self._empty_catalog(), band="r"
            )

    def test_update_catalog_footprints_isolated_only_returns(self):
        """With isolated sources but no blends, the function returns
        ``None`` without raising — there is nothing to hydrate.

        Pins the branch adjacent to the C-2 guard: an isolated-only
        model (which happens for fields with only u-band images) is a
        valid no-op, not an empty-input error.
        """
        isolated = {
            1: IsolatedSourceData(
                span_array=np.ones((3, 3), dtype=np.float32),
                origin=(0, 0),
                peak=(1, 1),
            )
        }
        modelData = LsstScarletModelData(isolated=isolated)
        self.assertEqual(len(modelData.blends), 0)

        result = mes.io.updateCatalogFootprints(
            modelData, self._empty_catalog(), band="r"
        )
        self.assertIsNone(result)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
