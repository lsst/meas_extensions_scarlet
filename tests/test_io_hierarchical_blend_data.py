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

"""Tests for the meas-owned ``LsstHierarchicalBlendData`` and its independent
schema lineage.
"""

import unittest

import numpy as np

import lsst.scarlet.lite as scl
import lsst.utils.tests
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.meas.extensions.scarlet.io import hierarchical_blend_data as hbd_module
from lsst.meas.extensions.scarlet.io import (
    LsstHierarchicalBlendData,
    LsstScarletModelData,
    updateCatalogFootprints,
)
from lsst.scarlet.lite.io.blend_base import ScarletBlendBaseData
from lsst.scarlet.lite.io.migration import MigrationRegistry


DEFAULT_SHAPE = (5, 6)
DEFAULT_ORIGIN = (10, 20)


def get_spans(shape=(5, 6)):
    spans = np.zeros(shape, dtype=bool)
    spans[1:4, 2:5] = True
    return spans


def get_child():
    return scl.io.ScarletBlendData(origin=DEFAULT_ORIGIN, shape=DEFAULT_SHAPE, sources={})


class TestLsstHierarchicalBlendData(lsst.utils.tests.TestCase):
    """Round-trip, conversion, and lineage tests for
    LsstHierarchicalBlendData.
    """

    def test_roundtrip(self):
        spans = get_spans()
        child = scl.io.ScarletBlendData(origin=(11, 22), shape=(3, 4), sources={})
        blend = LsstHierarchicalBlendData(
            children={7: child},
            span_array=spans,
            origin=DEFAULT_ORIGIN,
            metadata={"note": "detected parent"},
        )
        blend2 = ScarletBlendBaseData.from_dict(blend.as_dict())
        self.assertIsInstance(blend2, LsstHierarchicalBlendData)

        # Schema/dispatch tags.
        self.assertEqual(blend2.blend_type, "lsst_hierarchical")
        self.assertEqual(blend2.version, hbd_module.CURRENT_SCHEMA)

        # Footprint and provenance.
        np.testing.assert_array_equal(blend2.span_array, spans)
        self.assertEqual(blend2.span_array.dtype, np.dtype(bool))
        self.assertEqual(tuple(blend2.origin), DEFAULT_ORIGIN)
        self.assertFalse(blend2.legacy_spans)

        # Derived shape/bbox properties.
        self.assertEqual(blend2.shape, spans.shape)
        self.assertEqual(blend2.bbox.origin, DEFAULT_ORIGIN)
        self.assertEqual(blend2.bbox.shape, spans.shape)

        # Children round-trip to their concrete type with fields intact.
        self.assertEqual(set(blend2.children), {7})
        child2 = blend2.children[7]
        self.assertIsInstance(child2, scl.io.ScarletBlendData)
        self.assertEqual(tuple(child2.origin), (11, 22))
        self.assertEqual(tuple(child2.shape), (3, 4))

        # Residual metadata survives.
        self.assertEqual(blend2.metadata, {"note": "detected parent"})

    def test_registry_routes_new_tag(self):
        blend = LsstHierarchicalBlendData(
            children={1: get_child()}, span_array=get_spans(), origin=DEFAULT_ORIGIN
        )
        data = blend.as_dict()
        self.assertEqual(data["blend_type"], "lsst_hierarchical")
        self.assertIs(ScarletBlendBaseData.blend_registry["lsst_hierarchical"],
                      LsstHierarchicalBlendData)

    def test_scarlet_lite_hierarchical_not_overridden(self):
        self.assertIs(
            ScarletBlendBaseData.blend_registry["hierarchical"],
            scl.io.HierarchicalBlendData,
        )
        # Independent lineages, each with its own current version.
        self.assertEqual(MigrationRegistry.current["lsst_hierarchical"],
                         hbd_module.CURRENT_SCHEMA)
        self.assertEqual(MigrationRegistry.current["hierarchical"],
                         scl.io.hierarchical_blend.CURRENT_SCHEMA)

    def test_convert_synthesis_rejects_nested_children(self):
        nested = scl.io.HierarchicalBlendData(
            children={1: scl.io.HierarchicalBlendData(children={2: get_child()})}
        )
        with self.assertRaises(NotImplementedError):
            LsstHierarchicalBlendData.convert_from_hierarchical(nested.as_dict())

    def test_updateCatalogFootprints_rejects_non_hierarchical(self):
        modelData = LsstScarletModelData(
            bands=("g",),
            model_psf=np.ones((3, 3), dtype=np.float32),
            psf=np.ones((1, 3, 3), dtype=np.float32),
        )
        modelData.blends[1] = scl.io.ScarletBlendData(
            origin=(0, 0), shape=(3, 3), sources={}
        )
        catalog = SourceCatalog(SourceTable.makeMinimalSchema())
        with self.assertRaises(ValueError) as cm:
            updateCatalogFootprints(modelData, catalog, "g")
        self.assertIn("LsstHierarchicalBlendData", str(cm.exception))


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
