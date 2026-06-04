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

"""Tests for the LsstScarletModelData schema migrations."""

import copy
import unittest

import numpy as np

import lsst.scarlet.lite as scl
import lsst.utils.tests
from lsst.meas.extensions.scarlet.io import model_data as model_data_module
from lsst.meas.extensions.scarlet.io.model_data import (
    CURRENT_SCHEMA,
    MODEL_TYPE,
    _to_1_0_0,
    _to_1_0_1,
    _to_1_0_2,
)


class TestModelDataMigrations(lsst.utils.tests.TestCase):
    """Tests for the migration chain and schema constants in
    ``lsst.meas.extensions.scarlet.io.model_data``.

    Each migration function bumps the version and adds the keys
    introduced at that schema step. A regression in any of them
    silently corrupts deblend catalogs read from disk that were
    written by an earlier release.
    """

    def test_to_1_0_0_adds_isolated_key(self):
        """``_to_1_0_0`` adds ``isolated={}``, the ``model_type`` tag,
        and the schema version to pre-schema data.
        """
        # Pre-schema data carries only the ``blends`` key inherited
        # from scarlet_lite's ScarletModelData; no model_type,
        # isolated, or version was emitted before 1.0.0.
        pre = {"blends": {}}
        result = _to_1_0_0(copy.deepcopy(pre))
        self.assertEqual(result["isolated"], {})
        self.assertEqual(result["version"], "1.0.0")
        self.assertEqual(result["model_type"], MODEL_TYPE)

    def test_to_1_0_1_adds_footprint_metadata(self):
        """``_to_1_0_1`` adds ``metadata={"footprint": None}`` and the
        schema version to 1.0.0 data that had no metadata at all.
        """
        # 1.0.0 data has isolated, model_type, version — but no
        # metadata key; 1.0.1 introduced footprint metadata.
        v1_0_0 = {
            "blends": {},
            "isolated": {},
            "model_type": MODEL_TYPE,
            "version": "1.0.0",
        }
        result = _to_1_0_1(copy.deepcopy(v1_0_0))
        self.assertEqual(result["version"], "1.0.1")
        self.assertEqual(result["metadata"], {"footprint": None})

    def test_to_1_0_1_preserves_existing_metadata(self):
        """When 1.0.0 data already carries a ``metadata`` dict (without
        a ``footprint`` key), ``_to_1_0_1`` adds ``footprint=None``
        and leaves the other keys intact.
        """
        # Pins the ``setdefault(...).setdefault(...)`` contract — a
        # naive ``data["metadata"] = {"footprint": None}`` rewrite
        # would silently drop pre-existing metadata.
        v1_0_0 = {
            "blends": {},
            "isolated": {},
            "model_type": MODEL_TYPE,
            "version": "1.0.0",
            "metadata": {"survey": "DES"},
        }
        result = _to_1_0_1(copy.deepcopy(v1_0_0))
        self.assertEqual(result["version"], "1.0.1")
        self.assertEqual(
            result["metadata"], {"survey": "DES", "footprint": None}
        )

    def test_to_1_0_1_handles_none_metadata(self):
        """``_to_1_0_1`` tolerates an explicit ``metadata=None`` entry
        from older payloads.

        Regression test for finding IO-3 of
        ``audits/audit-2026-05-05.md``. The naive
        ``data.setdefault("metadata", {}).setdefault("footprint", None)``
        idiom returned the existing ``None`` and then raised
        ``AttributeError: 'NoneType' object has no attribute 'setdefault'``
        when the payload carried ``metadata=None`` explicitly — the
        same value emitted by
        ``scarlet_model_to_lsst_scarlet_model`` for converted v0
        archives. The migration must replace ``None`` with
        ``{"footprint": None}``.
        """
        v1_0_0 = {
            "blends": {},
            "isolated": {},
            "model_type": MODEL_TYPE,
            "version": "1.0.0",
            "metadata": None,
        }
        result = _to_1_0_1(copy.deepcopy(v1_0_0))
        self.assertEqual(result["version"], "1.0.1")
        self.assertEqual(result["metadata"], {"footprint": None})

    def test_to_1_0_2_promotes_real_hierarchical_spans(self):
        """``_to_1_0_2`` converts a legacy ``hierarchical`` blend that
        carries real ``spans``/``origin`` into an ``lsst_hierarchical``
        blend, promoting the spans verbatim with ``legacy_spans=False``.
        """
        spans = np.zeros((5, 6), dtype=bool)
        spans[1:4, 2:5] = True
        child = scl.io.ScarletBlendData(origin=(10, 20), shape=(5, 6), sources={})
        legacy = scl.io.HierarchicalBlendData(
            children={1: child},
            metadata={"spans": spans.astype(int), "origin": (10, 20)},
        )
        data = {
            "blends": {7: legacy.as_dict()},
            "isolated": {},
            "model_type": MODEL_TYPE,
            "version": "1.0.1",
            "metadata": {},
        }
        result = _to_1_0_2(copy.deepcopy(data))
        self.assertEqual(result["version"], "1.0.2")
        blend = result["blends"][7]
        self.assertEqual(blend["blend_type"], "lsst_hierarchical")
        converted = scl.io.ScarletBlendBaseData.from_dict(blend)
        self.assertIsInstance(converted, model_data_module.LsstHierarchicalBlendData)
        np.testing.assert_array_equal(converted.span_array, spans)
        self.assertEqual(tuple(converted.origin), (10, 20))
        self.assertFalse(converted.legacy_spans)

    def test_to_1_0_2_synthesizes_missing_spans(self):
        """When a legacy ``hierarchical`` blend has no spans, ``_to_1_0_2``
        synthesizes a filled rectangle from the children bbox and marks it
        ``legacy_spans=True``.
        """
        child = scl.io.ScarletBlendData(origin=(10, 20), shape=(5, 6), sources={})
        legacy = scl.io.HierarchicalBlendData(children={1: child})
        data = {
            "blends": {7: legacy.as_dict()},
            "isolated": {},
            "model_type": MODEL_TYPE,
            "version": "1.0.1",
            "metadata": {},
        }
        result = _to_1_0_2(copy.deepcopy(data))
        converted = scl.io.ScarletBlendBaseData.from_dict(result["blends"][7])
        self.assertTrue(converted.legacy_spans)
        self.assertEqual(converted.span_array.shape, (5, 6))
        self.assertTrue(converted.span_array.all())
        self.assertEqual(tuple(converted.origin), (10, 20))

    def test_to_1_0_2_ignores_flat_blends(self):
        """``_to_1_0_2`` leaves non-hierarchical (flat) top-level blends
        untouched — they have no meas-specific spans to promote.
        """
        flat = scl.io.ScarletBlendData(origin=(0, 0), shape=(3, 3), sources={})
        data = {
            "blends": {7: flat.as_dict()},
            "isolated": {},
            "model_type": MODEL_TYPE,
            "version": "1.0.1",
            "metadata": {},
        }
        result = _to_1_0_2(copy.deepcopy(data))
        self.assertEqual(result["blends"][7]["blend_type"], "blend")

    def test_schema_version_constants_match(self):
        """``CURRENT_SCHEMA`` matches what's recorded as current for
        ``MODEL_TYPE`` in the migration registry.
        """
        self.assertEqual(
            CURRENT_SCHEMA,
            scl.io.migration.MigrationRegistry.current[MODEL_TYPE],
        )


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
