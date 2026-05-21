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

import lsst.scarlet.lite as scl
import lsst.utils.tests
from lsst.meas.extensions.scarlet.io.model_data import (
    CURRENT_SCHEMA,
    MODEL_TYPE,
    SCARLET_LITE_SCHEMA,
    _to_1_0_0,
    _to_1_0_1,
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

    def test_schema_version_constants_match(self):
        """The schema constants line up with what's actually registered
        and with the scarlet_lite version installed.

        - ``SCARLET_LITE_SCHEMA`` is the scarlet_lite schema this
          package was last verified against; it must match
          ``scl.io.model_data.CURRENT_SCHEMA`` (the version of
          scarlet_lite actually installed). A drift here is what
          the module's import-time check raises on.
        - ``CURRENT_SCHEMA`` is the current
          ``meas_extensions_scarlet`` model schema; it must match
          what's recorded as current for ``MODEL_TYPE`` in the
          migration registry.
        """
        self.assertEqual(
            SCARLET_LITE_SCHEMA, scl.io.model_data.CURRENT_SCHEMA
        )
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
