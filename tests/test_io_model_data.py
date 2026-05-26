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
import importlib
import unittest
from unittest import mock

import lsst.scarlet.lite as scl
import lsst.utils.tests
from lsst.meas.extensions.scarlet.io import model_data as model_data_module
from lsst.meas.extensions.scarlet.io.model_data import (
    CURRENT_SCHEMA,
    MODEL_TYPE,
    SCARLET_LITE_SCHEMA,
    _checkScarletLiteSchema,
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


class TestScarletLiteSchemaCheck(lsst.utils.tests.TestCase):
    """Tests for the scarlet_lite schema-drift safety net.

    Covers finding C-7 of the ``audits/audit-2026-05-05.md`` audit:
    a stray trailing comma packed the version-comparison operands
    into a tuple of lists, so the very mechanism designed to detect
    schema drift raised ``TypeError`` instead of the intended
    ``RuntimeError`` the first time
    ``scl.io.model_data.CURRENT_SCHEMA`` ever differed from
    ``SCARLET_LITE_SCHEMA``. The same block also compared the wrong
    pair of versions (the meas_extensions schema against the pinned
    scarlet schema, instead of the installed scarlet schema against
    the pinned one), so even with the comma dropped the check did
    not match what its error message claimed.

    The fixed helper is a bidirectional drift guard: any mismatch
    between installed and pinned schema strings fires, because an
    older installed scarlet may not emit the keys this package
    expects and a newer one may have changed them.
    """

    def test_matching_versions(self):
        """Equal scarlet and pinned schemas → no raise."""
        # Sanity check: the no-drift case must stay silent.
        _checkScarletLiteSchema("1.0.0", "1.0.0")
        _checkScarletLiteSchema("2.5.7", "2.5.7")

    def test_drift_scarlet_newer_major(self):
        """Installed scarlet ahead by a major version → RuntimeError."""
        with self.assertRaises(RuntimeError) as cm:
            _checkScarletLiteSchema("2.0.0", "1.5.9")
        # Message names the installed scarlet version so the
        # developer knows which schema to migrate to.
        self.assertIn("2.0.0", str(cm.exception))

    def test_drift_scarlet_newer_minor(self):
        """Installed scarlet ahead by a minor version → RuntimeError."""
        with self.assertRaises(RuntimeError) as cm:
            _checkScarletLiteSchema("1.1.0", "1.0.5")
        self.assertIn("1.1.0", str(cm.exception))

    def test_drift_scarlet_newer_patch(self):
        """Installed scarlet ahead by a patch version → RuntimeError."""
        with self.assertRaises(RuntimeError) as cm:
            _checkScarletLiteSchema("1.0.1", "1.0.0")
        self.assertIn("1.0.1", str(cm.exception))

    def test_drift_scarlet_older(self):
        """Installed scarlet behind the pinned version → RuntimeError.

        Pins the bidirectional semantics: an older installed
        scarlet is just as much of a drift as a newer one, because
        the keys this package's IO layer expects to read or write
        may not exist yet in the older schema.
        """
        with self.assertRaises(RuntimeError) as cm:
            _checkScarletLiteSchema("1.0.0", "1.0.1")
        self.assertIn("1.0.0", str(cm.exception))
        with self.assertRaises(RuntimeError):
            _checkScarletLiteSchema("1.0.0", "1.1.0")
        with self.assertRaises(RuntimeError):
            _checkScarletLiteSchema("0.9.9", "1.0.0")

    def test_check_wired_at_import(self):
        """The check fires at module import on a real version drift.

        Reproduces the dormant failure path of finding C-7 from the
        ``audits/audit-2026-05-05.md`` audit. Patching
        ``scl.io.model_data.CURRENT_SCHEMA`` to a newer value and
        reloading the module re-runs the import-time check; the
        original bug raised ``TypeError`` from the malformed
        ``int(list)``, while the fix raises the actionable
        ``RuntimeError`` that names the new scarlet version.
        """
        # Patch the installed-scarlet version to something newer
        # than SCARLET_LITE_SCHEMA so the drift branch fires.
        with mock.patch.object(
            scl.io.model_data, "CURRENT_SCHEMA", "9.9.9"
        ):
            with self.assertRaises(RuntimeError) as cm:
                importlib.reload(model_data_module)
        self.assertIn("9.9.9", str(cm.exception))
        # Restore the module to its real state for the rest of the
        # test session — the reload above ran against the patched
        # value but the module is now imported with the wrong (now-
        # unpatched) state. Reloading once more rebinds everything
        # to the genuine constants.
        importlib.reload(model_data_module)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
