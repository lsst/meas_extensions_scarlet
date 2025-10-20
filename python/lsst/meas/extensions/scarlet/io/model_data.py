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

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import DTypeLike

import lsst.scarlet.lite as scl

from .source_data import IsolatedSourceData

CURRENT_SCHEMA = "1.0.0"
SCARLET_LITE_SCHEMA = "1.0.0"
MODEL_TYPE = "lsst"
scl.io.migration.MigrationRegistry.set_current(MODEL_TYPE, CURRENT_SCHEMA)

# Ensure that the ScarletModelData from scarlet lite hasn't changed.
if scl.io.model_data.CURRENT_SCHEMA != SCARLET_LITE_SCHEMA:
    raise RuntimeError(
        "Version mismatch between meas_extensions_scarlet and scarlet lite. "
        "This requires updating SCARLET_LITE_SCHEMA, CURRENT_SCHEMA, and a migration step "
        f"to match the ScarletModelData schema version {scl.io.model_data.CURRENT_SCHEMA}."
    )


class LsstScarletModelData(scl.io.ScarletModelData):
    """A ScarletModelData that includes isolated sources.

    Attributes
    ----------
    isolated : dict[int, IsolatedSourceData]
        A mapping of isolated source IDs to their data.
    version : dict[int, scl.io.ScarletBlendBaseData]
        The schema version of the serialized data.
    """
    isolated: dict[int, IsolatedSourceData]
    version: str = CURRENT_SCHEMA

    def __init__(
        self,
        isolated: dict[int, IsolatedSourceData] | None = None,
        blends: dict[int, scl.io.ScarletBlendBaseData] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(blends=blends, metadata=metadata)
        self.isolated = isolated if isolated is not None else {}

    def as_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization

        Returns
        -------
        result : dict[str, Any]
            The object encoded as a JSON-compatible dictionary.
        """
        data = super().as_dict()
        # Ensure that the we are not trying to serialize with an
        # incompatible schema.
        if data["version"] != SCARLET_LITE_SCHEMA:
            raise RuntimeError(
                f"Cannot serialize scarlet model with schema {data['version']}, "
                f"expected {SCARLET_LITE_SCHEMA}. "
                "This requires updating SCARLET_LITE_SCHEMA, CURRENT_SCHEMA, and a migration step."
            )
        data.update(
            {
                "isolated": {k: v.as_dict() for k, v in self.isolated.items()},
                "version": self.version,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> LsstScarletModelData:
        """Reconstruct `LsstScarletModelData` from JSON compatible dict.

        Parameters
        ----------
        data : dict
            Dictionary representation of the object
        dtype : DTypeLike
            Datatype of the resulting model.

        Returns
        -------
        result : LsstScarletModelData
            The reconstructed object
        """
        data = scl.io.migration.MigrationRegistry.migrate(MODEL_TYPE, data)
        isolated: dict[int, IsolatedSourceData] = {}
        for sid, source_data in data.get("isolated", {}).items():
            isolated[int(sid)] = IsolatedSourceData.from_dict(source_data, dtype=dtype)
        if "metadata" not in data:
            data["metadata"] = None
        return super().from_dict(data, dtype=dtype, isolated=isolated)


@scl.io.migration.migration(MODEL_TYPE, scl.io.migration.PRE_SCHEMA)
def _to_1_0_0(data: dict) -> dict:
    """Migrate a pre-schema model to schema version 1.0.0

    There were no changes to this data model in v1.0.0 but we need
    to provide a way to migrate pre-schema data.

    Parameters
    ----------
    data : dict
        The data to migrate.
    Returns
    -------
    result : dict
        The migrated data.
    """
    # Ensure that the model type and version are set and add an
    # empty isolated sources dictionary.
    if "model_type" not in data:
        data["model_type"] = MODEL_TYPE
    data["isolated"] = {}
    data["version"] = "1.0.0"
    return data
