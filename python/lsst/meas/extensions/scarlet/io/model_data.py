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

import json
import logging
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

import lsst.scarlet.lite as scl
from lsst.utils import DeprecatedDict

from .hierarchical_blend_data import LEGACY_HIERARCHICAL_TYPES, LsstHierarchicalBlendData
from .source_data import IsolatedSourceData

logger = logging.getLogger(__name__)

__all__ = ["LsstScarletModelData"]

CURRENT_SCHEMA = "1.0.3"
MODEL_TYPE = "lsst"
scl.io.migration.MigrationRegistry.set_current(MODEL_TYPE, CURRENT_SCHEMA)


class LsstScarletModelData:
    """A container that propagates scarlet models for an entire catalog,
    including isolated sources.

    This mirrors `~scarlet_lite.io.ScarletModelData` but carries information
    specific to the LSST science pipelines, and owns its schema independent
    from future changes to the scarlet_lite model.

    Notes
    -----
    The persisted LsstScarletModelData object is stored to a zip file in the
    science pipelines, where it's parameters are keys in the zip archive.
    Those utilities are out of the migration registry scope, so we cheat a
    little and package some of the attributes into a ``metadata`` dict
    only for serialization. In :meth:`__init__` we lefit those fields
    out of ``metadata`` and into typed attributes, and in :meth:`as_dict`
    we fold them back into the metadata blob.

    Attributes
    ----------
    blends
        A mapping of parent IDs to blend data.
    isolated
        A mapping of isolated source IDs to their data.
    metadata
        A dictionary of additional metadata not needed for processing.
        This is a `~lsst.utils.DeprecatedDict`: for a deprecation period
        it also exposes ``bands``, ``model_psf`` and
        ``psf`` as deprecated keys (mirrors of the typed attributes), which
        warn on access and will be removed after v31.
    bands
        The ordered band labels of the model, or ``None`` for legacy (pre-v30)
        archives that stored bands per blend rather than at the model level.
    model_psf
        The band-less model-space `~lsst.scarlet.lite.Psf` shared by all bands,
        or ``None`` if the archive carried no model-level PSF.
    psf
        The observed `~lsst.scarlet.lite.Psf` (one image per band), or ``None``
        for legacy (pre-v30) archives whose observed PSF was stored per blend.
        Reconstruction then falls back to the per-blend PSFs.
    legacy
        ``True`` when the model lacks any of the model-level attributes
        that are created and required by the current pipeline. This allows
        legacy models to load but serve as a warning that they are not
        fully compatible with the current pipeline.
    version
        The schema version of the serialized data.
    """
    model_type: str = MODEL_TYPE
    blends: dict[int, scl.io.ScarletBlendBaseData]
    isolated: dict[int, IsolatedSourceData]
    bands: tuple[str, ...] | None
    model_psf: scl.Psf | None
    psf: scl.Psf | None
    legacy: bool
    metadata: DeprecatedDict
    version: str = CURRENT_SCHEMA

    def __init__(
        self,
        isolated: dict[int, IsolatedSourceData] | None = None,
        blends: dict[int, scl.io.ScarletBlendBaseData] | None = None,
        metadata: dict[str, Any] | None = None,
        bands: tuple[str, ...] | None = None,
        model_psf: scl.Psf | None = None,
        psf: scl.Psf | None = None,
    ):
        self.blends = blends if blends is not None else {}
        self.isolated = isolated if isolated is not None else {}
        self.bands = tuple(bands) if bands is not None else None
        self.model_psf = model_psf
        self.psf = psf
        # A model missing any model-level field is a legacy (pre-v30) product
        # whose PSFs and bands lived per blend rather than at the model level.
        self.legacy = self.bands is None or self.model_psf is None or self.psf is None
        if self.legacy:
            logger.warning(
                "LsstScarletModelData is missing one or more model-level "
                "attributes (bands, model_psf, psf) required by the current "
                "pipeline. This legacy dataset can still be used for analysis, "
                "but it cannot be used for processing in the current science "
                "pipelines."
            )
        self.metadata = self._build_metadata(metadata, self.bands, model_psf, psf)

    @staticmethod
    def _build_metadata(
        metadata: dict[str, Any] | None,
        bands: tuple[str, ...] | None,
        model_psf: scl.Psf | None,
        psf: scl.Psf | None,
    ) -> DeprecatedDict:
        """Wrap ``metadata`` in a `DeprecatedDict`, mirroring the typed
        attributes as deprecated back-compat keys.

        The ``bands``, ``model_psf`` and ``psf`` keys mirror the typed
        attributes (the deprecation steers callers to those), so reading one
        from ``metadata`` returns the same object as the attribute. Any
        `~lsst.scarlet.lite.Psf` is supported, including a spatially-varying
        one that has no single image array.
        """
        data = dict(metadata) if metadata is not None else {}
        if bands is not None:
            data.setdefault("bands", tuple(bands))
        if model_psf is not None:
            data.setdefault("model_psf", model_psf)
        if psf is not None:
            data.setdefault("psf", psf)
        return DeprecatedDict(
            data,
            deprecations={
                key: (
                    f"Use the typed attribute LsstScarletModelData.{key} instead."
                )
                for key in ("bands", "model_psf", "psf")
            },
            version="v30.0",
        )

    def as_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization

        Returns
        -------
        result : dict[str, Any]
            The object encoded as a JSON-compatible dictionary.
            The mechanism for serializing to a zip file is outside of the
            migration path, so the goal is to keep the result dict relatively
            unchanged and hide new fields in the metadata blob, and extract
            them in from_dict. So we should try to keep the result keys
            as static as possible:
            - ``model_type``: The type of the model, used for dispatch in
              the migration registry.
            - ``blends``: The dictionary of blend data.
            - ``isolated``: The dictionary of isolated source data.
            - ``metadata``: The metadata blob containing additional
              information.
            - ``version``: The schema version of the serialized data.
        """
        # The PSFs and band list are load-bearing model-level fields surfaced
        # as typed attributes in memory, but they are *stored* inside the
        # transport ``metadata`` blob (each PSF as its companion data object's
        # dict, via the registry). Keeping them metadata-resident means the zip
        # IO format needs no per-field keys -- they ride along in the single
        # ``metadata`` entry. ``encode_metadata`` only special-cases ndarrays,
        # so the PSF dicts and band list pass through untouched.
        meta = dict(self.metadata) if self.metadata is not None else {}
        if self.psf is not None:
            meta["psf"] = self.psf.to_data().as_dict()
        if self.model_psf is not None:
            meta["model_psf"] = self.model_psf.to_data().as_dict()
        if self.bands is not None:
            meta["bands"] = list(self.bands)
        return {
            "model_type": MODEL_TYPE,
            "blends": {bid: b.as_dict() for bid, b in self.blends.items()},
            "isolated": {sid: s.as_dict() for sid, s in self.isolated.items()},
            "metadata": scl.io.utils.encode_metadata(meta) if meta else None,
            "version": self.version,
        }

    def json(self) -> str:
        """Serialize the data model to a JSON formatted string."""
        return json.dumps(self.as_dict())

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
        blends: dict[int, scl.io.ScarletBlendBaseData] = {}
        for bid, blend in data.get("blends", {}).items():
            if "blend_type" not in blend:
                # Default to a flat blend for legacy data.
                blend["blend_type"] = "blend"
            try:
                blends[int(bid)] = scl.io.ScarletBlendBaseData.from_dict(blend, dtype=dtype)
            except KeyError:
                raise scl.io.utils.PersistenceError(
                    f"Unknown blend type: {blend['blend_type']} for blend ID: {bid}"
                )
        isolated: dict[int, IsolatedSourceData] = {}
        for sid, source_data in data.get("isolated", {}).items():
            isolated[int(sid)] = IsolatedSourceData.from_dict(source_data, dtype=dtype)
        metadata = scl.io.utils.decode_metadata(data.get("metadata", None)) or {}
        # The PSFs and bands are stored inside the transport metadata (legacy
        # archives have them rewritten into this shape by the migration chain).
        # Lift them back out into typed attributes, leaving only residual
        # metadata behind so the attributes stay the single source of truth.
        # Pre-v30 (DP1) archives have no model-level observed ``psf`` or
        # ``bands`` (those were stored per blend), so these stay ``None`` and
        # reconstruction falls back to the per-blend PSFs.
        psf_data = metadata.pop("psf", None)
        psf = (
            scl.io.PsfBaseData.from_dict(psf_data, dtype=dtype).to_psf()
            if psf_data is not None else None
        )
        model_psf_data = metadata.pop("model_psf", None)
        model_psf = (
            scl.io.PsfBaseData.from_dict(model_psf_data, dtype=dtype).to_psf()
            if model_psf_data is not None else None
        )
        bands_data = metadata.pop("bands", None)
        bands = tuple(bands_data) if bands_data is not None else None
        return cls(
            isolated=isolated,
            blends=blends,
            metadata=metadata or None,
            psf=psf,
            model_psf=model_psf,
            bands=bands,
        )

    @classmethod
    def parse_obj(cls, data: dict) -> LsstScarletModelData:
        """Construct from a python-decoded JSON object (``json.load``)."""
        return cls.from_dict(data, dtype=np.float32)


@scl.io.migration.migration(MODEL_TYPE, scl.io.migration.PRE_SCHEMA)
def _to_1_0_0(data: dict) -> dict:
    """Migrate a pre-schema model to schema version 1.0.0.

    Pre-``metadata`` (v29) archives are legacy base
    `~lsst.scarlet.lite.io.ScarletModelData` products: an
    `LsstScarletModelData` with model-level typed PSF/band fields did not
    exist yet. They stored only the *model* PSF at the model level (top-level
    ``psf`` / ``psfShape`` entries) and kept the observed PSF and bands *per
    blend* (`ScarletBlendData` carried ``psf`` / ``bands`` / ``psf_center``),
    with no single catalog-wide observed PSF.

    The top-level model PSF is promoted into the modern ``metadata`` shape,
    so ``decode_metadata`` can reconstruct the array via ``array_keys``.
    There is no model-level observed ``psf`` or ``bands`` to recover, so they
    stay unset (``LsstScarletModelData.psf`` / ``.bands`` are ``None`` for
    such archives) and reconstruction falls back to the per-blend PSFs.

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
    # Pre-``metadata`` archives stored the model PSF as top-level ``psf`` /
    # ``psfShape`` entries. Promote them into the modern ``metadata`` shape so
    # ``decode_metadata`` can reconstruct the array via ``array_keys``. Mirrors
    # scarlet_lite's pre-schema ``scarlet_model`` migration.
    if "metadata" not in data and "psfShape" in data:
        data["metadata"] = {
            "model_psf": data.pop("psf"),
            "model_psf_shape": data.pop("psfShape"),
            "array_keys": ["model_psf"],
        }
    data["version"] = "1.0.0"
    return data


@scl.io.migration.migration(MODEL_TYPE, "1.0.0")
def _to_1_0_1(data: dict) -> dict:
    """Migrate a schema version 1.0.0 model to schema version 1.0.1

    There were no changes to this data model in v1.0.1 but we need
    to provide a way to migrate 1.0.0 data.

    Parameters
    ----------
    data : dict
        The data to migrate.
    Returns
    -------
    result : dict
        The migrated data.
    """
    data["version"] = "1.0.1"
    if data.get("metadata") is None:
        data["metadata"] = {}
    data["metadata"].setdefault("footprint", None)
    return data


@scl.io.migration.migration(MODEL_TYPE, "1.0.1")
def _to_1_0_2(data: dict) -> dict:
    """Migrate a schema version 1.0.1 model to schema version 1.0.2.

    1.0.1 (and earlier) stored top-level parent blends as scarlet_lite
    ``HierarchicalBlendData`` with the detected-parent footprint
    (``spans``/``origin``) in the blend ``metadata`` dict. 1.0.2 promotes each
    to ``LsstHierarchicalBlendData``, where those fields are typed attributes.

    Each legacy blend is handed to
    ``LsstHierarchicalBlendData.convert_from_hierarchical``. This runs before
    blend dispatch in ``from_dict``, so the rewritten dicts deserialize
    directly to the new class.

    Parameters
    ----------
    data : dict
        The data to migrate.

    Returns
    -------
    result : dict
        The migrated data.
    """
    blends = data.get("blends", {})
    for bid, blend in list(blends.items()):
        if blend.get("blend_type", "blend") in LEGACY_HIERARCHICAL_TYPES:
            blends[bid] = LsstHierarchicalBlendData.convert_from_hierarchical(blend)
    data["version"] = "1.0.2"
    return data


@scl.io.migration.migration(MODEL_TYPE, "1.0.2")
def _to_1_0_3(data: dict) -> dict:
    """Migrate a schema version 1.0.2 model to schema version 1.0.3.

    1.0.2 (and earlier) stored the model and observed PSFs inside the
    transported ``metadata`` dict as raw ``array_keys``-encoded arrays. 1.0.3
    keeps the PSFs (and the ``bands`` list) metadata-resident, but rewrites
    each PSF array *in place* into its `ImagePsfData` dict form so it
    round-trips through the PSF registry like a modern model. ``bands`` is a
    plain list, so it is left untouched. The migration is the one place where
    reading the legacy ``array_keys`` PSF arrays is correct.

    The migration runs before ``decode_metadata``, so each PSF array is still
    in its encoded ``<key>`` / ``<key>_shape`` / ``<key>_dtype`` form.

    Parameters
    ----------
    data : dict
        The data to migrate.

    Returns
    -------
    result : dict
        The migrated data.
    """
    metadata = data.get("metadata")
    if metadata:
        array_keys = list(metadata.get("array_keys", []))
        bands = list(metadata.get("bands", []))
        for key in ("psf", "model_psf"):
            if key not in metadata or key not in array_keys:
                continue
            shape = metadata.pop(f"{key}_shape", None)
            if shape is None:
                shape = metadata.pop(f"{key}Shape", None)
            dtype = metadata.pop(f"{key}_dtype", "float32")
            # The model PSF is band-less; normalize a 2D legacy array to a
            # ``(1, height, width)`` broadcast cube. The observed PSF carries
            # the model bands.
            if key == "model_psf" and shape is not None and len(shape) == 2:
                shape = [1, *shape]
            metadata[key] = {
                "psf_type": "image",
                "bands": bands if key == "psf" else [],
                "padding": 3,
                "version": "1.0.0",
                "dtype": dtype,
                "shape": shape,
                "data": metadata[key],
            }
            array_keys = [k for k in array_keys if k != key]
        if array_keys:
            metadata["array_keys"] = array_keys
        else:
            metadata.pop("array_keys", None)
    data["version"] = "1.0.3"
    return data
