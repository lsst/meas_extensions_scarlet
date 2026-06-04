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

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

import lsst.scarlet.lite as scl
from lsst.scarlet.lite import Box

from .source_data import _decode_span_array, _encode_span_array

__all__ = ["LsstHierarchicalBlendData"]

CURRENT_SCHEMA = "1.0.0"
BLEND_TYPE = "lsst_hierarchical"
scl.io.migration.MigrationRegistry.set_current(BLEND_TYPE, CURRENT_SCHEMA)

# `LsstHierarchicalBlendData` superceeds scarlet_lite's
# `HierarchicalBlendData`. We keep track of the legacy types so that we can
# migrate them to the new type when reading from disk.
LEGACY_HIERARCHICAL_TYPES = ("hierarchical", "hierarchical_blend")


@dataclass(kw_only=True)
class LsstHierarchicalBlendData(scl.io.ScarletBlendBaseData):
    """A meas-owned hierarchical blend that carries the detection footprint.

    The LSST-pipeline replacement for scarlet_lite's
    `~lsst.scarlet.lite.io.HierarchicalBlendData`, with attributes specific
    to the LSST science pipelines needs.

    Attributes
    ----------
    children
        Map from blend IDs to child blends.
    span_array
        The detected-parent footprint mask (``True`` inside the footprint).
    origin
        The ``(y, x)`` origin of ``span_array`` in observation coordinates.
    legacy_spans
        ``True`` when ``span_array`` was reconstructed by a migration rather
        than carried from the original detection. In that case the
        spans are *not* the true detection footprint, but approximated
        by the union of child footprints if possible. If not then the
        spans are a filled rectangle over the child bounding boxes.
    """

    blend_type: str = BLEND_TYPE
    version: str = CURRENT_SCHEMA
    children: dict[int, scl.io.ScarletBlendBaseData]
    span_array: np.ndarray
    origin: tuple[int, int]
    legacy_spans: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        """The ``(height, width)`` of the footprint span mask."""
        return self.span_array.shape[0], self.span_array.shape[1]

    @property
    def bbox(self) -> Box:
        """The bounding box of the detected-parent footprint."""
        return Box(self.span_array.shape, origin=self.origin)

    def as_dict(self) -> dict[str, Any]:
        """Return the object encoded into a dict for JSON serialization.

        Returns
        -------
        result : dict[str, Any]
            The object encoded as a JSON-compatible dict.
        """
        result: dict[str, Any] = {
            "blend_type": self.blend_type,
            "children": {bid: child.as_dict() for bid, child in self.children.items()},
            "span_array": _encode_span_array(self.span_array),
            "shape": self.shape,
            "origin": tuple(int(o) for o in self.origin),
            "legacy_spans": bool(self.legacy_spans),
            "version": self.version,
        }
        if self.metadata is not None:
            result["metadata"] = scl.io.utils.encode_metadata(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> LsstHierarchicalBlendData:
        """Reconstruct `LsstHierarchicalBlendData` from a JSON compatible dict.

        Parameters
        ----------
        data : dict
            Dictionary representation of the object.
        dtype : DTypeLike
            Datatype of the resulting model.

        Returns
        -------
        result : LsstHierarchicalBlendData
            The reconstructed object.
        """
        data = scl.io.migration.MigrationRegistry.migrate(BLEND_TYPE, data)
        children: dict[int, scl.io.ScarletBlendBaseData] = {}
        for blend_id, child in data["children"].items():
            try:
                children[int(blend_id)] = scl.io.ScarletBlendBaseData.from_dict(child, dtype=dtype)
            except KeyError:
                raise scl.io.utils.PersistenceError(
                    f"Unknown blend type: {child.get('blend_type')} for blend ID: {blend_id}"
                )
        shape = tuple(int(s) for s in data["shape"])
        # Spans are a boolean mask, unaffected by the model ``dtype``.
        span_array = _decode_span_array(data["span_array"], shape, bool)
        origin = tuple(int(o) for o in data["origin"])
        legacy_spans = bool(data.get("legacy_spans", False))
        metadata = scl.io.utils.decode_metadata(data.get("metadata", None))
        return cls(
            children=children,
            span_array=span_array,
            origin=origin,
            legacy_spans=legacy_spans,
            metadata=metadata,
        )

    @classmethod
    def convert_from_hierarchical(cls, blend_dict: dict) -> dict:
        """Promote a legacy scarlet_lite ``hierarchical`` blend dict to the
        ``lsst_hierarchical`` shape.

        Reads the legacy on-disk format directly (not through scarlet_lite's
        ``hierarchical`` migration chain, so it is unaffected by future
        ``HierarchicalBlendData`` changes in scarlet_lite).
        ``children`` dicts pass through untouched and migrate individually
        in :meth:`from_dict`.

        Real ``metadata["spans"]`` are promoted verbatim
        (``legacy_spans=False``).
        When absent, a filled rectangle over the children's bounding box is
        synthesized with ``legacy_spans=True``.

        Parameters
        ----------
        blend_dict : dict
            A legacy scarlet_lite ``hierarchical`` blend dict.

        Returns
        -------
        result : dict
            An ``lsst_hierarchical`` blend dict ready for
            :meth:`from_dict` dispatch.
        """
        children = blend_dict["children"]
        metadata = scl.io.utils.decode_metadata(blend_dict.get("metadata", None))

        if metadata is not None and "spans" in metadata:
            span_array = np.asarray(metadata["spans"], dtype=bool)
            origin = tuple(int(o) for o in metadata["origin"])
            legacy_spans = False
            residual = {k: v for k, v in metadata.items() if k not in ("spans", "origin")}
        else:
            origin, shape = cls._bbox_from_children(children)
            span_array = np.ones(shape, dtype=bool)
            legacy_spans = True
            residual = dict(metadata) if metadata else {}

        result: dict[str, Any] = {
            "blend_type": BLEND_TYPE,
            "version": CURRENT_SCHEMA,
            "children": children,
            "span_array": _encode_span_array(span_array),
            "shape": tuple(int(s) for s in span_array.shape),
            "origin": tuple(int(o) for o in origin),
            "legacy_spans": legacy_spans,
        }
        if residual:
            result["metadata"] = scl.io.utils.encode_metadata(residual)
        return result

    @staticmethod
    def _bbox_from_children(children: dict) -> tuple[tuple[int, int], tuple[int, int]]:
        """Compute the ``(origin, shape)`` covering all child blend dicts.

        This should only be used by the legacy-spans synthesis path,
        which supports only the flat `~lsst.scarlet.lite.io.ScarletBlendData`
        children the deblender produces.

        Raises
        ------
        ValueError
            If ``children`` is empty.
        """
        if not children:
            raise ValueError(
                "Cannot synthesize legacy spans for a hierarchical blend with no children."
            )
        origins = []
        shapes = []
        for child in children.values():
            if "origin" not in child or "shape" not in child:
                raise NotImplementedError(
                    "Legacy span synthesis only supports flat child blends with "
                    f"'origin'/'shape'; got blend_type {child.get('blend_type')!r}."
                )
            origins.append(tuple(int(o) for o in child["origin"]))
            shapes.append(tuple(int(s) for s in child["shape"]))
        min_y = min(o[0] for o in origins)
        min_x = min(o[1] for o in origins)
        max_y = max(o[0] + s[0] for o, s in zip(origins, shapes))
        max_x = max(o[1] + s[1] for o, s in zip(origins, shapes))
        origin = (min_y, min_x)
        shape = (max_y - min_y, max_x - min_x)
        return origin, shape


LsstHierarchicalBlendData.register()
