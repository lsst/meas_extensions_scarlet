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
import warnings

import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.meas.extensions.scarlet.io.model_data import LsstScarletModelData
from lsst.meas.extensions.scarlet.io.source_data import IsolatedSourceData
from lsst.pipe.base import NoWorkFound

import pipeline
from scenes import SCENES


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


class TestScarletModelToLsstScarletModel(lsst.utils.tests.TestCase):
    """Tests for ``scarlet_model_to_lsst_scarlet_model``.

    The converter wraps a base ``ScarletModelData`` in an
    ``LsstScarletModelData``. Previously it dropped the input's
    ``metadata`` and substituted ``None``, which then crashed the
    ``_to_1_0_1`` migration on any subsequent round-trip (see finding
    IO-3 of ``audits/audit-2026-05-05.md``).
    """

    @staticmethod
    def _base_model(metadata):
        return scl.io.ScarletModelData(blends={}, metadata=metadata)

    def test_propagates_metadata_when_present(self):
        """The returned ``LsstScarletModelData`` carries the same
        ``metadata`` dict as the source, not ``None``.
        """
        source_metadata = {"bands": ("g", "r"), "model_psf": "placeholder"}
        result = mes.io.utils.scarlet_model_to_lsst_scarlet_model(
            self._base_model(source_metadata)
        )
        self.assertEqual(result.metadata, source_metadata)

    def test_defaults_to_empty_dict_not_none(self):
        """When the source's ``metadata`` is ``None``, the converter
        substitutes an empty dict so the ``_to_1_0_1`` migration sees
        a real dict it can call ``setdefault`` on.
        """
        result = mes.io.utils.scarlet_model_to_lsst_scarlet_model(
            self._base_model(None)
        )
        self.assertEqual(result.metadata, {})


class TestScarletModelDelegate(lsst.utils.tests.TestCase):
    """Tests for
    ``lsst.meas.extensions.scarlet.io.utils.ScarletModelDelegate.handleParameters``.

    The delegate hooks ``LsstScarletModelData`` into Butler's
    parameter-application pipeline. The base ``StorageClassDelegate``
    annotates ``parameters`` as ``Mapping[str, Any] | None`` and
    treats ``None``/``{}`` as "no parameters" — its dispatch path in
    ``daf_butler.datastore.generic_base.post_process_get`` already
    short-circuits with ``if assemblerParams:``, so the bug below is
    latent in current usage, but in-memory-datastore and
    disassembled-composite reads pass ``None``/``{}`` straight
    through, and the delegate must agree with the base class on
    those values.
    """

    @staticmethod
    def _model_with_blends():
        # ``handleParameters`` only inspects the keys of
        # ``inMemoryDataset.blends`` (to slice the dict), so the
        # values can be arbitrary sentinels — no need to build real
        # ScarletBlendData objects.
        return LsstScarletModelData(
            blends={1: "blend-1", 2: "blend-2", 3: "blend-3"},
        )

    @staticmethod
    def _delegate():
        # ``StorageClassDelegate.__init__`` requires a
        # ``StorageClass`` argument, but ``handleParameters`` never
        # consults it; a sentinel is enough to construct the
        # delegate in isolation from a real Butler.
        from unittest.mock import Mock
        return mes.io.ScarletModelDelegate(storageClass=Mock())

    def test_handleParameters_none_returns_unchanged(self):
        """``parameters=None`` returns the dataset unchanged.

        Regression test for finding IO-6 of the
        ``audits/audit-2026-05-05.md`` audit. The bug was
        ``"blend_id" in None`` raising ``TypeError`` — the base-class
        contract permits ``None`` and treats it as "no parameters",
        so the delegate must do the same.
        """
        model = self._model_with_blends()
        delegate = self._delegate()

        result = delegate.handleParameters(model, None)

        self.assertIs(result, model)
        self.assertEqual(set(result.blends.keys()), {1, 2, 3})

    def test_handleParameters_empty_dict_returns_unchanged(self):
        """``parameters={}`` also returns unchanged.

        Mirrors ``StorageClassDelegate.handleParameters`` whose
        ``if parameters:`` guard treats the empty dict as "no
        parameters" rather than as "unsupported parameters". The
        pre-fix delegate raised ``ValueError("Unsupported parameters:
        {}")`` here because the ``elif parameters is not None`` branch
        fired on an empty dict. Companion to the ``None`` case under
        finding IO-6 of ``audits/audit-2026-05-05.md``.
        """
        model = self._model_with_blends()
        delegate = self._delegate()

        result = delegate.handleParameters(model, {})

        self.assertIs(result, model)
        self.assertEqual(set(result.blends.keys()), {1, 2, 3})

    def test_handleParameters_blend_id_filters(self):
        """``parameters={'blend_id': ...}`` keeps only the requested
        blends.

        Pins the filtering branch under finding IO-6 of the
        ``audits/audit-2026-05-05.md`` audit: the no-parameter fixes
        above must not regress the actual partial-load path.
        """
        model = self._model_with_blends()
        delegate = self._delegate()

        result = delegate.handleParameters(model, {"blend_id": 2})

        self.assertIs(result, model)
        self.assertEqual(set(result.blends.keys()), {2})
        # Iterable forms also work.
        model = self._model_with_blends()
        result = delegate.handleParameters(model, {"blend_id": [1, 3]})
        self.assertEqual(set(result.blends.keys()), {1, 3})

    def test_handleParameters_unsupported_raises(self):
        """Non-empty parameters without ``blend_id`` raise
        ``ValueError``.

        Pins the rejection branch under finding IO-6 of
        ``audits/audit-2026-05-05.md`` so a future relaxation of the
        no-parameter case does not silently start accepting
        unrecognized keys.
        """
        model = self._model_with_blends()
        delegate = self._delegate()

        with self.assertRaises(ValueError) as cm:
            delegate.handleParameters(model, {"something_else": 42})
        self.assertIn("something_else", str(cm.exception))


class TestMonochromaticDataToScarletDeprecation(lsst.utils.tests.TestCase):
    """Coverage retention for the deprecated
    ``monochromaticDataToScarlet`` (scheduled for removal after v31).

    The function is no longer called by any production code path in
    this package, but its public-API contract is preserved for external
    callers until removal. These tests keep the safety net by
    exercising it directly on a synthetic blend.
    """

    @staticmethod
    def _toy_blend_data():
        # One factorized component, distinct (y, x) peak so any silent
        # axis swap would be loud. Bands chosen so the round-trip
        # exercises the real-band → ("dummy",) projection.
        bands = ("g", "r", "i")
        h, w = 6, 8
        origin = (10, 20)
        peak = (12, 25)
        spectrum = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        morph = np.ones((h, w), dtype=np.float32)
        component_data = scl.io.ScarletFactorizedComponentData(
            origin=origin,
            peak=peak,
            spectrum=spectrum,
            morph=morph,
        )
        source_data = scl.io.ScarletSourceData(components=[component_data])
        blend_data = scl.io.ScarletBlendData(
            origin=origin,
            shape=(h, w),
            sources={42: source_data},
        )
        return bands, blend_data, peak

    def test_monochromatic_data_to_scarlet_emits_future_warning(self):
        """``monochromaticDataToScarlet`` emits a ``FutureWarning``
        flagging the migration to
        ``ScarletBlendData.minimal_data_to_blend(...)[band]``.

        Pinned so the deprecation stays visible until the function is
        removed after v31.
        """
        bands, blend_data, _peak = self._toy_blend_data()
        bbox = scl.Box(blend_data.shape, origin=blend_data.origin)
        observation = scl.Observation.empty(
            bands=("dummy",),
            psfs=np.ones((1, 5, 5), dtype=np.float32),
            model_psf=np.ones((1, 5, 5), dtype=np.float32),
            bbox=bbox,
            dtype=np.float32,
        )

        with self.assertWarns(FutureWarning):
            blend = mes.io.monochromaticDataToScarlet(
                blendData=blend_data, bandIndex=1, observation=observation,
            )

        self.assertEqual(len(blend.sources), 1)

    def test_monochromatic_band_constants_emit_future_warning(self):
        """Module-level ``monochromaticBand`` / ``monochromaticBands``
        access fires a ``FutureWarning`` and resolves to the original
        ``"dummy"`` placeholder.

        Pinned alongside the function deprecation so the constants and
        the function are removed together after v31.
        """
        from lsst.meas.extensions.scarlet.io import utils as io_utils

        with self.assertWarns(FutureWarning):
            self.assertEqual(io_utils.monochromaticBand, "dummy")
        with self.assertWarns(FutureWarning):
            self.assertEqual(io_utils.monochromaticBands, ("dummy",))

    def test_monochromatic_data_to_scarlet_preserves_factorized_peak(self):
        """``monochromaticDataToScarlet`` returns a source whose
        component peak matches the persisted ``(y, x)``.

        The deprecated function is the only remaining write-target for
        the legacy ``("dummy",)`` per-band reconstruction; pinning the
        peak guards against regressions in the
        ``FactorizedComponent`` rebuild path while it lives.
        """
        bands, blend_data, peak = self._toy_blend_data()
        bbox = scl.Box(blend_data.shape, origin=blend_data.origin)
        observation = scl.Observation.empty(
            bands=("dummy",),
            psfs=np.ones((1, 5, 5), dtype=np.float32),
            model_psf=np.ones((1, 5, 5), dtype=np.float32),
            bbox=bbox,
            dtype=np.float32,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            blend = mes.io.monochromaticDataToScarlet(
                blendData=blend_data, bandIndex=1, observation=observation,
            )

        self.assertEqual(blend.sources[0].components[0].peak, peak)


class TestLoadBlend(lsst.utils.tests.TestCase):
    """Tests for ``loadBlend``.

    ``loadBlend`` reconstructs a single per-blend scarlet ``Blend``
    object from a persisted ``ScarletBlendData`` and a multiband
    coadd. The legacy signature took ``model_psf`` directly and
    computed per-band PSFs from the coadd; the new signature accepts
    the full ``modelData`` and uses its already-fit PSFs verbatim,
    which is both a more faithful round-trip and the only path that
    works against modern persistence (legacy fields like
    ``psf_center`` were dropped from ``ScarletBlendData`` during the
    scarlet_lite refactor).
    """

    def _bundle(self):
        # ``pipeline.deblend`` is memoized per (scene, config) so this
        # is effectively a free lookup after the first invocation.
        return pipeline.deblend(
            pipeline.deconvolve(
                pipeline.detect(pipeline.build_image(SCENES["multi-blend"]))
            )
        )

    def _leaf_blend(self, modelData):
        # Return the first leaf ``ScarletBlendData`` (i.e. the first
        # child of the first hierarchical parent). ``loadBlend``'s
        # contract is per-leaf, not per-parent.
        for parent_blend in modelData.blends.values():
            for child in parent_blend.children.values():
                if isinstance(child, scl.io.ScarletBlendData):
                    return child
        self.fail("multi-blend scene unexpectedly produced no leaf blends")

    def test_loadBlend_modelData_uses_metadata_model_psf(self):
        """``loadBlend(..., modelData=...)`` builds an observation
        whose ``model_psf`` equals ``modelData.metadata['model_psf']``.

        The previous signature derived its PSFs from the coadd at the
        blend's ``psf_center``, which both required attributes
        ``ScarletBlendData`` no longer carries and re-derived a PSF
        that may not match the one used during the fit. Pinning the
        observation's ``model_psf`` to the metadata value guards the
        intended round-trip.
        """
        bundle = self._bundle()
        modelData = bundle.result.scarletModelData
        blendData = self._leaf_blend(modelData)

        blend, _ = mes.io.loadBlend(
            blendData, mCoadd=bundle.image.mCoadd, modelData=modelData,
        )

        np.testing.assert_array_equal(
            blend.observation.model_psf[0],
            modelData.metadata["model_psf"],
        )

    def test_loadBlend_model_psf_emits_future_warning(self):
        """Passing ``model_psf`` emits a ``FutureWarning`` flagging the
        removal scheduled for v31.

        The legacy path may still raise downstream (modern
        ``ScarletBlendData`` lacks the ``psf_center`` attribute it
        once required), so the assertion only pins the warning — any
        post-warning exception is swallowed.
        """
        bundle = self._bundle()
        modelData = bundle.result.scarletModelData
        blendData = self._leaf_blend(modelData)
        model_psf = modelData.metadata["model_psf"]

        with self.assertWarns(FutureWarning):
            try:
                mes.io.loadBlend(
                    blendData,
                    model_psf=model_psf,
                    mCoadd=bundle.image.mCoadd,
                )
            except Exception:
                pass

    def test_loadBlend_requires_psf_source(self):
        """Calling ``loadBlend`` with ``mCoadd`` but neither
        ``modelData`` nor ``model_psf`` raises ``ValueError``.

        Pins the precondition that some PSF source has to be passed,
        rather than silently constructing a degenerate observation.
        """
        bundle = self._bundle()
        modelData = bundle.result.scarletModelData
        blendData = self._leaf_blend(modelData)

        with self.assertRaises(ValueError):
            mes.io.loadBlend(blendData, mCoadd=bundle.image.mCoadd)

    def test_loadBlend_requires_mCoadd(self):
        """Calling ``loadBlend`` without ``mCoadd`` raises
        ``ValueError``.

        ``mCoadd`` has no default value at the API level (the
        signature uses ``None`` only to support the legacy positional
        order); omitting it should be a loud error rather than an
        ``AttributeError`` deep inside the observation construction.
        """
        bundle = self._bundle()
        modelData = bundle.result.scarletModelData
        blendData = self._leaf_blend(modelData)

        with self.assertRaises(ValueError):
            mes.io.loadBlend(blendData, modelData=modelData)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
