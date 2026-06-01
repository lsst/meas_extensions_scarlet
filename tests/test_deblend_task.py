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

"""Tests for ``ScarletDeblendTask``.

Exercises the deblend task on the cached ``multi-blend`` scene from
``pipeline.py``: catalog structure, heavy-footprint attachment and
model recovery, and skip / failure semantics. The skip tests run
against targeted single-blend scenes.
"""

import unittest
from unittest.mock import patch

import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np
from lsst.afw.detection import PeakTable
from lsst.afw.geom import SpanSet
from lsst.afw.table import Schema
from lsst.geom import Point2I
from lsst.meas.extensions.scarlet.scarletDeblendTask import (
    ScarletDeblendContext,
    ScarletDeblendTask,
    deblend,
)

import pipeline
from scenes import SCENES


class TestDeblendTask(lsst.utils.tests.TestCase):
    """Tests for ``ScarletDeblendTask`` in
    ``lsst.meas.extensions.scarlet.scarletDeblendTask``.

    Exercises the deblend task's catalog structure, heavy-footprint
    attachment / model recovery, and skip / failure semantics. The
    ``multi-blend`` scene from ``pipeline.py`` is shared across the
    catalog-structure and heavy-footprint tests; the skip tests run
    on targeted single-blend scenes.
    """

    def _deblend(self, scene, config=None):
        # Run the cached pipeline through deblend on a given scene
        # (with an optional non-default deblend config).
        # ``pipeline.deblend`` memoizes by (scene, configs), so repeated
        # calls across tests with the same arguments reuse the bundle.
        image = pipeline.build_image(scene)
        detection = pipeline.detect(image)
        deconv = pipeline.deconvolve(detection)
        return pipeline.deblend(deconv, config=config)

    def _attach_band_footprints(self, bundle, band, useFlux):
        # Hydrate every child row in the bundle's catalog with the
        # HeavyFootprint for one (band, useFlux) combination.
        image = bundle.image
        imageForRedistribution = image.mCoadd[band] if useFlux else None
        mes.io.updateCatalogFootprints(
            bundle.result.scarletModelData,
            bundle.result.deblendedCatalog,
            band=band,
            imageForRedistribution=imageForRedistribution,
            removeScarletData=False,
            updateFluxColumns=True,
        )

    def _iter_multipeak_children(self, bundle):
        # Yield ``(parent, child)`` for every child row whose parent
        # is a top-level multi-peak source — i.e. the rows that came
        # from the scarlet deblend rather than from an isolated parent.
        catalog = bundle.result.deblendedCatalog
        objectParents = bundle.result.objectParents
        parents = objectParents[
            (objectParents["parent"] == 0) & (objectParents["deblend_nPeaks"] > 1)
        ]
        for parent in parents:
            for child in catalog[catalog["parent"] == parent.get("id")]:
                yield parent, child

    def _scarlet_blend_for_child(self, bundle, parent, child, band):
        # Reconstruct the per-band scarlet blend and pick out the
        # source matching ``child.getId()``. Returns the blend (whose
        # ``observation`` can be rebound for flux redistribution), the
        # source, and the parent's afw footprint.
        modelData = bundle.result.scarletModelData
        parentBlendData = modelData.blends[parent.getId()]
        parentFootprint = parent.getFootprint()

        blendData = parentBlendData.children[child["deblend_blendId"]]
        full_blend = blendData.minimal_data_to_blend(
            model_psf=modelData.metadata["model_psf"][None, :, :],
            psf=modelData.metadata["psf"],
            bands=modelData.metadata["bands"],
        )
        blend = full_blend[band]
        source = next(
            src for src in blend.sources if src.metadata["id"] == child.getId()
        )
        return blend, source, parentFootprint

    def test_skip_too_big(self):
        """A parent footprint exceeding ``maxFootprintArea`` is skipped
        with the ``deblend_skipped_parentTooBig`` flag set.

        Uses the ``large_two_sersic`` scene (single parent, two large
        overlapping Sersics) with ``maxFootprintArea=2000``; the
        deconvolved footprint exceeds that limit so the parent is
        skipped.
        """
        config = ScarletDeblendTask.ConfigClass()
        config.maxFootprintArea = 2000
        config.catchFailures = False

        bundle = self._deblend(SCENES["large_two_sersic"], config=config)

        catalog = bundle.result.objectParents
        parents = catalog[catalog["parent"] == 0]
        self.assertEqual(len(parents), 1)
        parent = parents[0]
        self.assertTrue(parent.get("deblend_skipped"))
        self.assertTrue(parent.get("deblend_skipped_parentTooBig"))
        self.assertFalse(parent.get("deblend_skipped_tooManyPeaks"))

    def test_skip_too_many_peaks(self):
        """A parent with more than ``maxNumberOfPeaks`` peaks is
        skipped with the ``deblend_skipped_tooManyPeaks`` flag set.

        Uses the ``three_source_blend`` scene (single parent, three
        peaks) with ``maxNumberOfPeaks=2``.

        Also pins that a skipped blend leaves
        ``deblend_blendConvergenceFailedFlag`` unset: it was never fit,
        so convergence does not apply (finding C-1 of the
        ``audits/audit-2026-05-05.md`` audit).
        """
        config = ScarletDeblendTask.ConfigClass()
        config.maxNumberOfPeaks = 2
        config.catchFailures = False

        bundle = self._deblend(SCENES["three_source_blend"], config=config)

        catalog = bundle.result.objectParents
        parents = catalog[catalog["parent"] == 0]
        self.assertEqual(len(parents), 1)
        parent = parents[0]
        self.assertTrue(parent.get("deblend_skipped"))
        self.assertTrue(parent.get("deblend_skipped_tooManyPeaks"))
        self.assertFalse(parent.get("deblend_skipped_parentTooBig"))
        self.assertFalse(parent.get("deblend_blendConvergenceFailedFlag"))

    def test_all_subblends_failed_updates_parent_record(self):
        """When every sub-blend of a multi-peak parent fails, the
        aggregate ``deblend_*`` columns must land on the *parent*
        record, not on the last sub-blend.

        Triggers the all-sub-blends-failed branch by capping
        ``maxNumberOfPeaks=2`` on the three-peak ``three_source_blend``,
        so every sub-blend is skipped as ``deblend_skipped_tooManyPeaks``
        and the aggregate-update branch runs. Asserts the parent record
        carries the summary (``deblend_nPeaks`` matches the parent
        footprint's peak count). Under the bug the aggregate landed on
        the trailing sub-blend's record and the parent stayed at default.
        Regression test for finding C-5 of the
        ``audits/audit-2026-05-05.md`` audit.
        """
        config = ScarletDeblendTask.ConfigClass()
        config.maxNumberOfPeaks = 2
        config.catchFailures = False
        bundle = self._deblend(SCENES["three_source_blend"], config=config)
        catalog = bundle.result.objectParents
        parents = catalog[catalog["parent"] == 0]
        self.assertEqual(len(parents), 1)
        parent = parents[0]
        parentNPeaks = len(parent.getFootprint().peaks)
        self.assertEqual(parent.get("deblend_nPeaks"), parentNPeaks)

    def test_skip_doesnt_affect_other_parents(self):
        """One parent being skipped does not affect deblending of
        other parents.

        The ``multi-blend`` scene detects four parents; with
        ``maxNumberOfPeaks=2`` exactly one (the three-peak blend) is
        skipped. The remaining three parents are not skipped, and
        each non-isolated one produces ``nChild == nPeaks`` children
        — the deblend invariant.
        """
        config = ScarletDeblendTask.ConfigClass()
        config.maxNumberOfPeaks = 2
        config.catchFailures = False

        bundle = self._deblend(SCENES["multi-blend"], config=config)

        catalog = bundle.result.objectParents
        parents = catalog[catalog["parent"] == 0]
        self.assertEqual(np.sum(parents["deblend_skipped"]), 1)

        skipped = parents[parents["deblend_skipped"]]
        self.assertTrue(skipped[0].get("deblend_skipped_tooManyPeaks"))

        non_skipped = parents[~parents["deblend_skipped"]]
        self.assertEqual(len(non_skipped), 3)
        blend_parents = non_skipped[~non_skipped["deblend_skipped_isolatedParent"]]
        self.assertEqual(len(blend_parents), 2)
        for p in blend_parents:
            self.assertEqual(p.get("deblend_nChild"), p.get("deblend_nPeaks"))

    def test_convergence_flag_false_when_converged(self):
        """``deblend_blendConvergenceFailedFlag`` is unset for a blend
        that reaches convergence.

        The ``three_source_blend`` parent converges in well under
        ``maxIter`` iterations under the default config, so the flag —
        documented as "at least one source in the blend failed to
        converge" — must be `False`. Regression test for finding C-1
        of the ``audits/audit-2026-05-05.md`` audit (the flag was
        previously stored with inverted semantics, reporting a
        converged blend as failed).
        """
        defaultMaxIter = ScarletDeblendTask.ConfigClass().maxIter
        bundle = self._deblend(SCENES["three_source_blend"])
        parents = bundle.result.objectParents
        parents = parents[parents["parent"] == 0]
        self.assertEqual(len(parents), 1)
        parent = parents[0]
        # The blend stopped because it converged, not because it hit
        # the iteration cap.
        self.assertLess(parent.get("deblend_iterations"), defaultMaxIter)
        self.assertFalse(parent.get("deblend_blendConvergenceFailedFlag"))

    def test_convergence_flag_true_when_not_converged(self):
        """``deblend_blendConvergenceFailedFlag`` is set for a blend
        that exhausts ``maxIter`` without converging.

        Capping ``maxIter`` at 2 forces the ``three_source_blend``
        parent to stop at the iteration limit before the relative-error
        criterion is met, so the flag must be `True`. Regression test
        for finding C-1 of the ``audits/audit-2026-05-05.md`` audit.
        """
        config = ScarletDeblendTask.ConfigClass()
        config.maxIter = 2
        config.catchFailures = False
        bundle = self._deblend(SCENES["three_source_blend"], config=config)
        parents = bundle.result.objectParents
        parents = parents[parents["parent"] == 0]
        self.assertEqual(len(parents), 1)
        parent = parents[0]
        # The blend stopped at the iteration cap, i.e. it did not converge.
        self.assertEqual(parent.get("deblend_iterations"), config.maxIter)
        self.assertTrue(parent.get("deblend_blendConvergenceFailedFlag"))

    def test_sub_blend_progress_log_distinguishes_inner_loop(self):
        """The periodic logger inside the sub-blend loop names the
        sub-blend, not the outer parent count.

        Under the bug the inner-loop ``periodicLog.log`` reused the
        outer loop's format string, so long sub-blend runs of a
        single top-level parent kept emitting "Deblended N out of M
        parents" with stale ``N`` — making the task look stalled.
        Patches ``PeriodicLogger.LOGGING_INTERVAL`` so every ``log()``
        call fires regardless of wall time, then asserts at least one
        captured INFO message identifies a sub-blend. Regression test
        for finding DB-2 of the ``audits/audit-2026-05-05.md`` audit.
        """
        # Perturb a config field that no other test touches so the
        # ``pipeline.deblend`` memoization cache misses and the task
        # actually runs inside the ``assertLogs`` block.
        config = ScarletDeblendTask.ConfigClass()
        config.minIter = 1
        config.catchFailures = False

        # Negative interval guarantees ``time.time() > next_log_time``
        # on every call.
        with patch(
            "lsst.utils.logging.PeriodicLogger.LOGGING_INTERVAL", -1.0
        ):
            with self.assertLogs(level="INFO") as logs:
                self._deblend(SCENES["multi-blend"], config=config)

        self.assertTrue(
            any("sub-blend" in record for record in logs.output),
            f"No sub-blend progress message in: {logs.output}",
        )

    def test_is_masked_combines_bands_with_and(self):
        """``_isMasked`` counts a pixel toward the ``maskLimits``
        fraction only when *every* band has that mask bit set.

        This mirrors ``buildObservation``'s per-band weight zeroing
        — a pixel only becomes truly unconstrained when masked in
        all bands; if even one band leaves it unmasked, that band's
        data still constrains it. Sets ``INTRP`` in band 0 only
        across a multi-peak parent's footprint and asserts the
        parent is not flagged masked. Under the bug (OR across
        bands) every footprint pixel had ``INTRP`` set in *some*
        band and the parent was skipped. A second pass sets
        ``INTRP`` in every band and asserts the parent is then
        flagged, confirming AND still triggers when the bit is
        universal. Regression test for finding DB-3 of the
        ``audits/audit-2026-05-05.md`` audit.
        """
        bundle = self._deblend(SCENES["multi-blend"])
        parents = bundle.result.objectParents
        parents = parents[parents["parent"] == 0]
        parent = next(p for p in parents if p["deblend_nPeaks"] > 1)
        footprint = parent.getFootprint()

        # Deep-copy the per-band exposures so the cached bundle's
        # mask plane isn't corrupted for downstream tests.
        bands = bundle.image.mCoadd.bands
        mExposure = afwImage.MultibandExposure.fromExposures(
            bands, [exp.clone() for exp in bundle.image.mCoadd]
        )

        config = ScarletDeblendTask.ConfigClass()
        config.maskLimits = {"INTRP": 0.05}
        task = ScarletDeblendTask(
            schema=Schema(bundle.deconvolved.detection.schema),
            config=config,
        )

        intrp = mExposure.mask.getPlaneBitMask("INTRP")

        # Sanity: a clean mask plane is not flagged.
        self.assertFalse(task._isMasked(footprint, mExposure))

        # INTRP in band 0 only — not flagged under AND, was under OR.
        mExposure.mask.array[0] |= intrp
        self.assertFalse(task._isMasked(footprint, mExposure))

        # INTRP in every band — flagged under AND as well.
        for b in range(len(bands)):
            mExposure.mask.array[b] |= intrp
        self.assertTrue(task._isMasked(footprint, mExposure))

    def test_max_iter_zero_skips_fit_cleanly(self):
        """With ``maxIter=0`` the optimizer never runs, so the parent
        record's ``deblend_iterations`` must report 0 (not the synthetic
        2 the old code produced by stuffing two equal entries into
        ``blend.loss`` to keep downstream consumers indexable).

        Also pins ``deblend_blendConvergenceFailedFlag`` False: a blend
        that was never fit hasn't failed convergence, and
        ``_checkBlendConvergence`` must handle the empty-loss case
        without crashing. Regression test for finding DB-1 of the
        ``audits/audit-2026-05-05.md`` audit.
        """
        config = ScarletDeblendTask.ConfigClass()
        config.maxIter = 0
        config.catchFailures = False

        bundle = self._deblend(SCENES["three_source_blend"], config=config)
        parents = bundle.result.objectParents
        parents = parents[parents["parent"] == 0]
        self.assertEqual(len(parents), 1)
        parent = parents[0]

        self.assertEqual(parent.get("deblend_iterations"), 0)
        self.assertFalse(parent.get("deblend_blendConvergenceFailedFlag"))

    def test_detected_peak_skips_pseudo_peaks(self):
        """Each deblended source's ``detectedPeak`` is the real peak at
        its own center, even when a pseudo peak precedes it in the
        footprint's peak list.

        ``deblend`` filters pseudo peaks (e.g. sky objects) out of the
        list it initializes sources from, so the back-pointer to the
        ``PeakRecord`` must be indexed in that *filtered* list. Indexing
        the unfiltered ``footprint.peaks`` shifts every source's
        ``detectedPeak`` by the number of preceding pseudo peaks.
        Regression test for finding C-4 of the
        ``audits/audit-2026-05-05.md`` audit.
        """
        deconv = pipeline.deconvolve(
            pipeline.detect(pipeline.build_image(SCENES["three_source_blend"]))
        )
        config = ScarletDeblendTask.ConfigClass()
        context = ScarletDeblendContext.build(
            deconv.image.mCoadd, deconv.mDeconvolved, deconv.detection.catalog, config
        )

        # Rebuild the (single, three-peak) parent footprint with a sky
        # pseudo peak prepended ahead of the three real peaks.
        origFootprint = deconv.detection.catalog[0].getFootprint()
        peakSchema = PeakTable.makeMinimalSchema()
        skyKey = peakSchema.addField(
            "merge_peak_sky", type="Flag", doc="sky pseudo peak"
        )
        footprint = afwDet.Footprint(origFootprint.spans, peakSchema)
        pseudoPeak = footprint.addPeak(12, 14, 1.0)
        pseudoPeak.set(skyKey, True)
        pseudoId = pseudoPeak.getId()
        for peak in origFootprint.peaks:
            footprint.addPeak(peak.getIx(), peak.getIy(), 10.0)

        blend = deblend(context, footprint, config, spectrumInit=False)

        self.assertEqual(len(blend.sources), len(origFootprint.peaks))
        for source in blend.sources:
            detectedPeak = source.detectedPeak
            # The real peaks sit exactly on the source centers, so a
            # correctly-mapped detectedPeak lands on its source's center
            # (center is ordered (y, x)).
            self.assertEqual(detectedPeak.getIy(), source.center[0])
            self.assertEqual(detectedPeak.getIx(), source.center[1])
            self.assertNotEqual(detectedPeak.getId(), pseudoId)

    def test_parent_peak_mapper_propagates_extra_peak_fields(self):
        """``ScarletDeblendTask.parentPeakSchemaMapper`` carries extra
        ``merge_peak_*`` peak fields onto a parent record.

        The task copies a parent footprint's first peak onto the
        parent source record via this mapper. Without the extra-field
        mappings any ``merge_peak_*`` flags (for example
        ``merge_peak_sky``) on that peak are silently dropped,
        breaking propagation of pseudo-source flags to deconvolved
        sub-blend parents. Regression test for finding C-8 of the
        ``audits/audit-2026-05-05.md`` audit.
        """
        schema = afwTable.SourceTable.makeMinimalSchema()
        peakSchema = PeakTable.makeMinimalSchema()
        skyKey = peakSchema.addField(
            "merge_peak_sky", type="Flag", doc="sky pseudo peak"
        )
        task = ScarletDeblendTask(schema=schema, peakSchema=peakSchema)

        peakCat = afwDet.PeakCatalog(afwDet.PeakTable.make(peakSchema))
        peak = peakCat.addNew()
        peak.set(skyKey, True)

        parentCatalog = afwTable.SourceCatalog(
            afwTable.SourceTable.make(task.parentSchema)
        )
        parent = parentCatalog.addNew()
        parent.assign(peak, task.parentPeakSchemaMapper)

        self.assertTrue(parent.get("merge_peak_sky"))

    def test_build_intersecting_footprints_rejects_out_of_bounds_peak(self):
        """A peak whose pixel coordinates fall outside the bbox of
        ``footprintImage`` raises ``RuntimeError``, in every direction.

        ``_buildIntersectingFootprints`` indexed ``footprintImage.data``
        with the peak's offset from the image bbox origin inside a
        ``try/except IndexError``. NumPy only raises ``IndexError`` for
        out-of-range *positive* indices; negative indices silently wrap
        from the opposite edge of the array, so a peak west or south
        of the bbox origin used to be silently mapped to an unrelated
        pixel on the opposite edge and processed as if it lay there.
        The bounds check must reject all four directions. Regression
        test for finding C-9 of the ``audits/audit-2026-05-05.md``
        audit.
        """
        schema = afwTable.SourceTable.makeMinimalSchema()
        task = ScarletDeblendTask(schema=schema)
        # 5x5 footprintImage filled with zeros at origin (10, 10), so
        # NumPy wraparound from a negative index lands on a 0 cell and
        # the inner loop would silently skip the peak under the bug.
        footprintImage = scl.Image(
            np.zeros((5, 5), dtype=np.int32), yx0=(10, 10)
        )
        parentCatalog = afwTable.SourceCatalog(
            afwTable.SourceTable.make(task.parentSchema)
        )

        # One peak per out-of-bounds direction relative to the
        # ``[10..14, 10..14]`` bbox. ``south`` and ``west`` exercise
        # the negative-wraparound branches the bug missed; ``north``
        # and ``east`` exercise the positive-overflow branches that
        # NumPy raised on.
        directions = {
            "west": (5, 12),
            "south": (12, 5),
            "east": (20, 12),
            "north": (12, 20),
        }
        for name, (x, y) in directions.items():
            with self.subTest(direction=name):
                footprint = afwDet.Footprint(
                    SpanSet(), PeakTable.makeMinimalSchema()
                )
                footprint.addPeak(x, y, 1.0)
                with self.assertRaises(RuntimeError):
                    task._buildIntersectingFootprints(
                        parentId=0,
                        afwFootprint=footprint,
                        parentCatalog=parentCatalog,
                        sclFootprints=[],
                        footprintImage=footprintImage,
                    )

    def test_catalog_total_count(self):
        """The deblended catalog has one row per input model."""
        bundle = self._deblend(SCENES["multi-blend"])
        catalog = bundle.result.deblendedCatalog
        self.assertEqual(len(catalog), len(SCENES["multi-blend"].models))

    def test_catalog_sorted_by_parent_id(self):
        """Catalog rows are sorted by parent id."""
        bundle = self._deblend(SCENES["multi-blend"])
        catalog = bundle.result.deblendedCatalog
        np.testing.assert_array_equal(sorted(catalog["parent"]), catalog["parent"])

    def test_child_ids_above_parent_ids(self):
        """Every child source's id exceeds the maximum parent id."""
        bundle = self._deblend(SCENES["multi-blend"])
        catalog = bundle.result.deblendedCatalog
        objectParents = bundle.result.objectParents
        maxId = np.max(objectParents["id"])
        for src in catalog:
            if src["parent"] > 0:
                self.assertGreater(src["id"], maxId)

    def test_every_source_has_one_peak(self):
        """After ``updateCatalogFootprints``, every catalog source's
        footprint has exactly one peak.

        Isolated parents keep their (single-peak) detection footprint;
        children get HeavyFootprints attached by
        ``updateCatalogFootprints`` whose peaks come from the
        scarlet model. The deblend invariant is "one peak per
        source row" after the catalog has been hydrated.
        """
        bundle = self._deblend(SCENES["multi-blend"])
        catalog = bundle.result.deblendedCatalog
        modelData = bundle.result.scarletModelData
        mes.io.updateCatalogFootprints(
            modelData,
            catalog,
            band=bundle.image.bands[0],
            imageForRedistribution=None,
            removeScarletData=False,
            updateFluxColumns=True,
        )

        for src in catalog:
            self.assertEqual(len(src.getFootprint().peaks), 1)

    def test_nChild_consistency(self):
        """``deblend_nChild`` over multi-peak parents accounts for every child.

        Sum of ``deblend_nChild`` over the top-level multi-peak parents
        equals (total catalog size) − (number of top-level isolated
        parents) — i.e. every non-parent row in the catalog is a child
        of exactly one multi-peak parent.
        """
        bundle = self._deblend(SCENES["multi-blend"])
        catalog = bundle.result.deblendedCatalog
        objectParents = bundle.result.objectParents
        isolated = catalog[catalog["parent"] == 0]
        parents = objectParents[
            (objectParents["parent"] == 0) & (objectParents["deblend_nPeaks"] > 1)
        ]
        self.assertEqual(
            np.sum(parents["deblend_nChild"]), len(catalog) - len(isolated)
        )

    def test_isolated_parents_marked(self):
        """``deblend_skipped_isolatedParent`` is set on every single-peak
        parent.
        """
        bundle = self._deblend(SCENES["multi-blend"])
        objectParents = bundle.result.objectParents
        isolatedParents = objectParents[
            (objectParents["parent"] == 0)
            & (objectParents["deblend_nPeaks"] == 1)
        ]
        self.assertEqual(
            np.sum(objectParents["deblend_skipped_isolatedParent"]),
            len(isolatedParents),
        )

    def test_isolated_source_persisted(self):
        """Each isolated parent has a matching ``SourceData`` in
        ``modelData.isolated``: same span array, same origin.
        """
        bundle = self._deblend(SCENES["multi-blend"])
        catalog = bundle.result.deblendedCatalog
        modelData = bundle.result.scarletModelData
        isolated = catalog[catalog["parent"] == 0]
        self.assertEqual(len(isolated), len(modelData.isolated))
        for sid, source in modelData.isolated.items():
            catalog_footprint = catalog.find(sid).getFootprint()
            np.testing.assert_array_equal(
                source.span_array, catalog_footprint.spans.asArray()
            )
            self.assertTupleEqual(
                source.origin[::-1], tuple(catalog_footprint.getBBox().getMin())
            )

    def test_heavy_footprint_flux_at_peak(self):
        """Heavy-footprint flux at the source's peak equals
        ``deblend_peak_instFlux`` in every band, with and without
        flux redistribution.

        This works in the test image because the detected peak is in
        the same location as the scarlet peak; if the peak were
        shifted the flux value would still be correct but
        ``deblend_peak_center`` would not be the right pixel.
        """
        bundle = self._deblend(SCENES["multi-blend"])
        for useFlux in [False, True]:
            for band in bundle.image.bands:
                with self.subTest(band=band, useFlux=useFlux):
                    self._attach_band_footprints(bundle, band, useFlux)
                    for _, child in self._iter_multipeak_children(bundle):
                        fp = child.getFootprint()
                        img = fp.extractImage(fill=0.0)
                        px = child.get("deblend_peak_center_x")
                        py = child.get("deblend_peak_center_y")
                        flux = img[Point2I(px, py)]
                        self.assertEqual(
                            flux, child.get("deblend_peak_instFlux")
                        )

    def test_heavy_footprint_band_columns_populated(self):
        """``updateCatalogFootprints`` populates every band-dependent
        ``deblend_*`` column it owns on each deblended child.

        Pins both the four pre-existing writes
        (``deblend_zeroFlux``, ``deblend_dataCoverage``,
        ``deblend_scarletFlux``, ``deblend_peak_instFlux``) and the
        four blendedness/overlap metric writes
        (``deblend_maxOverlap``, ``deblend_fluxOverlap``,
        ``deblend_fluxOverlapFraction``, ``deblend_blendedness``).
        The metric writes were previously missing: their values were
        computed onto ``source.metrics`` by ``setDeblenderMetrics``
        but never pushed onto the catalog record, so every child
        carried the schema's default ``NaN`` for the four
        ``np.float32`` metric fields. Runs with ``useFlux=True`` so
        the ``deblend_dataCoverage`` branch is also covered.
        Regression test for finding DB-7 of the
        ``audits/audit-2026-05-05.md`` audit.
        """
        bundle = self._deblend(SCENES["multi-blend"])
        band = bundle.image.bands[0]
        self._attach_band_footprints(bundle, band, useFlux=True)

        for _, child in self._iter_multipeak_children(bundle):
            with self.subTest(childId=child.getId()):
                self.assertFalse(child.get("deblend_zeroFlux"))
                self.assertGreater(child.get("deblend_dataCoverage"), 0)
                self.assertGreater(child.get("deblend_scarletFlux"), 0)
                self.assertFalse(
                    np.isnan(child.get("deblend_peak_instFlux"))
                )
                self.assertGreater(child.get("deblend_maxOverlap"), 0)
                self.assertGreater(child.get("deblend_fluxOverlap"), 0)
                self.assertGreater(
                    child.get("deblend_fluxOverlapFraction"), 0
                )
                self.assertGreater(child.get("deblend_blendedness"), 0)

    def test_heavy_footprint_peak_position(self):
        """The HeavyFootprint's peak position and the scarlet model's
        source center both match ``deblend_peak_center_{x,y}``.

        The assertion is geometric (independent of band / useFlux),
        so this runs once against the first band's HeavyFootprints.
        """
        bundle = self._deblend(SCENES["multi-blend"])
        band = bundle.image.bands[0]
        self._attach_band_footprints(bundle, band, useFlux=False)

        for parent, child in self._iter_multipeak_children(bundle):
            fp = child.getFootprint()
            px = child.get("deblend_peak_center_x")
            py = child.get("deblend_peak_center_y")
            self.assertEqual(px, fp.getPeaks()[0].getIx())
            self.assertEqual(py, fp.getPeaks()[0].getIy())

            _, source, _ = self._scarlet_blend_for_child(
                bundle, parent, child, band
            )
            self.assertEqual(source.center[1], px)
            self.assertEqual(source.center[0], py)

    def test_heavy_footprint_matches_model(self):
        """The HeavyFootprint pixel data matches the scarlet model.

        Two variants per band:

        - ``useFlux=False``: convolve the source's bare scarlet model
          with the band PSF and compare to the HeavyFootprint image.
        - ``useFlux=True``: rebind the blend's observation to the
          observed coadd, call ``conserve_flux`` to redistribute, and
          compare the flux-weighted model to the HeavyFootprint
          (projected onto the redistributed image's frame).
        """
        bundle = self._deblend(SCENES["multi-blend"])
        image = bundle.image
        for useFlux in [False, True]:
            for band in image.bands:
                with self.subTest(band=band, useFlux=useFlux):
                    self._attach_band_footprints(bundle, band, useFlux)
                    imageForRedistribution = (
                        image.mCoadd[band] if useFlux else None
                    )

                    for parent, child in self._iter_multipeak_children(bundle):
                        fp = child.getFootprint()
                        img = fp.extractImage(fill=0.0)
                        blend, source, parentFootprint = (
                            self._scarlet_blend_for_child(
                                bundle, parent, child, band
                            )
                        )

                        if useFlux:
                            assert imageForRedistribution is not None
                            x0, y0 = parentFootprint.getBBox().getMin()
                            yx0 = (y0, x0)
                            _images = imageForRedistribution[
                                parentFootprint.getBBox()
                            ].image.array
                            blend.observation.images = scl.Image(
                                _images[None, :, :],
                                yx0=yx0,
                                bands=(band,),
                            )
                            blend.observation.weights = scl.Image(
                                parentFootprint.spans.asArray()[None, :, :],
                                yx0=yx0,
                                bands=(band,),
                            )
                            blend.conserve_flux()
                            model = source.flux_weighted_image.data[0]
                            my0, mx0 = source.flux_weighted_image.yx0
                            image_f = afwImage.ImageF(
                                model, xy0=Point2I(mx0, my0)
                            )
                            fp.insert(image_f)
                            np.testing.assert_almost_equal(
                                image_f.array, model
                            )
                        else:
                            bbox = mes.utils.bboxToScarletBox(fp.getBBox())
                            model = blend.observation.convolve(
                                source.get_model().project(bbox=bbox),
                                mode="real",
                            ).data[0]
                            np.testing.assert_almost_equal(img.array, model)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
