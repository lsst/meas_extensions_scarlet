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

import lsst.afw.image as afwImage
import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np
from lsst.geom import Point2I
from lsst.meas.extensions.scarlet.scarletDeblendTask import ScarletDeblendTask

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

        image = pipeline.build_image(SCENES["large_two_sersic"])
        detection = pipeline.detect(image)
        deconv = pipeline.deconvolve(detection)
        bundle = pipeline.deblend(deconv, config=config)

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
        """
        config = ScarletDeblendTask.ConfigClass()
        config.maxNumberOfPeaks = 2
        config.catchFailures = False

        image = pipeline.build_image(SCENES["three_source_blend"])
        detection = pipeline.detect(image)
        deconv = pipeline.deconvolve(detection)
        bundle = pipeline.deblend(deconv, config=config)

        catalog = bundle.result.objectParents
        parents = catalog[catalog["parent"] == 0]
        self.assertEqual(len(parents), 1)
        parent = parents[0]
        self.assertTrue(parent.get("deblend_skipped"))
        self.assertTrue(parent.get("deblend_skipped_tooManyPeaks"))
        self.assertFalse(parent.get("deblend_skipped_parentTooBig"))

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

        image = pipeline.build_image(SCENES["multi-blend"])
        detection = pipeline.detect(image)
        deconv = pipeline.deconvolve(detection)
        bundle = pipeline.deblend(deconv, config=config)

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

    def test_footprints(self):
        image = pipeline.build_image(SCENES["multi-blend"])
        detection = pipeline.detect(image)
        deconv = pipeline.deconvolve(detection)
        bundle = pipeline.deblend(deconv)

        catalog = bundle.result.deblendedCatalog
        objectParents = bundle.result.objectParents
        modelData = bundle.result.scarletModelData
        observedPsf = modelData.metadata["psf"]
        modelPsf = modelData.metadata["model_psf"]

        # Check that isolated sources are handled correctly
        isolated = catalog[(catalog["parent"] == 0)]
        self.assertEqual(len(isolated), len(modelData.isolated))
        for sid, source in modelData.isolated.items():
            catalog_footprint = catalog.find(sid).getFootprint()
            isolated_array = catalog_footprint.spans.asArray()
            np.testing.assert_array_equal(source.span_array, isolated_array)

            # Check that the origin is correct
            self.assertTupleEqual(source.origin[::-1], tuple(catalog_footprint.getBBox().getMin()))

        # Verify that the isolated parent flag is being set
        isolatedParents = objectParents[
            (objectParents["parent"] == 0)
            & (objectParents["deblend_nPeaks"] == 1)
        ]
        self.assertEqual(np.sum(objectParents["deblend_skipped_isolatedParent"]), len(isolatedParents))

        # Attach the footprints in each band and compare to the full
        # data model. This is done in each band, both with and without
        # flux re-distribution to test all of the different possible
        # options of loading catalog footprints.
        for useFlux in [False, True]:
            for band in image.bands:
                bandIndex = image.bands.index(band)
                coadd = image.mCoadd[band]

                if useFlux:
                    imageForRedistribution = coadd
                else:
                    imageForRedistribution = None

                mes.io.updateCatalogFootprints(
                    modelData,
                    catalog,
                    band=band,
                    imageForRedistribution=imageForRedistribution,
                    removeScarletData=False,
                    updateFluxColumns=True,
                )

                # Check that the number of deblended children is consistent
                parents = objectParents[
                    (objectParents["parent"] == 0) & (objectParents["deblend_nPeaks"] > 1)]
                self.assertEqual(
                    np.sum(parents["deblend_nChild"]), len(catalog) - len(isolated)
                )

                for parent in parents:
                    children = catalog[catalog["parent"] == parent.get("id")]

                    # Extract the parent blend data
                    parentBlendData = modelData.blends[parent.getId()]
                    parentFootprint = parent.getFootprint()
                    x0, y0 = parentFootprint.getBBox().getMin()
                    width, height = parentFootprint.getBBox().getDimensions()
                    yx0 = (y0, x0)

                    for child in children:
                        fp = child.getFootprint()
                        img = fp.extractImage(fill=0.0)
                        # Check that the flux at the center is correct.
                        # Note: this only works in this test image because the
                        # detected peak is in the same location as the
                        # scarlet peak.
                        # If the peak is shifted,
                        # the flux value will be correct but
                        # deblend_peak_center is not the correct location.
                        px = child.get("deblend_peak_center_x")
                        py = child.get("deblend_peak_center_y")
                        flux = img[Point2I(px, py)]
                        self.assertEqual(flux, child.get("deblend_peak_instFlux"))

                        # Check that the peak positions match the catalog entry
                        peaks = fp.getPeaks()
                        self.assertEqual(px, peaks[0].getIx())
                        self.assertEqual(py, peaks[0].getIy())

                        # Load the data to check against the HeavyFootprint
                        blendData = parentBlendData.children[child["deblend_blendId"]]
                        # We need to set an observation in order to convolve
                        # the model.
                        modelBox = scl.Box((height, width), origin=(y0, x0))
                        observation = scl.Observation.empty(
                            bands=("dummy",),
                            psfs=observedPsf[bandIndex][None, :, :],
                            model_psf=modelPsf[None, :, :],
                            bbox=modelBox,
                            dtype=np.float32,
                        )
                        blend = mes.io.monochromaticDataToScarlet(
                            blendData=blendData,
                            bandIndex=bandIndex,
                            observation=observation,
                        )

                        # Get the scarlet model for the source
                        source = next(
                            src for src in blend.sources if src.metadata["id"] == child.getId()
                        )
                        self.assertEqual(source.center[1], px)
                        self.assertEqual(source.center[0], py)

                        if useFlux:
                            assert imageForRedistribution is not None
                            # Get the flux re-weighted model and test against
                            # the HeavyFootprint.
                            # The HeavyFootprint needs to be projected onto
                            # the image of the flux-redistributed model,
                            # since the HeavyFootprint
                            # may trim rows or columns.
                            _images = imageForRedistribution[
                                parentFootprint.getBBox()
                            ].image.array
                            blend.observation.images = scl.Image(
                                _images[None, :, :],
                                yx0=yx0,
                                bands=("dummy",),
                            )
                            blend.observation.weights = scl.Image(
                                parentFootprint.spans.asArray()[None, :, :],
                                yx0=yx0,
                                bands=("dummy",),
                            )
                            blend.conserve_flux()
                            model = source.flux_weighted_image.data[0]
                            my0, mx0 = source.flux_weighted_image.yx0
                            image_f = afwImage.ImageF(model, xy0=Point2I(mx0, my0))
                            fp.insert(image_f)
                            np.testing.assert_almost_equal(image_f.array, model)
                        else:
                            # Get the model for the source and test
                            # against the HeavyFootprint
                            bbox = fp.getBBox()
                            bbox = mes.utils.bboxToScarletBox(bbox)
                            model = blend.observation.convolve(
                                source.get_model().project(bbox=bbox), mode="real"
                            ).data[0]
                            np.testing.assert_almost_equal(img.array, model)

        # Check that all sources have the correct number of peaks
        maxId = np.max(objectParents["id"])
        for src in catalog:
            fp = src.getFootprint()
            self.assertEqual(len(fp.peaks), 1)
            if src["parent"] > 0:
                # Check that source IDs are greater than the max parent ID
                self.assertGreater(src["id"], maxId)

        # Ensure that sources are sorted by parent ID
        np.testing.assert_array_equal(sorted(catalog["parent"]), catalog["parent"])

        # Check that the catalog matches the expected results
        self.assertEqual(len(catalog), len(SCENES["multi-blend"].models))


if __name__ == "__main__":
    unittest.main()
