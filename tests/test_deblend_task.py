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

Exercises the deblend task's failure / skip semantics. The main
``test_footprints`` end-to-end test will move here in a later step;
for now this file holds only the skip-condition tests.
"""

import unittest

import lsst.utils.tests
import numpy as np
from lsst.meas.extensions.scarlet.scarletDeblendTask import ScarletDeblendTask

import pipeline
from scenes import SCENES


class TestDeblendTask(lsst.utils.tests.TestCase):
    """Tests for ``ScarletDeblendTask`` skip semantics in
    ``lsst.meas.extensions.scarlet.scarletDeblendTask``.
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


if __name__ == "__main__":
    unittest.main()
