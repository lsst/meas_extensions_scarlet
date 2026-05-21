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

    def test_skipped(self):
        # Use tight configs to force skipping a 3 source footprint
        # and "large" footprint
        config = ScarletDeblendTask.ConfigClass()
        config.maxFootprintArea = 2000
        config.maxNumberOfPeaks = 2
        config.catchFailures = False

        image = pipeline.build_image(SCENES["multi-blend"])
        detection = pipeline.detect(image)
        deconv = pipeline.deconvolve(detection)
        bundle = pipeline.deblend(deconv, config=config)

        catalog = bundle.result.objectParents
        parents = catalog[catalog["parent"] == 0]
        self.assertEqual(np.sum(parents["deblend_skipped"]), 2)
        self.assertEqual(np.sum(parents["deblend_skipped_parentTooBig"]), 1)
        self.assertEqual(np.sum(parents["deblend_skipped_tooManyPeaks"]), 1)


if __name__ == "__main__":
    unittest.main()
