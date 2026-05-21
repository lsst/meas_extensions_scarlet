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

"""Tests for ``DeconvolveExposureTask``.

Compares the per-band deconvolved exposures produced by the task
against the truth ``deconvolved`` image (the model rendered with the
narrow model PSF only). The two existing tests cover the default
``useFootprints=True`` path and the catalog-free
``useFootprints=False`` path; both consume the cached
``multi-blend`` scene from ``pipeline.py``.
"""

import unittest

import lsst.meas.extensions.scarlet as mes
import lsst.utils.tests
import numpy as np
from lsst.meas.extensions.scarlet.deconvolveExposureTask import DeconvolveExposureTask
from lsst.meas.extensions.scarlet.scarletDeblendTask import ScarletDeblendTask

import pipeline
from scenes import SCENES


class TestDeconvolveTask(lsst.utils.tests.TestCase):
    """Tests for ``DeconvolveExposureTask`` in
    ``lsst.meas.extensions.scarlet.deconvolveExposureTask``.

    Both tests run the deconvolve task on the cached ``multi-blend``
    scene and compare its output to the truth ``deconvolved`` image
    (the model convolved with the narrow model PSF only). The
    assertions ignore a 3×3 region around each source center because
    Sersic models have sharp peaks that the deconvolver does not
    recover bit-exactly.
    """

    def test_default_deconvolve(self):
        image = pipeline.build_image(SCENES["multi-blend"])
        detection = pipeline.detect(image)
        deconv = pipeline.deconvolve(detection)

        diff = image.deconvolved.data - deconv.mDeconvolved.image.array
        # Due to peakiness of Sersic models the center has a sharp peak,
        # so we ignore a 3x3 region around each source center
        for model in SCENES["multi-blend"].models:
            yc, xc = model.center
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    diff[:, yc+y, xc+x] = 0
        self.assertTrue(np.max(diff[:2]) < 10*np.std(image.noise))
        self.assertTrue(np.max(diff[2]) < 20*np.std(image.noise))

        context = mes.scarletDeblendTask.ScarletDeblendContext.build(
            image.mCoadd,
            deconv.mDeconvolved,
            detection.catalog,
            ScarletDeblendTask.ConfigClass()
        )

        self.assertEqual(len(context.footprints), 4)

    def test_catalog_free_deconvolve(self):
        config = DeconvolveExposureTask.ConfigClass()
        config.useFootprints = False
        image = pipeline.build_image(SCENES["multi-blend"])
        detection = pipeline.detect(image)
        deconv = pipeline.deconvolve(detection, config=config)

        diff = image.deconvolved.data - deconv.mDeconvolved.image.array
        # Due to peakiness of Sersic models the center has a sharp peak,
        # so we ignore a 3x3 region around each source center
        for model in SCENES["multi-blend"].models:
            yc, xc = model.center
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    diff[:, yc+y, xc+x] = 0
        self.assertTrue(np.max(diff[:2]) < 10*np.std(image.noise))
        self.assertTrue(np.max(diff[2]) < 20*np.std(image.noise))


if __name__ == "__main__":
    unittest.main()
