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

import unittest
import numpy as np
import scarlet
import lsst.meas.extensions.scarlet as lmeScarlet
import lsst.utils.tests


class TestLsstSource(lsst.utils.tests.TestCase):
    def setUp(self):
        self.shape = (5, 11, 15)
        B, Ny, Nx = self.shape

        x = np.linspace(-2, 2, 5)
        y = np.linspace(-2, 2, 5)
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)

        self.true_sed = np.arange(B)
        self.true_morph = np.zeros(self.shape[1:])

        self.skycoord = (np.array(self.true_morph.shape) - 1) // 2
        cy, cx = self.skycoord
        self.true_morph[cy-2:cy+3, cx-2:cx+3] = 3-r

    def test_init(self):
        morph = self.true_morph.copy()
        morph[5, 3] = 10

        # Test init
        images = self.true_sed[:, None, None] * morph[None, :, :]
        scene = lmeScarlet.LsstScene(self.shape)
        observation = lmeScarlet.LsstObservation(images)
        bg_rms = np.ones_like(self.true_sed) * 1e-3
        src = lmeScarlet.ExtendedSource(self.skycoord, scene, observation, bg_rms)
        self.assertFloatsAlmostEqual(src.pixel_center, self.skycoord)
        self.assertEqual(src.symmetric, False)
        self.assertEqual(src.monotonic, True)
        self.assertEqual(src.center_step, 5)
        self.assertEqual(src.delay_thresh, 10)

        self.assertFloatsAlmostEqual(src.sed/3, self.true_sed)
        self.assertFloatsAlmostEqual(src.morph*3, self.true_morph)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
