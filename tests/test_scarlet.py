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
from scarlet.initialization import init_source
import lsst.utils.tests

from utils import numpyToStack, initData


class TestLsstSource(lsst.utils.tests.TestCase):
    def test_to_heavy(self):
        shape = (5, 31, 55)
        B, Ny, Nx = shape
        coords = [(20, 10), (10, 30), (17, 42)]
        result = initData(shape, coords, [3, 2, 1])
        targetPsfImage, psfImages, images, channels, seds, morphs, targetPsf, psfs = result
        images = images.astype(np.float32)
        seds = seds.astype(np.float32)

        frame = scarlet.Frame(shape, psf=targetPsf, channels=np.arange(B))
        observation = scarlet.Observation(images, psf=psfImages, channels=np.arange(B)).match(frame)
        foot, peak, bbox = numpyToStack(images, coords[0], (15, 3))
        xmin = bbox.getMinX()
        ymin = bbox.getMinY()
        center = np.array([peak.getIy()-ymin, peak.getIx()-xmin], dtype=int)
        src = init_source(frame=frame, center=center, observations=[observation], thresh=0)

        # Convolve the model with the observed PSF
        model = src.get_model(frame=src.frame)
        model = observation.render(model)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
