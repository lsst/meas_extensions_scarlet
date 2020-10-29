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
from scarlet.initialization import initSource

import lsst.meas.extensions.scarlet as mes
import lsst.utils.tests

from utils import numpyToStack, initData


class TestLsstSource(lsst.utils.tests.TestCase):
    def test_init(self):
        # Initialize the model
        shape = (5, 31, 55)
        B, Ny, Nx = shape

        x = np.linspace(-2, 2, 5)
        y = np.linspace(-2, 2, 5)
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)

        trueSed = np.arange(B)
        trueMorph = np.zeros(shape[1:])

        center = (np.array(trueMorph.shape) - 1) // 2
        cy, cx = center
        trueMorph[cy-2:cy+3, cx-2:cx+3] = 3-r

        morph = trueMorph.copy()
        # Make a point that is not monotonic or symmetric to ensure
        # that it is supressed.
        morph[5, 3] = 10

        # Create the scarlet objects
        images = trueSed[:, None, None] * morph[None, :, :]
        frame = scarlet.Frame(shape, channels=np.arange(B))
        observation = scarlet.Observation(images, channels=np.arange(B))

        # init stack objects
        foot, peak, bbox = numpyToStack(images, center, (15, 3))
        # init source
        xmin = bbox.getMinX()
        ymin = bbox.getMinY()
        center = np.array([peak.getIy()-ymin, peak.getIx()-xmin], dtype=int)
        src = initSource(frame=frame, center=center, observation=observation, thresh=0, downgrade=False)

        # scarlet has more flexible models now,
        # so `sed` and `morph` are no longer attributes,
        # meaning we have to extract them ourselves.
        sed = src.children[0].parameters[0]._data
        morph = src.children[1].parameters[0]._data

        self.assertFloatsAlmostEqual(sed/3, trueSed)
        src_morph = np.zeros(frame.shape[1:], dtype=morph.dtype)
        src_morph[src._model_frame_slices[1:]] = (morph*3)[src._model_slices[1:]]
        self.assertFloatsAlmostEqual(src_morph, trueMorph, rtol=1e-7)
        self.assertFloatsEqual(src.center, center)
        self.assertEqual(foot.getBBox(), bbox)

    def test_to_heavy(self):
        shape = (5, 31, 55)
        B, Ny, Nx = shape
        coords = [(20, 10), (10, 30), (17, 42)]
        result = initData(shape, coords, [3, 2, 1])
        targetPsfImage, psfImages, images, channels, seds, morphs, targetPsf, psfs = result
        images = images.astype(np.float32)
        seds = seds.astype(np.float32)

        frame = scarlet.Frame(shape, psfs=targetPsf, channels=np.arange(B))
        observation = scarlet.Observation(images, psfs=psfImages, channels=np.arange(B)).match(frame)
        foot, peak, bbox = numpyToStack(images, coords[0], (15, 3))
        xmin = bbox.getMinX()
        ymin = bbox.getMinY()
        center = np.array([peak.getIy()-ymin, peak.getIx()-xmin], dtype=int)
        src = initSource(frame=frame, center=center, observation=observation, thresh=0, downgrade=False)

        # Convolve the model with the observed PSF
        model = src.get_model(frame=src.frame)
        model = observation.render(model)

        # Test Model to Heavy
        filters = [f for f in "grizy"]
        src.detectedPeak = peak
        hFoot = mes.source.modelToHeavy(src, filters, bbox.getMin(), observation)
        hModel = hFoot.getImage(fill=0).image.array

        self.assertEqual(bbox, hFoot.getBBox())
        self.assertFloatsAlmostEqual(hModel, model, rtol=1e-4, atol=1e-4)

        # Test the peak in each band
        for single in hFoot:
            peaks = single.getPeaks()
            self.assertEqual(len(peaks), 1)
            hPeak = peaks[0]
            self.assertEqual(hPeak.getIx()-xmin, coords[0][1])
            self.assertEqual(hPeak.getIy()-ymin, coords[0][0])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
