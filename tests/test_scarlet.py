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

import lsst.meas.extensions.scarlet as mes
import lsst.utils.tests
from lsst.afw.detection import PeakTable, multiband

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
        frame = mes.LsstFrame(shape)
        observation = mes.LsstObservation(images)

        # init stack objects
        foot, peak, bbox = numpyToStack(images, center, (15, 3))
        # init source
        src = mes.source.initSource(frame=frame, peak=peak, observation=observation, bbox=bbox, thresh=0)

        self.assertFloatsAlmostEqual(src.sed/3, trueSed)
        self.assertFloatsAlmostEqual(src.morph*3, trueMorph, rtol=1e-7)
        self.assertEqual(src.detectedPeak, peak)
        self.assertEqual(foot.getBBox(), bbox)

    def test_to_heavy(self):
        shape = (5, 31, 55)
        B, Ny, Nx = shape
        coords = [(20, 10), (10, 30), (17, 42)]
        result = initData(shape, coords, [3, 2, 1])
        targetPsfImage, psfImages, images, channels, seds, morphs, targetPsf, psfs = result
        images = images.astype(np.float32)
        seds = seds.astype(np.float32)

        frame = mes.LsstFrame(shape, psfs=targetPsfImage[None])
        observation = mes.LsstObservation(images, psfs=psfImages).match(frame)
        foot, peak, bbox = numpyToStack(images, coords[0], (15, 3))
        src = mes.source.initSource(frame=frame, peak=peak, observation=observation, bbox=bbox, thresh=0)
        # Get the HeavyFootprint
        peakSchema = PeakTable.makeMinimalSchema()
        hFoot = mes.morphToHeavy(src, peakSchema=peakSchema)
        hBBox = hFoot.getBBox()

        hMorph = multiband.heavyFootprintToImage(hFoot, fill=0).image.array
        sBbox = scarlet.bbox.Box.from_data(src.morph)
        self.assertFloatsAlmostEqual(hMorph, sBbox.extract_from(src.morph.astype(np.float32)))
        self.assertEqual(hBBox.getMinX(), sBbox.start[-1])
        self.assertEqual(hBBox.getMinY(), sBbox.start[-2])
        self.assertEqual(hBBox.getMaxX(), sBbox.stop[-1]-1)
        self.assertEqual(hBBox.getMaxY(), sBbox.stop[-2]-1)

        peaks = hFoot.getPeaks()
        self.assertEqual(len(peaks), 1)
        hPeak = peaks[0]
        self.assertEqual(hPeak.getIx(), coords[0][1])
        self.assertEqual(hPeak.getIy(), coords[0][0])

        # Test Model to Heavy
        filters = [f for f in "grizy"]
        hFoot = mes.modelToHeavy(src, filters, bbox.getMin(), observation)
        hModel = hFoot.getImage(fill=0).image.array

        self.assertEqual(bbox, hFoot.getBBox())
        self.assertFloatsAlmostEqual(hModel, observation.render(src.get_model()), rtol=1e-4, atol=1e-4)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
