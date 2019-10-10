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
        frame = lmeScarlet.LsstFrame(shape)
        observation = lmeScarlet.LsstObservation(images)
        bgRms = np.ones_like(trueSed) * 1e-3

        # init stack objects
        foot, peak, bbox = numpyToStack(images, center, (15, 3))
        # init source
        src = lmeScarlet.LsstSource(frame, peak, observation, bgRms, bbox)

        self.assertFloatsAlmostEqual(src.pixel_center, center)
        self.assertEqual(src.symmetric, True)
        self.assertEqual(src.monotonic, True)
        self.assertEqual(src.center_step, 5)
        self.assertEqual(src.delay_thresh, 10)

        self.assertFloatsAlmostEqual(src.sed/3, trueSed)
        self.assertFloatsAlmostEqual(src.morph*3, trueMorph, rtol=1e-7)
        self.assertEqual(src.detectedPeak, peak)
        self.assertEqual(foot.getBBox(), bbox)

    def test_get_model(self):
        shape = (6, 31, 55)
        B, Ny, Nx = shape
        coords = [(20, 10), (10, 30), (17, 42)]
        result = initData(shape, coords, [3, 2, 1])
        targetPsfImage, psfImages, images, channels, seds, morphs, targetPsf, psfs = result

        frame = lmeScarlet.LsstFrame(shape, psfs=targetPsfImage[None])
        observation = lmeScarlet.LsstObservation(images, psfs=psfImages).match(frame)
        bgRms = np.ones((B, )) * 1e-3
        foot, peak, bbox = numpyToStack(images, coords[0], (15, 3))
        src = lmeScarlet.LsstSource(frame, peak, observation, bgRms, bbox)
        sedScale = targetPsfImage.max() / observation.frame.psfs.max(axis=(1, 2))
        # Use the correct SED and morphology. This will be different than
        # the ones ExtenedSource initializes to, since the morphology cannot
        # be initialized with the correct PSF (yet)
        src._sed = seds[0] * sedScale
        src._morph = morphs[0]
        model = src.get_model(observation=observation)
        truth = observation.render((seds[0]*sedScale)[:, None, None] * morphs[0][None, :, :])
        self.assertFloatsAlmostEqual(model, truth)

        # Test without passing an observation to get_model
        model = src.get_model()
        truth = (seds[0]*sedScale)[:, None, None] * morphs[0][None, :, :]
        self.assertFloatsAlmostEqual(model, truth)

    def test_to_heavy(self):
        shape = (5, 31, 55)
        B, Ny, Nx = shape
        coords = [(20, 10), (10, 30), (17, 42)]
        result = initData(shape, coords, [3, 2, 1])
        targetPsfImage, psfImages, images, channels, seds, morphs, targetPsf, psfs = result

        frame = lmeScarlet.LsstFrame(shape, psfs=targetPsfImage[None])
        observation = lmeScarlet.LsstObservation(images, psfs=psfImages).match(frame)
        bgRms = np.ones((B, )) * 1e-3
        foot, peak, bbox = numpyToStack(images, coords[0], (15, 3))
        src = lmeScarlet.LsstSource(frame, peak, observation, bgRms, bbox)
        # Get the HeavyFootprint
        peakSchema = PeakTable.makeMinimalSchema()
        src._sed = src._sed.astype(np.float32)
        src._morph = src._morph.astype(np.float32)
        hFoot = src.morphToHeavy(peakSchema=peakSchema)
        hBBox = hFoot.getBBox()

        hMorph = multiband.heavyFootprintToImage(hFoot, fill=0).image.array
        sBbox = scarlet.bbox.trim(src.morph)
        self.assertFloatsAlmostEqual(hMorph, src.morph[sBbox.slices])
        self.assertEqual(hBBox.getMinX(), sBbox.left)
        self.assertEqual(hBBox.getMinY(), sBbox.bottom)
        self.assertEqual(hBBox.getMaxX(), sBbox.right)
        self.assertEqual(hBBox.getMaxY(), sBbox.top)

        peaks = hFoot.getPeaks()
        self.assertEqual(len(peaks), 1)
        hPeak = peaks[0]
        self.assertEqual(hPeak.getIx(), coords[0][1])
        self.assertEqual(hPeak.getIy(), coords[0][0])

        # Test Model to Heavy
        filters = [f for f in "grizy"]
        hFoot = src.modelToHeavy(filters, bbox.getMin(), observation)
        hModel = hFoot.getImage(fill=0).image.array

        self.assertEqual(bbox, hFoot.getBBox())
        self.assertFloatsAlmostEqual(hModel, src.get_model(observation=observation), rtol=1e-4, atol=1e-4)


class TestLsstBlend(lsst.utils.tests.TestCase):
    def test_fit_pointSource(self):
        # This is a test from scarlet,
        # but we implement it here to test the `pointSource`
        # method of `LsstSource` along with fitting the blend.
        shape = (6, 31, 55)
        coords = [(20, 10), (10, 30), (17, 42)]
        amplitudes = [3, 2, 1]
        result = initData(shape, coords, amplitudes)
        targetPsfImage, psfImages, images, channels, seds, morphs, targetPsf, psfs = result
        B, Ny, Nx = shape

        frame = lmeScarlet.LsstFrame(shape, psfs=targetPsfImage[None], dtype=np.float)
        observation = lmeScarlet.LsstObservation(images, psfs=psfImages).match(frame)
        bgRms = np.ones((B, )) * 1e-3
        sources = []
        for coord in coords:
            foot, peak, bbox = numpyToStack(images, coord, (15, 3))
            sources.append(lmeScarlet.LsstSource(frame, peak, observation, bgRms, bbox, pointSource=True))
        blend = lmeScarlet.Blend(sources, observation)
        # Try to run for 10 iterations
        # Since the model is already near exact, it should converge
        # on the 2nd iteration (since it doesn't calculate the initial loss)
        blend.fit(10)

        self.assertEqual(blend.it, 2)
        self.assertFloatsAlmostEqual(blend.L_sed, 2.5481250470053265, rtol=1e-5, atol=1e-5)
        self.assertFloatsAlmostEqual(blend.L_morph, 9024.538938935855, rtol=1e-5, atol=1e-5)
        self.assertTrue(blend.mse[0] > blend.mse[1])

    def test_get_model(self):
        shape = (6, 31, 55)
        coords = [(20, 10), (10, 30), (17, 42)]
        amplitudes = [3, 2, 1]
        result = initData(shape, coords, amplitudes)
        targetPsfImage, psfImages, images, channels, seds, morphs, targetPsf, psfs = result
        B, Ny, Nx = shape

        frame = lmeScarlet.LsstFrame(shape, psfs=targetPsfImage[None])
        observation = lmeScarlet.LsstObservation(images, psfs=psfImages).match(frame)
        bgRms = np.ones((B, )) * 1e-3
        sources = []
        for coord in coords:
            foot, peak, bbox = numpyToStack(images, coord, (15, 3))
            sources.append(lmeScarlet.LsstSource(frame, peak, observation, bgRms, bbox, pointSource=True))
        blend = lmeScarlet.LsstBlend(sources, observation)

        self.assertEqual(len(blend.observations), 1)
        self.assertEqual(blend.observations[0], observation)
        self.assertEqual(blend.mse, [])
        model = blend.get_model(observation=observation)
        self.assertFloatsAlmostEqual(model, images, rtol=1e-5, atol=1e-5)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
