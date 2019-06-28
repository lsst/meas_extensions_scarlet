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
import scipy.signal
import scarlet

import lsst.meas.extensions.scarlet as lmeScarlet
import lsst.utils.tests
from lsst.geom import Box2I, Point2I, Extent2I
from lsst.afw.geom import Span, SpanSet
from lsst.afw.detection import Footprint, PeakTable, multiband


def numpy_to_stack(images, center, offset):
    """Convert numpy and python objects to stack objects
    """
    cy, cx = center
    bands, height, width = images.shape
    x0, y0 = offset
    full_image = np.ones((bands, height+10, width+20))
    bbox = Box2I(Point2I(x0, y0), Extent2I(width, height))
    full_image[:, y0:y0+height, x0:x0+width] = images
    spans = [Span(y, x0, x0+width-1) for y in range(y0, y0+height)]
    spanset = SpanSet(spans)
    foot = Footprint(spanset)
    foot.addPeak(cx+x0, cy+y0, images[:, cy, cx].max())
    peak = foot.getPeaks()[0]
    return foot, peak, bbox


def init_data(shape, coords, amplitudes=None, convolve=True):
    """Initialize data for the tests
    """

    B, Ny, Nx = shape
    K = len(coords)

    if amplitudes is None:
        amplitudes = np.ones((K,))
    assert K == len(amplitudes)

    _seds = [
        np.arange(B, dtype=float),
        np.arange(B, dtype=float)[::-1],
        np.ones((B,), dtype=float)
    ]
    seds = np.array([_seds[n % 3]*amplitudes[n] for n in range(K)])

    morphs = np.zeros((K, Ny, Nx))
    for k, coord in enumerate(coords):
        morphs[k, coord[0], coord[1]] = 1
    images = seds.T.dot(morphs.reshape(K, -1)).reshape(shape)

    if convolve:
        psf_radius = 20
        psf_shape = (2*psf_radius+1, 2*psf_radius+1)
        psf_center = (psf_radius, psf_radius)
        target_psf = scarlet.psf.generate_psf_image(scarlet.psf.gaussian, psf_shape, psf_center,
                                                    amplitude=1, sigma=.9)
        target_psf /= target_psf.sum()

        psfs = np.array([scarlet.psf.generate_psf_image(scarlet.psf.gaussian, psf_shape, psf_center,
                                                        amplitude=1, sigma=1+.2*b) for b in range(B)])
        # Convolve the image with the psf in each channel
        # Use scipy.signal.convolve without using FFTs as a sanity check
        images = np.array([scipy.signal.convolve(img, psf, method="direct", mode="same")
                           for img, psf in zip(images, psfs)])
        # Convolve the true morphology with the target PSF,
        # also using scipy.signal.convolve as a sanity check
        morphs = np.array([scipy.signal.convolve(m, target_psf, method="direct", mode="same")
                           for m in morphs])
        morphs /= morphs.max()
        psfs /= psfs.sum(axis=(1, 2))[:, None, None]

    channels = range(len(images))
    return target_psf, psfs, images, channels, seds, morphs


class TestLsstSource(lsst.utils.tests.TestCase):
    def test_init(self):
        # Initialize the model
        shape = (5, 31, 55)
        B, Ny, Nx = shape

        x = np.linspace(-2, 2, 5)
        y = np.linspace(-2, 2, 5)
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)

        true_sed = np.arange(B)
        true_morph = np.zeros(shape[1:])

        center = (np.array(true_morph.shape) - 1) // 2
        cy, cx = center
        true_morph[cy-2:cy+3, cx-2:cx+3] = 3-r

        morph = true_morph.copy()
        # Make a point that is not monotonic or symmetric to ensure
        # that it is supressed.
        morph[5, 3] = 10

        # Create the scarlet objects
        images = true_sed[:, None, None] * morph[None, :, :]
        frame = lmeScarlet.LsstFrame(shape)
        observation = lmeScarlet.LsstObservation(images)
        bg_rms = np.ones_like(true_sed) * 1e-3

        # init stack objects
        foot, peak, bbox = numpy_to_stack(images, center, (15, 3))
        # init source
        src = lmeScarlet.LsstSource(frame, peak, observation, bg_rms, bbox)

        self.assertFloatsAlmostEqual(src.pixel_center, center)
        self.assertEqual(src.symmetric, False)
        self.assertEqual(src.monotonic, True)
        self.assertEqual(src.center_step, 5)
        self.assertEqual(src.delay_thresh, 10)

        self.assertFloatsAlmostEqual(src.sed/3, true_sed)
        self.assertFloatsAlmostEqual(src.morph*3, true_morph)
        self.assertEqual(src.detectedPeak, peak)
        self.assertEqual(foot.getBBox(), bbox)

    def test_get_model(self):
        shape = (6, 31, 55)
        B, Ny, Nx = shape
        coords = [(20, 10), (10, 30), (17, 42)]
        target_psf, psfs, images, channels, seds, morphs = init_data(shape, coords, [3, 2, 1])

        frame = lmeScarlet.LsstFrame(shape, psfs=target_psf[None])
        observation = lmeScarlet.LsstObservation(images, psfs=psfs).match(frame)
        bg_rms = np.ones((B, )) * 1e-3
        foot, peak, bbox = numpy_to_stack(images, coords[0], (15, 3))
        src = lmeScarlet.LsstSource(frame, peak, observation, bg_rms, bbox)
        sed_scale = target_psf.max() / observation.frame.psfs.max(axis=(1, 2))
        # Use the correct SED and morphology. This will be different than
        # the ones ExtenedSource initializes to, since the morphology cannot
        # be initialized with the correct PSF (yet)
        src._sed = seds[0] * sed_scale
        src._morph = morphs[0]
        model = src.get_model(observation=observation)
        truth = observation.render((seds[0]*sed_scale)[:, None, None] * morphs[0][None, :, :])
        self.assertFloatsAlmostEqual(model, truth)

        # Test without passing an observation to get_model
        model = src.get_model()
        truth = (seds[0]*sed_scale)[:, None, None] * morphs[0][None, :, :]
        self.assertFloatsAlmostEqual(model, truth)

    def test_to_heavy(self):
        shape = (5, 31, 55)
        B, Ny, Nx = shape
        coords = [(20, 10), (10, 30), (17, 42)]
        target_psf, psfs, images, channels, seds, morphs = init_data(shape, coords, [3, 2, 1])

        frame = lmeScarlet.LsstFrame(shape, psfs=target_psf[None])
        observation = lmeScarlet.LsstObservation(images, psfs=psfs).match(frame)
        bg_rms = np.ones((B, )) * 1e-3
        foot, peak, bbox = numpy_to_stack(images, coords[0], (15, 3))
        src = lmeScarlet.LsstSource(frame, peak, observation, bg_rms, bbox)
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
        self.assertFloatsAlmostEqual(hModel, src.get_model(observation=observation), rtol=1e-7, atol=1e-7)


class TestLsstBlend(lsst.utils.tests.TestCase):
    def test_fit_point_source(self):
        # This is a test from scarlet,
        # but we implement it here to test the `point_source`
        # method of `LsstSource` along with fitting the blend.
        shape = (6, 31, 55)
        coords = [(20, 10), (10, 30), (17, 42)]
        amplitudes = [3, 2, 1]
        target_psf, psfs, images, channels, seds, morphs = init_data(shape, coords, amplitudes)
        B, Ny, Nx = shape

        frame = lmeScarlet.LsstFrame(shape, psfs=target_psf[None])
        observation = lmeScarlet.LsstObservation(images, psfs=psfs).match(frame)
        bg_rms = np.ones((B, )) * 1e-3
        sources = []
        for coord in coords:
            foot, peak, bbox = numpy_to_stack(images, coord, (15, 3))
            sources.append(lmeScarlet.LsstSource(frame, peak, observation, bg_rms, bbox, point_source=True))
        blend = lmeScarlet.Blend(sources, observation)
        # Try to run for 10 iterations
        # Since the model is already near exact, it should converge
        # on the 2nd iteration (since it doesn't calculate the initial loss)
        blend.fit(10)

        self.assertEqual(blend.it, 2)
        self.assertFloatsAlmostEqual(blend.L_sed, 2.5481250470053265, rtol=1e-10, atol=1e-10)
        self.assertFloatsAlmostEqual(blend.L_morph, 9024.538938935855)
        self.assertFloatsAlmostEqual(np.array(blend.mse),
                                     np.array([3.875628098330452e-15, 3.875598349723412e-15]))
        self.assertTrue(blend.mse[0] > blend.mse[1])

    def test_get_model(self):
        shape = (6, 31, 55)
        coords = [(20, 10), (10, 30), (17, 42)]
        amplitudes = [3, 2, 1]
        target_psf, psfs, images, channels, seds, morphs = init_data(shape, coords, amplitudes)
        B, Ny, Nx = shape

        frame = lmeScarlet.LsstFrame(shape, psfs=target_psf[None])
        observation = lmeScarlet.LsstObservation(images, psfs=psfs).match(frame)
        bg_rms = np.ones((B, )) * 1e-3
        sources = []
        for coord in coords:
            foot, peak, bbox = numpy_to_stack(images, coord, (15, 3))
            sources.append(lmeScarlet.LsstSource(frame, peak, observation, bg_rms, bbox, point_source=True))
        blend = lmeScarlet.LsstBlend(sources, observation)

        self.assertEqual(len(blend.observations), 1)
        self.assertEqual(blend.observations[0], observation)
        self.assertEqual(blend.mse, [])
        model = blend.get_model(observation=observation)
        self.assertFloatsAlmostEqual(model, images, rtol=1e-7, atol=1e-7)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
