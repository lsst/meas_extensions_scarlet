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

import numpy as np
import scipy.signal
import scarlet

from lsst.geom import Box2I, Point2I, Extent2I
from lsst.afw.geom import Span, SpanSet
from lsst.afw.detection import Footprint


def numpyToStack(images, center, offset):
    """Convert numpy and python objects to stack objects
    """
    cy, cx = center
    bands, height, width = images.shape
    x0, y0 = offset
    bbox = Box2I(Point2I(x0, y0), Extent2I(width, height))
    spans = [Span(y, x0, x0+width-1) for y in range(y0, y0+height)]
    spanset = SpanSet(spans)
    foot = Footprint(spanset)
    foot.addPeak(cx+x0, cy+y0, images[:, cy, cx].max())
    peak = foot.getPeaks()[0]
    return foot, peak, bbox


def initData(shape, coords, amplitudes=None, convolve=True):
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
