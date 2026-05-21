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

import lsst.afw.image as afwImage
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

    def test_deconvolve_one_isolated_psf(self):
        """Deconvolving a single isolated PSF source conserves flux.

        Integrated flux in a 5×5 box around the source center matches
        the truth within 2σ of expected sum-of-noise (≈ √25·σ) in
        every band. The deconvolver redistributes flux between
        adjacent pixels — per-pixel peak amplitude can drift by
        several σ — but the total flux over the source's support is
        preserved. See audit finding DC-1 for the per-pixel
        amplitude residual.
        """
        scene = SCENES["one_isolated_psf"]
        image = pipeline.build_image(scene)
        detection = pipeline.detect(image)
        deconv = pipeline.deconvolve(detection)

        # The truth image's ``yx0`` is the offset of the local array
        # within the scene's global coordinate frame, so model centers
        # must be translated before indexing into the array.
        yx0_y, yx0_x = image.deconvolved.yx0
        expected_noise = np.sqrt(25) * np.std(image.noise)
        for model in scene.models:
            yc, xc = model.center
            lyc, lxc = yc - yx0_y, xc - yx0_x
            box = (slice(None), slice(lyc - 2, lyc + 3), slice(lxc - 2, lxc + 3))
            truth_flux = image.deconvolved.data[box].sum(axis=(1, 2))
            recovered_flux = deconv.mDeconvolved.image.array[box].sum(axis=(1, 2))
            for b, band in enumerate(image.bands):
                self.assertLess(
                    abs(recovered_flux[b] - truth_flux[b]),
                    2 * expected_noise,
                    f"{band} band: flux not conserved "
                    f"(diff {recovered_flux[b] - truth_flux[b]:.4f}, "
                    f"limit {2 * expected_noise:.4f})",
                )

    def test_deconvolve_preserves_image_metadata(self):
        """Deconvolved output preserves bbox, PSF, and WCS from input.

        For each band, the deconvolved exposure has the same bounding
        box as its input coadd, the same WCS (deconvolution is per-pixel,
        no geometric change), and a PSF whose kernel image matches the
        input PSF's (the task does not synthesize a new PSF).
        """
        image = pipeline.build_image(SCENES["multi-blend"])
        detection = pipeline.detect(image)
        deconv = pipeline.deconvolve(detection)

        for band in image.bands:
            in_exp = image.mCoadd[band]
            out_exp = deconv.mDeconvolved[band]
            self.assertEqual(out_exp.getBBox(), in_exp.getBBox())
            self.assertEqual(out_exp.getWcs(), in_exp.getWcs())

            out_psf = out_exp.getPsf()
            self.assertIsNotNone(out_psf)
            in_psf = in_exp.getPsf()
            np.testing.assert_array_equal(
                out_psf.computeImage(out_psf.getAveragePosition()).array,
                in_psf.computeImage(in_psf.getAveragePosition()).array,
            )

    def test_deconvolve_with_nan_input(self):
        """A NaN pixel in the input does not propagate to the
        deconvolved output.

        The NaN is inserted at the center of the first source so it
        falls inside a detected footprint and is actually processed by
        the deconvolution (rather than being skipped as outside all
        footprints). Audit area C-11 lives here; if the contract
        changes, this test moves with it.
        """
        image = pipeline.build_image(SCENES["multi-blend"])
        detection = pipeline.detect(image)

        # Clone the multiband exposure so the NaN insertion does not
        # pollute the cached input shared with other tests.
        cloned = afwImage.MultibandExposure.fromExposures(
            image.bands,
            [image.mCoadd[band].clone() for band in image.bands],
        )
        yc, xc = SCENES["multi-blend"].models[0].center
        for band in image.bands:
            cloned[band].image.array[yc, xc] = np.nan

        # Run the deconvolve task directly; ``pipeline.deconvolve`` is
        # cached and would otherwise reuse the un-mutated input.
        task = DeconvolveExposureTask()
        for band in image.bands:
            result = task.run(cloned[band], detection.catalog)
            self.assertFalse(
                np.any(np.isnan(result.deconvolved.image.array)),
                f"NaN propagated to deconvolved output in {band} band",
            )


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
