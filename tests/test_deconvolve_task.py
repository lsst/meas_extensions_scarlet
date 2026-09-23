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
import unittest.mock
import warnings

import lsst.afw.detection as afwDetection
import lsst.afw.image as afwImage
import lsst.geom as geom
from lsst.afw.detection import GaussianPsf
import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np
from lsst.meas.extensions.scarlet.deconvolveExposureTask import (
    DeconvolveExposureTask,
    calculateUpdateStep,
    calculate_update_step,
)
from lsst.meas.extensions.scarlet import ScarletStitchedPsf
from lsst.meas.extensions.scarlet.scarletDeblendTask import ScarletDeblendTask

import pipeline
from scenes import SCENES
from utils import makeStitchedPsf


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
        """Deconvolved output preserves bbox and WCS, and carries the
        model-frame PSF.

        For each band, the deconvolved exposure has the same bounding
        box as its input coadd and the same WCS (deconvolution is
        per-pixel, no geometric change). Its PSF, however, is the narrow
        scarlet model PSF the image was deconvolved to -- not the input
        coadd's wider observed PSF -- and attaching it must leave the
        input coadd's own PSF untouched.
        """
        image = pipeline.build_image(SCENES["multi-blend"])
        detection = pipeline.detect(image)
        deconv = pipeline.deconvolve(detection)

        model_psf = scl.utils.integrated_circular_gaussian(sigma=0.8)

        for band in image.bands:
            in_exp = image.mCoadd[band]
            out_exp = deconv.mDeconvolved[band]
            self.assertEqual(out_exp.getBBox(), in_exp.getBBox())
            self.assertEqual(out_exp.getWcs(), in_exp.getWcs())

            # The output PSF is the model-frame PSF, not the observed one.
            out_psf = out_exp.getPsf()
            self.assertIsNotNone(out_psf)
            np.testing.assert_array_almost_equal(
                out_psf.computeKernelImage(out_psf.getAveragePosition()).array,
                model_psf,
            )
            # The input coadd's observed PSF is left intact.
            in_psf = in_exp.getPsf()
            self.assertFalse(
                np.array_equal(
                    in_psf.computeKernelImage(in_psf.getAveragePosition()).array,
                    model_psf,
                )
            )

    def test_deconvolve_fista_and_zero_init_knobs(self):
        """The ``useFista`` and ``useZeroInit`` knobs change the solver
        without breaking the fit.

        ``_deconvolve`` defaults to plain gradient ascent initialized
        with the observed image. ``useFista`` swaps in the accelerated
        proximal-gradient (FISTA) solver and ``useZeroInit`` starts the
        iterate at zero instead. All four combinations must converge to
        a finite, non-negative model that respects the supplied
        footprint mask, and FISTA must reach a log-likelihood no worse
        than plain gradient ascent.
        """
        image = pipeline.build_image(SCENES["multi-blend"])
        detection = pipeline.detect(image)
        band = image.bands[0]
        coadd = image.mCoadd[band]

        bbox = coadd.getBBox()
        width, height = bbox.getDimensions()
        x0, y0 = bbox.getMin()
        footprintImage = afwDetection.footprintsToNumpy(
            detection.catalog, shape=(height, width), xy0=(x0, y0)
        )

        losses = {}
        for useFista in (False, True):
            for useZeroInit in (False, True):
                config = DeconvolveExposureTask.ConfigClass()
                config.useFista = useFista
                config.useZeroInit = useZeroInit
                task = DeconvolveExposureTask(config=config)
                observation = task._buildObservation(coadd, detection.catalog, band)
                model, loss = task._deconvolve(observation, footprintImage=footprintImage)

                self.assertTrue(np.all(np.isfinite(model.data)))
                # The non-negativity proximal operator holds.
                self.assertGreaterEqual(model.data.min(), 0)
                # Flux only lives inside the input footprints.
                self.assertTrue(np.all(model.data[0][footprintImage == 0] == 0))
                losses[(useFista, useZeroInit)] = loss[-1]

        # FISTA reaches a log-likelihood at least as high as plain
        # gradient ascent from the same initialization.
        for useZeroInit in (False, True):
            self.assertGreaterEqual(
                losses[(True, useZeroInit)],
                losses[(False, useZeroInit)] - 1e-6,
            )

    def test_deconvolve_zero_init_starts_from_zero(self):
        """``useZeroInit`` controls the first iterate of the solver.

        With ``useZeroInit=True`` the deconvolved image is seeded with
        zeros; with ``useZeroInit=False`` (the default) it is seeded with
        the observed image. A single iteration is enough to distinguish
        them: after one gradient step the two seeds have not yet
        converged, so their models differ.
        """
        image = pipeline.build_image(SCENES["multi-blend"])
        detection = pipeline.detect(image)
        band = image.bands[0]
        coadd = image.mCoadd[band]

        models = {}
        for useZeroInit in (False, True):
            config = DeconvolveExposureTask.ConfigClass()
            config.maxIter = 1
            config.minIter = 0
            config.useZeroInit = useZeroInit
            task = DeconvolveExposureTask(config=config)
            observation = task._buildObservation(coadd, detection.catalog, band)
            model, _ = task._deconvolve(observation)
            models[useZeroInit] = model.data

        self.assertFalse(np.allclose(models[True], models[False]))

    def test_deconvolve_breaks_on_nonfinite_residual(self):
        """The deconvolution loop stops early when every residual
        pixel is non-finite rather than running every iteration to
        ``maxIter`` on meaningless data.

        The original loop computed ``loss = -0.5 * np.sum(residual**2)``
        with no NaN guard, so a single NaN in ``residual`` poisoned
        every subsequent ``loss`` entry; both the convergence test
        ``np.abs(loss[-1] - loss[-2]) < eRel * np.abs(loss[-1])`` and
        the divergence test ``loss[-1] < loss[-2]`` return ``False``
        for NaN, so neither convergence nor step-halving triggered
        and the loop ran to ``maxIter`` on garbage. The fix moves to
        ``np.nansum`` for partial-NaN robustness and breaks the loop
        when ``residual`` is entirely non-finite.

        The production ``_buildObservation`` sanitizes both
        ``images`` and ``weights``, so this test bypasses it and
        constructs an ``scl.Observation`` with all-NaN images
        directly. It stands in for any mid-iteration scenario where
        ``convolve`` produces NaN everywhere (numerical artifacts,
        degenerate PSF). Regression test for finding C-11 of the
        ``audits/audit-2026-05-05.md`` audit.
        """
        config = DeconvolveExposureTask.ConfigClass()
        config.maxIter = 20
        config.minIter = 0
        task = DeconvolveExposureTask(config=config)

        shape = (1, 8, 8)
        psf = scl.utils.integrated_circular_gaussian(sigma=0.8).astype(
            np.float32
        )
        observation = scl.Observation(
            images=np.full(shape, np.nan, dtype=np.float32),
            variance=np.ones(shape, dtype=np.float32),
            weights=np.ones(shape, dtype=np.float32),
            psfs=psf[None],
            model_psf=psf[None],
            bands=("dummy",),
            convolution_mode="fft",
        )

        _, loss = task._deconvolve(observation)

        self.assertLess(len(loss), config.maxIter)
        self.assertFalse(np.isfinite(loss[-1]))

    def test_calculate_update_step_lipschitz(self):
        """``calculateUpdateStep`` returns the Lipschitz FISTA step
        ``safety / L`` with ``L = max_k |K_hat(k)|**2``.

        When the observed PSF equals the model PSF the difference kernel
        is the identity (a delta), so its optical transfer function is
        unity everywhere and ``L = 1`` exactly. The step is then the
        ``safety`` factor itself, and because the step is ``safety / L``
        it scales linearly with ``safety``.
        """
        shape = (1, 32, 32)
        psf = scl.utils.integrated_circular_gaussian(sigma=0.8).astype(np.float32)
        # Observed PSF == model PSF -> identity difference kernel -> L = 1.
        observation = scl.Observation(
            images=np.ones(shape, dtype=np.float32),
            variance=np.ones(shape, dtype=np.float32),
            weights=np.ones(shape, dtype=np.float32),
            psf=scl.ImagePsf(psf[None]),
            model_psf=scl.ImagePsf(psf[None]),
            bands=("dummy",),
            convolution_mode="fft",
        )

        step = calculateUpdateStep(observation)
        self.assertAlmostEqual(step, 1.0, places=5)
        self.assertTrue(np.isfinite(step))

        # The step is safety / L, so it scales linearly with safety.
        self.assertAlmostEqual(calculateUpdateStep(observation, safety=0.5), 0.5, places=5)
        self.assertAlmostEqual(
            calculateUpdateStep(observation, safety=0.5),
            0.5 * step,
            places=6,
        )

    def test_calculate_update_step_default_when_no_kernel(self):
        """``calculateUpdateStep`` returns ``safety`` when there is no
        difference kernel.

        With no PSF matching the forward operator ``A`` is the identity,
        so ``L = 1`` and the step is the bare ``safety`` factor. A
        ``Mock`` standing in for the observation supplies
        ``diff_kernel=None`` directly, exercising the early return without
        constructing a full PSF-less observation.
        """
        observation = unittest.mock.Mock(diff_kernel=None)
        self.assertEqual(calculateUpdateStep(observation), 1.0)
        self.assertEqual(calculateUpdateStep(observation, safety=0.25), 0.25)

    def test_calculate_update_step_legacy_excludes_masked_pixels(self):
        """The legacy ``calculate_update_step`` divides by the count of
        unmasked pixels rather than the full image size.

        The original sparsity/SNR scale algorithm now lives only in the
        deprecated ``calculate_update_step`` (``calculateUpdateStep``
        having moved to the Lipschitz step). An even earlier
        implementation computed ``sparsity = np.sum(signal_mask) /
        image.size``; the denominator counted every pixel in the array
        even when many of them carried zero weight (border, NO_DATA,
        BAD). On heavily masked inputs such as tract edges this
        artificially shrinks ``sparsity`` and in turn the update step.
        The retained algorithm restricts both numerator and denominator
        to pixels with non-zero weight, so the sparsity reflects the
        fraction of *valid* pixels carrying signal.

        Two observations are built that differ only in their weight
        plane: ``full`` has weights ``1`` everywhere; ``half`` masks
        the bottom half of the image (which contains no signal). A
        signal-amplitude/noise pair is chosen so the resulting scale
        does not saturate at the ``1.0`` cap. Without the mask-aware
        denominator both observations yield the same step; with it the
        masked observation yields a step that is roughly twice as large
        because the denominator halves while the signal count is
        preserved.

        Regression test for finding DC-8 of the
        ``audits/audit-2026-05-05.md`` audit.
        """
        shape = (1, 32, 32)
        noise = 1.0
        image = np.zeros(shape, dtype=np.float32)
        # 16-pixel signal block in the top-left; amplitude tuned so
        # ``scale = sparsity * sqrt(snr) / 0.1`` lands well below 1.0
        # in the unmasked case.
        image[0, :4, :4] = 5.0
        variance = np.full(shape, noise**2, dtype=np.float32)
        psf = scl.utils.integrated_circular_gaussian(sigma=0.8).astype(np.float32)

        full_weights = np.ones(shape, dtype=np.float32)
        half_weights = np.ones(shape, dtype=np.float32)
        half_weights[:, 16:, :] = 0

        def _make_obs(weights):
            return scl.Observation(
                images=image,
                variance=variance,
                weights=weights,
                psf=scl.ImagePsf(psf[None]),
                model_psf=scl.ImagePsf(psf[None]),
                bands=("dummy",),
                convolution_mode="fft",
            )

        # The legacy algorithm is deprecated; suppress its FutureWarning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            step_full = calculate_update_step(_make_obs(full_weights))
            step_half = calculate_update_step(_make_obs(half_weights))

        self.assertLess(step_full, 1.0)
        self.assertGreater(step_half, step_full)
        # The signal count is preserved across both observations and
        # the masked denominator is exactly half the full denominator,
        # so the masked step should be ~2× larger when neither caps.
        self.assertAlmostEqual(step_half / step_full, 2.0, places=5)

    def test_calculate_update_step_deprecation(self):
        """``calculate_update_step`` is deprecated and runs the legacy
        scale algorithm rather than the new Lipschitz step.

        The snake_case name is retained for one release as a deprecated
        shim. It no longer forwards to ``calculateUpdateStep``: that
        function now returns the Lipschitz FISTA step, while the legacy
        sparsity/SNR scale stayed behind under the old name. Calling it
        must emit a single ``FutureWarning`` pointing at the new name and
        return the legacy scale, which differs from the new step.

        Regression test for finding DC-10 of the
        ``audits/audit-2026-05-05.md`` audit.
        """
        shape = (1, 32, 32)
        image = np.zeros(shape, dtype=np.float32)
        # A signal block so the legacy scale lands strictly between its
        # ``min_scale`` floor and the ``1.0`` cap -- a genuinely computed
        # value, not a clamped default.
        image[0, :4, :4] = 5.0
        psf = scl.utils.integrated_circular_gaussian(sigma=0.8).astype(np.float32)
        observation = scl.Observation(
            images=image,
            variance=np.ones(shape, dtype=np.float32),
            weights=np.ones(shape, dtype=np.float32),
            psf=scl.ImagePsf(psf[None]),
            model_psf=scl.ImagePsf(psf[None]),
            bands=("dummy",),
            convolution_mode="fft",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = calculate_update_step(observation)

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, FutureWarning)
        ]
        self.assertEqual(len(deprecation_warnings), 1)
        self.assertIn("calculateUpdateStep", str(deprecation_warnings[0].message))

        # The legacy scale is a genuine, in-range result of the old
        # algorithm and no longer matches the new Lipschitz step.
        self.assertTrue(np.isfinite(legacy))
        self.assertGreater(legacy, 0.01)
        self.assertLess(legacy, 1.0)
        self.assertNotAlmostEqual(legacy, calculateUpdateStep(observation))

    def test_model_to_exposure_decouples_mask_and_variance(self):
        """``_modelToExposure`` detaches the output mask/variance from
        the input coadd and invalidates the variance plane.

        Convolution-then-deconvolution changes the per-pixel noise
        covariance, so the input coadd's variance plane no longer
        corresponds to the pixel values of the deconvolved model;
        propagating it unchanged would advertise an incorrect variance
        as if it were valid. The previous implementation also aliased
        the output's mask and variance to the input coadd's by
        reference, so any downstream mutation of the deconvolved
        exposure's mask/variance would silently leak back into the
        input.

        The output exposure now carries (a) a deep-copied mask and
        (b) a fresh ``inf``-filled variance plane signalling "no
        information about the noise here". Regression test for finding
        C-12 of the ``audits/audit-2026-05-05.md`` audit.
        """
        bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(16, 16))
        coadd = afwImage.ExposureF(bbox)
        coadd.image.array[:] = 1.0
        coadd.variance.array[:] = 5.0
        edge_bit = coadd.mask.getPlaneBitMask("EDGE")
        coadd.mask.array[0, 0] = edge_bit

        # The input coadd carries an observed PSF that must not be
        # disturbed when the output adopts the model-frame PSF.
        coadd.setPsf(GaussianPsf(15, 15, 1.0))

        task = DeconvolveExposureTask()
        model = np.full((16, 16), 2.0, dtype=coadd.image.array.dtype)
        modelPsf = scl.ImagePsf(
            scl.utils.integrated_circular_gaussian(sigma=0.8)[None]
        )
        out = task._modelToExposure(model, coadd, modelPsf)

        # Pre-existing mask bits survive the copy.
        self.assertTrue(out.mask.array[0, 0] & edge_bit != 0)
        # Variance plane is invalidated by filling with inf.
        np.testing.assert_array_equal(out.variance.array, np.inf)

        # The output carries the model-frame PSF, converted to an LSST PSF.
        outPsf = out.getPsf()
        self.assertIsNotNone(outPsf)
        np.testing.assert_array_almost_equal(
            outPsf.computeKernelImage(outPsf.getAveragePosition()).array,
            modelPsf.data[0],
        )

        # Mutating the output mask/variance does not affect the input.
        out.mask.array[5, 5] |= edge_bit
        out.variance.array[5, 5] = 999.0
        self.assertEqual(coadd.mask.array[5, 5], 0)
        self.assertEqual(coadd.variance.array[5, 5], 5.0)
        # Attaching the model PSF to the output left the input's PSF intact.
        self.assertEqual(coadd.getPsf().getSigma(), 1.0)

    def test_build_observation_stitched_psf(self):
        """A coadd carrying a ``StitchedPsf`` builds a stitched observation.

        When the input is a cell-based coadd (its PSF is an
        ``lsst.cell_coadds.StitchedPsf``), ``_buildObservation`` builds a
        spatially-varying ``ScarletStitchedPsf`` over the cell grid instead of
        wrapping a single kernel image. A flat coadd (a ``GaussianPsf``) still
        takes the constant ``ImagePsf`` path. Both observed PSFs match the
        coadd dtype, so the difference kernel is non-trivial.
        """
        cell, grid = 15, 2
        size = cell * grid
        bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(size, size))
        coadd = afwImage.ExposureF(bbox)
        rng = np.random.RandomState(5)
        coadd.image.array[:] = rng.rand(size, size).astype(np.float32)
        coadd.variance.array[:] = 1.0

        task = DeconvolveExposureTask()

        coadd.setPsf(makeStitchedPsf(sigma=1.2, cell=cell, grid=grid))
        observation = task._buildObservation(coadd, catalog=None, band="g")
        self.assertIsInstance(observation.psf, ScarletStitchedPsf)
        self.assertIsInstance(observation.diff_kernel, ScarletStitchedPsf)
        self.assertEqual(observation.psf.dtype, coadd.image.array.dtype)

        # A flat coadd keeps the constant-PSF path.
        coadd.setPsf(GaussianPsf(11, 11, 1.2))
        flat = task._buildObservation(coadd, catalog=None, band="g")
        self.assertIsInstance(flat.psf, scl.ImagePsf)

    def test_build_observation_stitched_psf_opt_out(self):
        """``useStitchedPsf=False`` forces the flat path on a cell coadd.

        A coadd whose PSF is a ``StitchedPsf`` normally builds a
        ``ScarletStitchedPsf``; with ``useStitchedPsf=False`` it instead wraps
        a single PSF kernel evaluated at the image center as a constant
        ``ImagePsf``, the faster (less accurate) option.
        """
        cell, grid = 15, 2
        size = cell * grid
        bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(size, size))
        coadd = afwImage.ExposureF(bbox)
        rng = np.random.RandomState(5)
        coadd.image.array[:] = rng.rand(size, size).astype(np.float32)
        coadd.variance.array[:] = 1.0
        coadd.setPsf(makeStitchedPsf(sigma=1.2, cell=cell, grid=grid))

        config = DeconvolveExposureTask.ConfigClass()
        config.useStitchedPsf = False
        task = DeconvolveExposureTask(config=config)
        observation = task._buildObservation(coadd, catalog=None, band="g")
        self.assertIsInstance(observation.psf, scl.ImagePsf)
        self.assertNotIsInstance(observation.psf, ScarletStitchedPsf)

    def test_model_psf_sigma_config(self):
        """``modelPsfSigma`` controls the model-frame PSF width.

        The default builds the same kernel as ``sigma=0.8``; setting the
        config to another value rebuilds the observation's model PSF and,
        through ``run``, the PSF attached to the deconvolved exposure.
        """
        bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(16, 16))
        coadd = afwImage.ExposureF(bbox)
        coadd.image.array[:] = 0.0
        coadd.image.array[8, 8] = 100.0
        coadd.variance.array[:] = 1.0
        coadd.setPsf(GaussianPsf(15, 15, 1.0))

        # The default reproduces the historical sigma=0.8 kernel.
        default = DeconvolveExposureTask()
        observation = default._buildObservation(coadd, catalog=None, band="g")
        np.testing.assert_array_almost_equal(
            observation.model_psf.data[0],
            scl.utils.integrated_circular_gaussian(sigma=0.8),
        )

        # A non-default sigma propagates to both the observation model PSF
        # and the PSF attached to the deconvolved exposure.
        config = DeconvolveExposureTask.ConfigClass()
        config.modelPsfSigma = 1.5
        task = DeconvolveExposureTask(config=config)
        observation = task._buildObservation(coadd, catalog=None, band="g")
        expected = scl.utils.integrated_circular_gaussian(sigma=1.5)
        np.testing.assert_array_almost_equal(observation.model_psf.data[0], expected)

        result = task.run(coadd, catalog=None, band="g")
        outPsf = result.deconvolved.getPsf()
        np.testing.assert_array_almost_equal(
            outPsf.computeKernelImage(outPsf.getAveragePosition()).array,
            expected,
        )

    def test_deconvolve_stitched_psf_end_to_end(self):
        """``run`` deconvolves a cell-coadd (stitched-PSF) exposure.

        Exercises the spatially-varying forward convolution and its adjoint
        (the gradient pass) through the full deconvolution loop, confirming the
        stitched PSF drops into the optimizer the same way a constant PSF does.
        The recovered model is finite and preserves the input bounding box.
        """
        cell, grid = 15, 2
        size = cell * grid
        bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(size, size))
        coadd = afwImage.ExposureF(bbox)
        # A single bright source so the deconvolver has signal to recover.
        coadd.image.array[:] = 0.0
        coadd.image.array[size // 2, size // 2] = 100.0
        coadd.variance.array[:] = 1.0
        coadd.setPsf(makeStitchedPsf(sigma=1.4, cell=cell, grid=grid))

        task = DeconvolveExposureTask()
        result = task.run(coadd, catalog=None, band="g")

        self.assertEqual(result.deconvolved.getBBox(), bbox)
        self.assertTrue(np.all(np.isfinite(result.deconvolved.image.array)))
        # The deconvolver concentrates flux near the source center.
        self.assertGreater(result.deconvolved.image.array[size // 2, size // 2], 0)

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
