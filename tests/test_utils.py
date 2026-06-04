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

"""Tests for ``lsst.meas.extensions.scarlet.utils``."""

import unittest
import warnings

import lsst.afw.image as afwImage
import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np
import scipy.signal
from lsst.afw.detection import Footprint, GaussianPsf, InvalidPsfError, PeakTable, Psf
from lsst.afw.geom import SpanSet
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.geom import Extent2I, Point2D, Point2I
from lsst.pipe.base import NoWorkFound


class BadPsf(Psf):
    def __init__(self, validPoint: Point2D, psf: GaussianPsf):
        self.validPoint = validPoint
        self.psf = psf
        super().__init__()

    def computeKernelImage(self, location: Point2D):
        if location == self.validPoint:
            return self.psf.computeKernelImage(location)
        raise InvalidPsfError(f"Invalid PSF at location {location}")


class MultiPointBadPsf(Psf):
    """A PSF valid at multiple discrete integer locations, with a
    potentially different underlying ``GaussianPsf`` per location.

    Used to drive the multiband-PSF fallback search through scenarios
    where the same band would have multiple acceptable fallback
    locations with distinguishable kernels.
    """

    def __init__(self, validPsfs: dict[tuple[int, int], GaussianPsf]):
        self.validPsfs = validPsfs
        super().__init__()

    def computeKernelImage(self, location: Point2D):
        key = (int(location.getX()), int(location.getY()))
        if key in self.validPsfs:
            return self.validPsfs[key].computeKernelImage(location)
        raise InvalidPsfError(f"Invalid PSF at location {location}")


class TestUtils(lsst.utils.tests.TestCase):
    def setUp(self):
        self.bands = tuple("gri")

    def test_computeNearestPsfGood(self):
        # Test that using a valid PSF works normally
        psf, psfImage = self._generateGoodPsf()
        coadd = self._generateCoadd(psf)

        # Test that computing the PSF works
        derivedPsf, center, dist = mes.utils.computeNearestPsf(coadd, None, "g", Point2D(25, 25))
        np.testing.assert_array_equal(derivedPsf.array, psfImage)
        self.assertEqual(center, Point2D(25, 25))
        self.assertEqual(dist, 0)

    def test_computeNearestPsfRecoverable(self):
        # Test that using a PSF not defined at the initial location
        # will fallback to a valid location.
        psf, psfImage = self._generateGoodPsf()
        coadd = self._generateCoadd(BadPsf(Point2D(1, 1), psf))
        catalog = self._generateCatalog(self.bands, [[(1, 1, 10)]])

        # Test that computing the PSF works after finding a new location.
        # Since the PSF above is *only* defined at (1, 1) it will fail to
        # compute a PSF image at (4, 5) but should fall back to (1, 1).
        # Per finding U-6 of the ``audits/audit-2026-05-05.md`` audit,
        # the fallback path previously returned a ``Point2I`` while the
        # direct-success path returned a ``Point2D``; both now uniformly
        # return ``Point2D`` and ``Point2D != Point2I`` even at the same
        # coordinates, so the type check below is also a regression guard
        # on the unified return type.
        derivedPsf, center, dist = mes.utils.computeNearestPsf(coadd, catalog, None, Point2D(4, 5))
        np.testing.assert_array_equal(derivedPsf.array, psfImage)
        self.assertIsInstance(center, Point2D)
        self.assertEqual(center, Point2D(1, 1))
        self.assertEqual(dist, 5)

    def test_computeNearestPsfBad(self):
        # Test that a PSF that cannot find a matching location returns None
        psf = self._generateGoodPsf()
        coadd = self._generateCoadd(BadPsf(Point2D(1, 1), psf))
        catalog = self._generateCatalog(self.bands)

        # Test that computing the PSF cannot generate a PSF
        derivedPsf, center, dist = mes.utils.computeNearestPsf(coadd, catalog, None, Point2D(4, 5))
        self.assertIsNone(derivedPsf)
        self.assertIsNone(center)
        self.assertIsNone(dist)

    def test_computeNearestPsfMultiBandGood(self):
        # Test that a valid PSF in every band works normally
        bands = tuple("gri")
        psfs, psfImage = self._generateMultibandPsf([1.0, 1.2, 1.4])
        mCoadd = self._generateMultibandCoadd(psfs, bands)

        # Test that computing the PSF works
        psfArray, newCoadd = mes.utils.computeNearestPsfMultiBand(mCoadd, Point2D(25, 25), None)
        np.testing.assert_array_equal(psfArray, psfImage)
        self.assertTupleEqual(newCoadd.bands, bands)

    def test_computeNearestPsfMultiBandRecoverable(self):
        # Test that a Psf at a different location is still recoverable
        bands = tuple("gri")
        psfs, psfImage = self._generateMultibandPsf([1.0, 1.2, 1.4])
        psfs[1] = BadPsf(Point2D(1, 1), psfs[1])
        mCoadd = self._generateMultibandCoadd(psfs, bands)
        catalog = self._generateCatalog(self.bands, [[(1, 1, 10)]])

        # Test that computing the PSF works because the catalog has a peak
        # at the location of the BadPsf.
        psfArray, newCoadd = mes.utils.computeNearestPsfMultiBand(mCoadd, Point2D(25, 25), catalog)
        np.testing.assert_array_equal(psfArray, psfImage)
        self.assertTupleEqual(newCoadd.bands, bands)

    def test_computeNearestPsfMultiBandIncomplete(self):
        # Test that missing a PSF in one band returns a PSF and
        # an exposure that are missing bands.
        bands = tuple("gri")
        psfs, psfImage = self._generateMultibandPsf([1.0, 1.2, 1.4])
        psfs[1] = BadPsf(Point2D(1, 1), psfs[1])
        mCoadd = self._generateMultibandCoadd(psfs, bands)
        catalog = self._generateCatalog(self.bands)

        # Test that computing the PSF works for the g- and i-band PSFs that
        # are not BadPsf.
        psfArray, newCoadd = mes.utils.computeNearestPsfMultiBand(mCoadd, Point2D(25, 25), catalog)
        np.testing.assert_array_equal(psfArray, np.delete(psfImage, 1, axis=0))
        self.assertTupleEqual(newCoadd.bands, tuple("gi"))

    def test_computeNearestPsfMultiBandBad(self):
        # Test that None is returned if none of the PSFs can be computed
        bands = tuple("gri")
        psfs, psfImage = self._generateMultibandPsf([1.0, 1.2, 1.4])
        psfs = [BadPsf(Point2D(1, 1), psfs) for psf in psfs]
        mCoadd = self._generateMultibandCoadd(psfs, bands)
        catalog = self._generateCatalog(self.bands)

        psfArray, newCoadd = mes.utils.computeNearestPsfMultiBand(mCoadd, Point2D(25, 25), catalog)
        self.assertIsNone(psfArray)
        self.assertIsNone(newCoadd)

    def test_computeNearestPsfMultiBand_search_center_does_not_drift(self):
        """When one band falls back, the search for subsequent bands
        stays anchored at the requested center.

        Requested center is ``(25, 25)``. Band g's PSF is valid only at
        ``(1, 1)`` (≈34 px from center). Band r's PSF is valid at both
        ``(10, 10)`` with σ=2.0 and ``(30, 30)`` with σ=1.2. Sorting the
        catalog peaks by distance from the *requested center* puts
        ``(30, 30)`` first for band r; sorting by distance from band g's
        fallback ``(1, 1)`` puts ``(10, 10)`` first. The bug carried band
        g's fallback into r's search, so r returned the σ=2.0 kernel at
        ``(10, 10)``. The fix re-anchors r's search at ``(25, 25)``, so r
        returns the σ=1.2 kernel at ``(30, 30)``. Regression test for
        finding C-6 of the ``audits/audit-2026-05-05.md`` audit.
        """
        bands = ("g", "r")
        g_inner = GaussianPsf(41, 41, 1.0)
        r_at_10 = GaussianPsf(41, 41, 2.0)
        r_at_30 = GaussianPsf(41, 41, 1.2)
        psfs = [
            BadPsf(Point2D(1, 1), g_inner),
            MultiPointBadPsf({(10, 10): r_at_10, (30, 30): r_at_30}),
        ]
        mCoadd = self._generateMultibandCoadd(psfs, bands)
        catalog = self._generateCatalog(
            bands, [[(1, 1, 10), (10, 10, 10), (30, 30, 10)]]
        )

        psfArray, newCoadd = mes.utils.computeNearestPsfMultiBand(
            mCoadd, Point2D(25, 25), catalog
        )

        self.assertTupleEqual(newCoadd.bands, bands)
        # GaussianPsf kernels are stationary so their bboxes don't depend
        # on the position they were computed at; the peak amplitude is
        # 1/(2πσ²), so each σ value has a distinct kernel max we can
        # discriminate on. Band g should land at (1, 1) (its only valid
        # location, σ=1.0); band r should land at (30, 30) (σ=1.2 — the
        # closest valid r position to the requested center). Under the
        # bug, band r would land at (10, 10) and return the σ=2.0 kernel.
        arr = np.asarray(psfArray)
        expected_g = g_inner.computeKernelImage(Point2D(1, 1)).array
        expected_r = r_at_30.computeKernelImage(Point2D(30, 30)).array
        self.assertAlmostEqual(arr[0].max(), expected_g.max(), places=4)
        self.assertAlmostEqual(arr[1].max(), expected_r.max(), places=4)

    def test_computeNearestPsfMultiBand_upgrades_all_bands_to_common(self):
        """When the fallback location found for a failing band is also
        valid for the bands that succeeded at the center, every band is
        re-sampled at that common location — including bands that
        already had a PSF at the center. The previously-successful
        center PSFs are discarded.

        Band g is invalid at the requested center ``(25, 25)`` but valid
        at ``(40, 40)`` with σ=1.0. Band r is valid at *both* ``(25, 25)``
        with σ=1.2 *and* ``(40, 40)`` with σ=1.5. Because the common
        fallback ``(40, 40)`` is also valid for r, the upgrade fires and
        r's returned kernel is the σ=1.5 one (re-sampled at (40, 40)),
        not the σ=1.2 kernel r had at the center.
        """
        bands = ("g", "r")
        g_at_40 = GaussianPsf(41, 41, 1.0)
        r_at_center = GaussianPsf(41, 41, 1.2)
        r_at_40 = GaussianPsf(41, 41, 1.5)
        psfs = [
            BadPsf(Point2D(40, 40), g_at_40),
            MultiPointBadPsf({(25, 25): r_at_center, (40, 40): r_at_40}),
        ]
        mCoadd = self._generateMultibandCoadd(psfs, bands)
        catalog = self._generateCatalog(bands, [[(40, 40, 10)]])

        psfArray, newCoadd = mes.utils.computeNearestPsfMultiBand(
            mCoadd, Point2D(25, 25), catalog
        )

        self.assertTupleEqual(newCoadd.bands, bands)
        # Both bands re-sampled at (40, 40): g matches σ=1.0, and r
        # matches σ=1.5 (the (40, 40) kernel), not σ=1.2 (the kernel r
        # held at the center). Failing this assertion would mean either
        # the upgrade did not run or r kept its center PSF.
        arr = np.asarray(psfArray)
        expected_g = g_at_40.computeKernelImage(Point2D(40, 40)).array
        expected_r = r_at_40.computeKernelImage(Point2D(40, 40)).array
        self.assertAlmostEqual(arr[0].max(), expected_g.max(), places=4)
        self.assertAlmostEqual(arr[1].max(), expected_r.max(), places=4)

    def test_computeNearestPsfMultiBand_falls_back_at_two_locations(self):
        """When a band needs to fall back but the fallback location is
        invalid for a band that succeeded at the center, the successful
        band keeps its center PSF — it is not dropped.

        Requested center is ``(25, 25)``. Band g's PSF is valid only at
        ``(40, 40)``; band r's PSF is valid only at ``(25, 25)``. The
        catalog has a single peak at ``(40, 40)``, so g falls back there.
        ``(40, 40)`` is invalid for r, so r stays at the center — both
        bands are kept. Under the bug, band g's fallback shifted r's
        search center to ``(40, 40)``; r's direct compute at
        ``(40, 40)`` then failed and the only catalog peak also failed
        for r, so r was silently dropped from the multiband PSF.
        """
        bands = ("g", "r")
        g_inner = GaussianPsf(41, 41, 1.0)
        r_inner = GaussianPsf(41, 41, 1.2)
        psfs = [
            BadPsf(Point2D(40, 40), g_inner),
            BadPsf(Point2D(25, 25), r_inner),
        ]
        mCoadd = self._generateMultibandCoadd(psfs, bands)
        catalog = self._generateCatalog(bands, [[(40, 40, 10)]])

        psfArray, newCoadd = mes.utils.computeNearestPsfMultiBand(
            mCoadd, Point2D(25, 25), catalog
        )

        self.assertTupleEqual(newCoadd.bands, bands)
        # g lands at (40, 40) (σ=1.0); r stays at the requested center
        # (σ=1.2). The (40, 40) fallback is *not* valid for r, so the
        # upgrade-to-common path is correctly skipped.
        arr = np.asarray(psfArray)
        expected_g = g_inner.computeKernelImage(Point2D(40, 40)).array
        expected_r = r_inner.computeKernelImage(Point2D(25, 25)).array
        self.assertAlmostEqual(arr[0].max(), expected_g.max(), places=4)
        self.assertAlmostEqual(arr[1].max(), expected_r.max(), places=4)

    def test_computePsfKernelImage_catalog_emits_future_warning(self):
        """Passing the deprecated ``catalog`` argument to
        ``computePsfKernelImage`` emits a ``FutureWarning``.

        Per finding U-7 of the ``audits/audit-2026-05-05.md`` audit,
        ``catalog`` is a dead argument -- the body never references it.
        Rather than remove the parameter (which would silently break
        any external caller passing it positionally or by keyword) the
        fix deprecates it and steers callers toward
        ``computeNearestPsfMultiBand`` for nearest-PSF fallback. The
        warning lets users find and remove the dead-arg call site
        before the parameter is dropped after v31.

        The test pins the warning channel (``FutureWarning``) and
        confirms the function still returns a valid result -- the
        ``catalog`` argument is ignored, so the output is identical to
        the ``catalog=None`` path.
        """
        bands = tuple("gri")
        psfs, psfImage = self._generateMultibandPsf([1.0, 1.2, 1.4])
        mCoadd = self._generateMultibandCoadd(psfs, bands)
        catalog = self._generateCatalog(bands)

        with self.assertWarns(FutureWarning):
            psfArray, newCoadd = mes.utils.computePsfKernelImage(
                mCoadd, Point2D(25, 25), catalog=catalog,
            )

        # The catalog argument is ignored, so the output matches the
        # catalog=None / catalog-absent path exactly.
        np.testing.assert_array_equal(psfArray, psfImage)
        self.assertTupleEqual(newCoadd.bands, bands)

    def test_buildObservation_no_divide_warning_on_zero_variance(self):
        """``buildObservation`` does not emit numpy ``RuntimeWarning``
        when the input variance plane contains zeros.

        Per finding U-5 of the ``audits/audit-2026-05-05.md`` audit,
        the inverse-variance weights were computed as
        ``weights = 1 / mExposure.variance.array`` without an
        ``errstate`` guard. Any zero pixel in the variance plane
        produced ``RuntimeWarning: divide by zero encountered in
        divide``, and any non-finite pixel produced
        ``RuntimeWarning: invalid value encountered in divide``. The
        warnings are spurious -- the immediately following
        ``weights[~np.isfinite(weights)] = 0`` line replaces every
        offending value with the intended sentinel -- but they pollute
        production logs and look like real numerical problems. The
        fix suppresses the spurious warnings via ``np.errstate``.

        The fixture coadd's variance plane defaults to all zeros,
        which under the bug fires the warning on every pixel; the
        modelPsf and a valid per-band PSF satisfy
        ``buildObservation``'s preconditions so the function runs all
        the way through and the test exercises both the divide site
        and the downstream weight-zeroing.
        """
        modelPsf = scl.utils.integrated_circular_gaussian(sigma=0.8).astype(np.float32)
        bands = tuple("gri")
        psfs, _ = self._generateMultibandPsf([1.0, 1.2, 1.4])
        mCoadd = self._generateMultibandCoadd(psfs, bands)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            observation = mes.utils.buildObservation(
                modelPsf, Point2I(25, 25), mCoadd
            )

        # Sanity check that the call actually went through the
        # divide-by-zero path: every weight should have been zeroed.
        np.testing.assert_array_equal(
            observation.weights, np.zeros_like(observation.weights)
        )

    def test_buildObservationBadPsfs(self):
        # Test that creating an observation with all bad PSFs
        # raises NoWorkFound
        modelPsf = scl.utils.integrated_circular_gaussian(sigma=0.8).astype(np.float32)
        bands = tuple("gri")
        psfs, psfImage = self._generateMultibandPsf([1.0, 1.2, 1.4])
        psfs = [BadPsf(Point2D(1, 1), psf) for psf in psfs]
        mCoadd = self._generateMultibandCoadd(psfs, bands)
        catalog = self._generateCatalog(self.bands)

        # Test that building the observation fails without a catalog
        with self.assertRaises(NoWorkFound):
            mes.utils.buildObservation(modelPsf, Point2I(25, 25), mCoadd)

        # Test that building the observation fails even with a catalog
        with self.assertRaises(NoWorkFound):
            mes.utils.buildObservation(modelPsf, Point2I(25, 25), mCoadd, catalog=catalog)

    def _generateGoodPsf(self, sigma: float = 1.0):
        # Generate a PSF and Image of the PSF
        psfRadius = 20
        psfShape = (2 * psfRadius + 1, 2 * psfRadius + 1)
        psf = GaussianPsf(psfShape[1], psfShape[0], sigma)
        psfImage = psf.computeImage(psf.getAveragePosition()).array
        return psf, psfImage

    def _generateMultibandPsf(self, sigmas: list[float]):
        # Generate a multiband PSF with a BadPsf for each None value in sigmas
        psfs = []
        psfImages = []
        for sigma in sigmas:
            psf, psfImage = self._generateGoodPsf(sigma)
            psfs.append(psf)
            psfImages.append(psfImage)
        return psfs, np.asarray(psfImages)

    def _generateCoadd(self, psf: Psf):
        # Create an empty exposure
        masked_image = afwImage.MaskedImage(Extent2I(50, 50), dtype=np.float32)
        coadd = afwImage.Exposure(masked_image, dtype=np.float32)
        coadd.setPsf(psf)
        return coadd

    def _generateMultibandCoadd(self, psfs: Psf, bands: list[str]):
        # Create an empty multi-band exposure
        coadds = []
        for psf in psfs:
            coadds.append(self._generateCoadd(psf))
        return afwImage.MultibandExposure.fromExposures(bands, coadds)

    def _generateCatalog(self, bands, footprints: list[list[tuple[int, int, int]]] | None = None):
        # Generate a catalog with a source for each footprint
        if footprints is None:
            footprints = []
        schema = SourceTable.makeMinimalSchema()
        peakSchema = PeakTable.makeMinimalSchema()
        for band in bands:
            schema.addField(f"merge_footprint_{band}", type="Flag")
            peakSchema.addField(f"merge_peak_{band}", type="Flag")

        table = SourceTable.make(schema)
        catalog = SourceCatalog(table)

        for peaks in footprints:
            src = catalog.addNew()
            footprint = Footprint(SpanSet(), peakSchema)
            for peak in peaks:
                footprint.addPeak(*peak)
            src.setFootprint(footprint)

            for band in bands:
                src[f"merge_footprint_{band}"] = True
                footprint.peaks[f"merge_peak_{band}"] = True
        return catalog


class TestNonzeroBandSupport(lsst.utils.tests.TestCase):
    """Tests for ``nonzeroBandSupport`` in
    ``lsst.meas.extensions.scarlet.utils``.

    The helper consolidates three previously inconsistent idioms for
    "this pixel is in the source's support across bands" (``> 0``,
    ``np.max != 0``, ``np.any != 0``) into a single canonical
    ``np.any(data != 0, axis=0)``. The discriminator between the
    canonical form and the historical idioms is a pixel whose band
    values are all zero except for a negative entry, or a mix of
    negative and zero (which ``np.max != 0`` excludes when the
    largest value is exactly zero). Regression coverage for
    finding U-2 of the ``audits/audit-2026-05-05.md`` audit.
    """

    def test_nonzeroBandSupport_includes_negative_only_pixels(self):
        """A pixel that is negative in some bands and zero in others
        counts as in the support.

        Layout of the 2-band ``(2, 3, 3)`` input:

        - ``(0, 0)``: ``[+1, 0]`` — positive, in support.
        - ``(0, 1)``: ``[-1, 0]`` — negative-and-zero mix (``max == 0``
          excludes this; the canonical helper includes it).
        - ``(0, 2)``: ``[0, 0]`` — all-zero, not in support.
        - ``(1, 0)``: ``[-1, -1]`` — uniformly negative; ``> 0``
          excludes, the canonical helper includes.
        - All other pixels zero.
        """
        data = np.zeros((2, 3, 3), dtype=np.float32)
        data[0, 0, 0] = 1.0
        data[0, 0, 1] = -1.0
        data[:, 1, 0] = -1.0

        result = mes.utils.nonzeroBandSupport(data)

        expected = np.array(
            [
                [True, True, False],
                [True, False, False],
                [False, False, False],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_nonzeroBandSupport_all_zero(self):
        """An all-zero cube returns an all-False support mask."""
        data = np.zeros((3, 4, 4), dtype=np.float32)
        result = mes.utils.nonzeroBandSupport(data)
        np.testing.assert_array_equal(result, np.zeros((4, 4), dtype=bool))

    def test_nonzeroBandSupport_all_positive(self):
        """A strictly-positive cube returns an all-True support mask."""
        data = np.ones((3, 2, 2), dtype=np.float32)
        result = mes.utils.nonzeroBandSupport(data)
        np.testing.assert_array_equal(result, np.ones((2, 2), dtype=bool))

    def test_nonzeroBandSupport_single_band(self):
        """A single-band cube reduces along the band axis cleanly.

        The historical ``np.max != 0`` form silently failed for a
        ``(1, h, w)`` slice whose only non-zero pixel was negative —
        the test pixel at ``(0, 1)`` distinguishes ``!= 0`` from
        ``> 0`` even with only one band.
        """
        data = np.array(
            [[[0.0, -1.0], [2.0, 0.0]]], dtype=np.float32
        )
        result = mes.utils.nonzeroBandSupport(data)
        np.testing.assert_array_equal(
            result, np.array([[False, True], [True, False]])
        )


class TestMultibandConvolve(lsst.utils.tests.TestCase):
    """Tests for ``multiband_convolve`` in
    ``lsst.meas.extensions.scarlet.utils``.

    ``multiband_convolve`` iterates over ``zip(images, psfs, strict=True)``
    and calls ``scipy.signal.convolve(..., mode="same")`` per band. Both
    arguments must be 3-D ``(bands, h, w)`` — the function does *not*
    broadcast a 2-D PSF across bands; the caller is responsible for
    that (see ``tests/utils.py::DeblenderTestModel.render``).
    """

    def test_multiband_convolve_per_band_psf(self):
        """Each band is convolved with its own PSF.

        Passes three distinct Gaussian PSFs (sigma = 0.8, 1.2, 1.6) and
        verifies that ``result[b]`` equals
        ``scipy.signal.convolve(images[b], psfs[b], mode="same")``
        computed independently for each band. A regression that
        cross-routed bands (e.g. always using ``psfs[0]``) would fail.
        """
        rng = np.random.RandomState(0)
        images = rng.rand(3, 21, 21).astype(np.float32)
        psfs = np.stack([
            scl.utils.integrated_circular_gaussian(sigma=s).astype(np.float32)
            for s in (0.8, 1.2, 1.6)
        ])

        result = mes.utils.multiband_convolve(images, psfs)

        self.assertEqual(result.shape, images.shape)
        for b in range(3):
            expected = scipy.signal.convolve(images[b], psfs[b], mode="same")
            np.testing.assert_allclose(result[b], expected, atol=1e-6)

    def test_multiband_convolve_identity_psf(self):
        """A centered delta PSF returns the input unchanged.

        Pins the ``mode="same"`` contract: with a 3×3 PSF that is zero
        everywhere except a 1 at the center, the per-band convolution
        is an identity transformation. Any shape or centering bug in
        the wrapper would shift or truncate the output.
        """
        rng = np.random.RandomState(1)
        images = rng.rand(3, 11, 11).astype(np.float32)
        psfs = np.zeros((3, 3, 3), dtype=np.float32)
        psfs[:, 1, 1] = 1.0

        result = mes.utils.multiband_convolve(images, psfs)

        np.testing.assert_allclose(result, images, atol=1e-6)

    def test_multiband_convolve_shape_mismatch_raises(self):
        """Mismatched band counts raise ``ValueError``.

        Pins the ``zip(images, psfs, strict=True)`` contract; a
        regression that drops ``strict=True`` would silently broadcast
        or truncate.
        """
        images = np.zeros((3, 11, 11), dtype=np.float32)
        psfs = np.zeros((2, 5, 5), dtype=np.float32)

        with self.assertRaises(ValueError):
            mes.utils.multiband_convolve(images, psfs)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
