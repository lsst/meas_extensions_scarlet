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

import lsst.afw.image as afwImage
import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np
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
        derivedPsf, center, dist = mes.utils.computeNearestPsf(coadd, catalog, None, Point2D(4, 5))
        np.testing.assert_array_equal(derivedPsf.array, psfImage)
        self.assertEqual(center, Point2I(1, 1))
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


if __name__ == "__main__":
    unittest.main()
