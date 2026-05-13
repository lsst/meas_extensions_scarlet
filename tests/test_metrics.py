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

"""Tests for ``setDeblenderMetrics``."""

import unittest

import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite as scl
import lsst.utils.tests
import numpy as np


# Three-band setup shared by every test. setDeblenderMetrics only reads
# source models and the blend's bounding box, so arbitrary band names
# and dummy (zero-image / unit-variance) observations are sufficient.
BANDS = ("g", "r", "i")
IMAGE_SHAPE = (20, 20)


def _build_blend(source_specs):
    """Build a minimal scarlet Blend from synthetic-source specs.

    Parameters
    ----------
    source_specs : `list` [`tuple`]
        One tuple per source: ``(morph, origin, peak, spectrum)``.
        ``morph`` is a 2D ``numpy.ndarray``; ``origin`` is the
        ``(y, x)`` bottom-left corner of the morph in image
        coordinates; ``peak`` is the ``(y, x)`` of the source's
        peak; ``spectrum`` is the per-band amplitude.
    """
    n_bands = len(BANDS)
    obs_shape = (n_bands,) + IMAGE_SHAPE
    psfs = np.eye(5, dtype=np.float32)[None].repeat(n_bands, axis=0)
    observation = scl.Observation(
        images=np.zeros(obs_shape, dtype=np.float32),
        variance=np.ones(obs_shape, dtype=np.float32),
        weights=np.ones(obs_shape, dtype=np.float32),
        psfs=psfs,
        bands=BANDS,
    )
    sources = []
    for morph, origin, peak, spectrum in source_specs:
        model = (
            np.asarray(morph, dtype=np.float32)[None, :, :]
            * np.asarray(spectrum, dtype=np.float32)[:, None, None]
        )
        image = scl.Image(model, yx0=origin, bands=BANDS)
        component = scl.component.CubeComponent(model=image, peak=peak)
        sources.append(scl.Source([component]))
    return scl.Blend(sources=sources, observation=observation)


class TestSetDeblenderMetrics(lsst.utils.tests.TestCase):
    """Tests for ``setDeblenderMetrics`` in
    ``lsst.meas.extensions.scarlet.metrics``.

    The function operates on an ``scl.Blend`` and assigns a
    ``DeblenderMetrics`` instance to each source's ``metrics`` attribute.
    The synthetic blends below use uniform-flux rectangular morphs so
    every metric has a closed-form expected value.
    """

    def test_setDeblenderMetrics_isolated(self):
        """A single-source blend has all four metrics equal to zero in
        every band.

        With no neighbor model, ``neighborOverlap`` is identically zero
        so ``maxOverlap``, ``fluxOverlap``, and ``fluxOverlapFraction``
        are zero. ``blendedness`` is ``1 - sum(m²) / sum(M·m)``; with
        ``M ≡ m`` over the source's support the ratio is 1 and
        blendedness is 0.
        """
        morph = np.ones((5, 5), dtype=np.float32)
        blend = _build_blend([(morph, (5, 5), (7, 7), [1.0, 1.0, 1.0])])

        mes.metrics.setDeblenderMetrics(blend)

        zeros = np.zeros(len(BANDS), dtype=np.float64)
        metrics = blend.sources[0].metrics
        np.testing.assert_array_equal(metrics.maxOverlap, zeros)
        np.testing.assert_array_equal(metrics.fluxOverlap, zeros)
        np.testing.assert_array_equal(metrics.fluxOverlapFraction, zeros)
        np.testing.assert_array_equal(metrics.blendedness, zeros)

    def test_setDeblenderMetrics_two_disjoint(self):
        """Two non-overlapping sources both have zero overlap metrics.

        The morphs occupy disjoint bboxes so each source's footprint
        contains no neighbor flux; ``neighborOverlap`` is zero, and the
        blendedness collapses to the isolated-source case for each
        source independently.
        """
        morph = np.ones((3, 3), dtype=np.float32)
        # First morph: rows 2-4, cols 2-4. Second: rows 12-14, cols 12-14.
        # Clear gap of ≥ 7 pixels in each axis.
        blend = _build_blend([
            (morph, (2, 2), (3, 3), [1.0, 1.0, 1.0]),
            (morph, (12, 12), (13, 13), [1.0, 1.0, 1.0]),
        ])

        mes.metrics.setDeblenderMetrics(blend)

        zeros = np.zeros(len(BANDS), dtype=np.float64)
        for src in blend.sources:
            np.testing.assert_array_equal(src.metrics.maxOverlap, zeros)
            np.testing.assert_array_equal(src.metrics.fluxOverlap, zeros)
            np.testing.assert_array_equal(
                src.metrics.fluxOverlapFraction, zeros
            )
            np.testing.assert_array_equal(src.metrics.blendedness, zeros)

    def test_setDeblenderMetrics_overlapping_sources(self):
        """Two uniform 5×5 sources that overlap in a 3×3 region produce
        analytically derivable overlap metrics.

        Setup
        -----
        - Each source: 5×5 morph filled with 1.0, flat spectrum
          ``(1, 1, 1)``.
        - Source A at origin ``(5, 5)`` covers rows 5-9, cols 5-9.
        - Source B at origin ``(7, 7)`` covers rows 7-11, cols 7-11.
        - Overlap region: rows 7-9, cols 7-9 — nine pixels.

        Expected values per band, identical for both sources by
        symmetry:

        - ``maxOverlap = 1.0`` — the neighbor's contribution at any
          overlap pixel.
        - ``fluxOverlap = 9.0`` — sum of neighbor flux over the nine
          overlap pixels.
        - ``fluxOverlapFraction = 9 / 25 = 0.36`` — overlap flux over
          this source's total flux of 25.
        - ``blendedness = 1 - sum(m²) / sum(M·m)``.
          Non-overlap pixels (16) contribute ``m² = 1, M·m = 1``;
          overlap pixels (9) contribute ``m² = 1, M·m = 2``.
          ``sum(m²) = 25``, ``sum(M·m) = 16 + 18 = 34``.
          ``blendedness = 1 - 25/34 = 9/34 ≈ 0.2647``.
        """
        morph = np.ones((5, 5), dtype=np.float32)
        blend = _build_blend([
            (morph, (5, 5), (7, 7), [1.0, 1.0, 1.0]),
            (morph, (7, 7), (9, 9), [1.0, 1.0, 1.0]),
        ])

        mes.metrics.setDeblenderMetrics(blend)

        n_bands = len(BANDS)
        for src in blend.sources:
            np.testing.assert_allclose(
                src.metrics.maxOverlap, [1.0] * n_bands
            )
            np.testing.assert_allclose(
                src.metrics.fluxOverlap, [9.0] * n_bands
            )
            np.testing.assert_allclose(
                src.metrics.fluxOverlapFraction,
                [9.0 / 25.0] * n_bands,
            )
            np.testing.assert_allclose(
                src.metrics.blendedness,
                [9.0 / 34.0] * n_bands,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
