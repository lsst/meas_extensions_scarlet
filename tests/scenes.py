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

"""Named, atomic source layouts for the deblender test suite.

Each :class:`Scene` carries a list of :class:`DeblenderTestModel`
instances plus the band ordering it is to be rendered in. Scenes
hold no rendered image data; rendering is a job for
``pipeline.build_image``, which caches its output keyed by scene
name.

The ``"multi-blend"`` scene is the union of the four per-blend
scenes and reproduces the historic 8-source / 4-blend layout that
``test_deblend.py`` has used since the package was first written.
"""

from dataclasses import dataclass

import numpy as np

from utils import DeblenderTestModel, PsfModel, SersicModel


# Band ordering shared by every scene (`tuple` [`str`]).
BANDS: tuple[str, ...] = tuple("gri")


@dataclass(frozen=True)
class Scene:
    """Named layout of synthetic sources for one deblender test case.

    A scene is pure metadata: it names the layout, lists the bands it is
    to be rendered in, and carries the source models. Turning it into
    pixel data is the job of ``pipeline.build_image``.

    Parameters
    ----------
    name : `str`
        Identifier used as the rendering cache key and as the test ID
        in parametrized tests.
    bands : `tuple` [`str`]
        Band ordering for the rendered image; the model spectra are
        indexed in this order.
    description : `str`
        Short human-readable summary of the layout, surfaced in test
        failure messages.
    models : `list` [`DeblenderTestModel`]
        Source models that make up the scene.
    """

    name: str
    bands: tuple[str, ...]
    description: str
    models: list[DeblenderTestModel]


def _one_isolated_psf_models() -> list[DeblenderTestModel]:
    """Build the models for the ``one_isolated_psf`` scene."""
    return [
        PsfModel(center=(30, 15), spectrum=np.array([8, 2, 1]), bands=BANDS),
    ]


def _psf_plus_sersic_blend_models() -> list[DeblenderTestModel]:
    """Build the models for the ``psf_plus_sersic_blend`` scene."""
    return [
        SersicModel(
            center=(40, 20), major=5, minor=2, radius=15, theta=-np.pi / 4,
            n=1, spectrum=np.array([2, 4, 8]), bands=BANDS,
        ),
        PsfModel(center=(12, 20), spectrum=np.array([1, 2, 8]), bands=BANDS),
    ]


def _three_source_blend_models() -> list[DeblenderTestModel]:
    """Build the models for the ``three_source_blend`` scene."""
    return [
        SersicModel(
            center=(25, 70), major=5, minor=2, radius=20, theta=np.pi / 48,
            n=1, spectrum=np.array([2, 5, 8]), bands=BANDS,
        ),
        PsfModel(center=(32, 60), spectrum=np.array([1, 2, 8]), bands=BANDS),
        PsfModel(center=(16, 80), spectrum=np.array([8, 2, 1]), bands=BANDS),
    ]


def _large_two_sersic_models() -> list[DeblenderTestModel]:
    """Build the models for the ``large_two_sersic`` scene."""
    return [
        SersicModel(
            center=(70, 70), major=5, minor=2, radius=25, theta=0,
            n=1, spectrum=np.array([2, 10, 18]), bands=BANDS,
        ),
        SersicModel(
            center=(85, 85), major=5, minor=2, radius=25, theta=np.pi / 2,
            n=1, spectrum=np.array([5, 10, 20]), bands=BANDS,
        ),
    ]


# Registry of every available scene, keyed by scene name
SCENES: dict[str, Scene] = {
    "one_isolated_psf": Scene(
        name="one_isolated_psf",
        bands=BANDS,
        description="One isolated PSF source, away from any neighbor.",
        models=_one_isolated_psf_models(),
    ),
    "psf_plus_sersic_blend": Scene(
        name="psf_plus_sersic_blend",
        bands=BANDS,
        description="One Sersic with a nearby PSF overlapping its wing.",
        models=_psf_plus_sersic_blend_models(),
    ),
    "three_source_blend": Scene(
        name="three_source_blend",
        bands=BANDS,
        description="One Sersic plus two PSFs in a single blend.",
        models=_three_source_blend_models(),
    ),
    "large_two_sersic": Scene(
        name="large_two_sersic",
        bands=BANDS,
        description="Two large overlapping Sersics.",
        models=_large_two_sersic_models(),
    ),
    "multi-blend": Scene(
        name="multi-blend",
        bands=BANDS,
        description=(
            "Composite scene: one isolated PSF, one PSF+Sersic blend, "
            "one three-source blend, and one two-Sersic blend. "
            "Reproduces the historic 8-source / 4-blend layout from "
            "test_deblend.py."
        ),
        models=(
            _one_isolated_psf_models()
            + _psf_plus_sersic_blend_models()
            + _three_source_blend_models()
            + _large_two_sersic_models()
        ),
    ),
}
