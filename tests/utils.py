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

from typing import Sequence

import lsst.scarlet.lite as scl
import numpy as np
import lsst.meas.extensions.scarlet as mes


class DeblenderTestModel:
    center: tuple[float, float]
    spectrum: np.ndarray
    morph: np.ndarray
    bbox: scl.Box
    bands: Sequence[str]

    def render(self, psf: np.ndarray) -> scl.Image:
        model = self.spectrum[:, None, None]*self.morph[None, :, :]
        if len(psf.shape) == 2:
            psf = np.repeat(psf[None, :, :], model.shape[0], axis=0)
        model = mes.utils.multiband_convolve(model, psf)
        return scl.Image(model, bands=self.bands, yx0=self.bbox.origin)


class SersicModel(DeblenderTestModel):
    def __init__(
        self,
        center: tuple[int, int],
        major: float,
        minor: float,
        radius: int,
        theta: float,
        n: float,
        spectrum: np.ndarray,
        bands: Sequence[str],
    ):
        self.center = center
        self.spectrum = spectrum.astype(np.float32)
        self.bands = bands
        bbox = scl.Box((2*radius+1, 2*radius+1), (center[0]-radius, center[1]-radius))
        self.bbox = bbox
        frame = scl.models.parametric.EllipseFrame(
            center[0],
            center[1],
            major,
            minor,
            theta,
            bbox,
        )
        morph = scl.models.parametric.sersic((n,), frame).astype(np.float32)
        self.morph = morph/np.max(morph)


class PsfModel(DeblenderTestModel):
    def __init__(
        self,
        center,
        spectrum: np.ndarray,
        bands: Sequence[str],
        radius: int = 7,
    ):
        self.center = center
        self.spectrum = spectrum.astype(np.float32)
        self.bands = bands
        bbox = scl.Box((2*radius+1, 2*radius+1), (center[0]-radius, center[1]-radius))
        self.bbox = bbox
        self.morph = np.zeros(bbox.shape, dtype=np.float32)
        _center = (self.morph.shape[0]-1)//2, (self.morph.shape[1]-1)//2
        self.morph[*_center] = 1


def initData(
    models: list[DeblenderTestModel],
    modelPsf: np.ndarray,
    imagePsf: np.ndarray,
) -> tuple[scl.Image, scl.Image]:
    # Find the bounding box that contains all of the models
    bbox = models[0].bbox
    bands = models[0].bands
    for model in models[1:]:
        bbox = bbox | model.bbox
        assert bands == model.bands
    bbox = bbox.grow(5)

    for dim in bbox.origin:
        if dim < 0:
            raise ValueError("Invalid setup, at least one source has a footprint below (0, 0)")

    deconvolved = scl.Image.from_box(bbox, bands=bands, dtype=modelPsf.dtype)
    convolved = scl.Image.from_box(bbox, bands=bands, dtype=imagePsf.dtype)
    for model in models:
        deconvolved += model.render(modelPsf)
        convolved += model.render(imagePsf)
    return deconvolved, convolved
