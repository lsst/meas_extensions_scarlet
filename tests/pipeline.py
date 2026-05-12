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

"""Staged pipeline helpers for the deblender test suite.

The four stages — :func:`build_image`, :func:`detect`,
:func:`deconvolve`, :func:`deblend` — are pure functions over
:class:`Scene` and task-config inputs. Each stage memoizes its
result keyed by the scene name and the configs that have flowed
through, so asking for the same (scene, config) tuple in a later
test method returns the same bundle without re-running the work.

Tests that want to *mutate* a bundle's contents (e.g. by calling
``updateCatalogFootprints`` against a catalog) must copy the
piece they intend to mutate first; the bundles themselves are
frozen but the LSST objects they reference are not.
"""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

import lsst.afw.image as afwImage
import lsst.scarlet.lite as scl
from lsst.afw.detection import GaussianPsf
from lsst.afw.table import Schema, SchemaMapper, SourceCatalog, SourceTable
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.extensions.scarlet.deconvolveExposureTask import DeconvolveExposureTask
from lsst.meas.extensions.scarlet.scarletDeblendTask import ScarletDeblendTask
from lsst.pipe.base import Struct

from scenes import Scene
from utils import initData


# Per-band Gaussian PSF sigmas, in pixels, indexed in scene-band order.
_PSF_SIGMAS: tuple[float, ...] = (1.0, 1.2, 1.4)
# Half-size of the square PSF kernel in pixels (kernel is 2*r+1 on a side).
_PSF_RADIUS: int = 20
# Sigma of the narrow model PSF that scenes are rendered into, in pixels.
_MODEL_PSF_SIGMA: float = 0.8
# Peak-to-peak amplitude of the uniform noise added to convolved images.
_NOISE_AMPLITUDE: float = 0.05
# RNG seed for the noise draw; fixed so the cache is reproducible.
_NOISE_SEED: int = 0


@dataclass(frozen=True)
class ImageBundle:
    """Rendered image stage of the pipeline for one scene.

    Holds both the raw model and the observed (PSF-convolved, noisy)
    image of a single scene, plus the PSF objects needed by downstream
    stages.

    Parameters
    ----------
    scene : `Scene`
        Scene that produced this bundle.
    deconvolved : `lsst.scarlet.lite.Image`
        Model image convolved with the narrow model PSF only (the
        "truth" image scarlet is asked to recover).
    convolved : `lsst.scarlet.lite.Image`
        Model image convolved with the full per-band image PSFs;
        noise-free.
    noise : `numpy.ndarray`
        Per-pixel noise realization added to ``convolved`` to produce
        ``noisy_image``.
    noisy_image : `lsst.scarlet.lite.Image`
        ``convolved`` + ``noise``; the observation tests run against.
    mCoadd : `lsst.afw.image.MultibandExposure`
        ``noisy_image`` wrapped as the multiband exposure consumed by
        detection and deblending.
    psfs : `tuple` [`lsst.afw.detection.GaussianPsf`]
        Per-band image-space PSF objects attached to ``mCoadd``.
    modelPsf : `numpy.ndarray`
        Pixelized narrow model PSF used to render ``deconvolved``.
    imagePsf : `numpy.ndarray`
        Pixelized per-band image PSFs used to render ``convolved``.
    """

    scene: Scene
    deconvolved: scl.Image
    convolved: scl.Image
    noise: np.ndarray
    noisy_image: scl.Image
    mCoadd: afwImage.MultibandExposure
    psfs: tuple[GaussianPsf, ...]
    modelPsf: np.ndarray
    imagePsf: np.ndarray

    @property
    def bands(self) -> tuple[str, ...]:
        """Band ordering for this bundle (`tuple` [`str`], read-only)."""
        return self.scene.bands


@dataclass(frozen=True)
class DetectionBundle:
    """Detection stage of the pipeline for one (scene, config) pair.

    Parameters
    ----------
    image : `ImageBundle`
        Upstream rendered-image bundle that was detected on.
    catalog : `lsst.afw.table.SourceCatalog`
        Catalog of detected sources in the output (mapped) schema.
    schema : `lsst.afw.table.Schema`
        Output schema, after the schema mapper has been applied.
    inputSchema : `lsst.afw.table.Schema`
        Minimal input schema that the detection task was constructed
        with.
    schemaMapper : `lsst.afw.table.SchemaMapper`
        Mapper from ``inputSchema`` to ``schema``.
    detectionTask : `lsst.meas.algorithms.SourceDetectionTask`
        The detection task instance, kept for inspection in tests.
    detection_config_key : `str`
        Serialized form of the detection config used as the cache key.
    """

    image: ImageBundle
    catalog: SourceCatalog
    schema: Schema
    inputSchema: Schema
    schemaMapper: SchemaMapper
    detectionTask: SourceDetectionTask
    detection_config_key: str


@dataclass(frozen=True)
class DeconvolveBundle:
    """Deconvolution stage of the pipeline for one config tuple.

    Parameters
    ----------
    detection : `DetectionBundle`
        Upstream detection bundle that was deconvolved.
    mDeconvolved : `lsst.afw.image.MultibandExposure`
        Per-band deconvolved exposures produced by
        `DeconvolveExposureTask`.
    deconvolveTask : \
            `lsst.meas.extensions.scarlet.DeconvolveExposureTask`
        The deconvolve task instance, kept for inspection in tests.
    deconvolve_config_key : `str`
        Serialized form of the deconvolve config used as the cache key.
    """

    detection: DetectionBundle
    mDeconvolved: afwImage.MultibandExposure
    deconvolveTask: DeconvolveExposureTask
    deconvolve_config_key: str

    @property
    def image(self) -> ImageBundle:
        """Rendered-image bundle this stage was built from \
(`ImageBundle`, read-only)."""
        return self.detection.image


@dataclass(frozen=True)
class DeblendBundle:
    """Deblend stage of the pipeline for one config tuple.

    Parameters
    ----------
    deconvolved : `DeconvolveBundle`
        Upstream deconvolved bundle that was deblended.
    result : `lsst.pipe.base.Struct`
        Raw result struct returned by ``ScarletDeblendTask.run``.
    deblendTask : \
            `lsst.meas.extensions.scarlet.ScarletDeblendTask`
        The deblend task instance, kept for inspection in tests.
    deblend_config_key : `str`
        Serialized form of the deblend config used as the cache key.
    """

    deconvolved: DeconvolveBundle
    result: Struct
    deblendTask: ScarletDeblendTask
    deblend_config_key: str

    @property
    def detection(self) -> DetectionBundle:
        """Detection bundle this stage was built from \
(`DetectionBundle`, read-only)."""
        return self.deconvolved.detection

    @property
    def image(self) -> ImageBundle:
        """Rendered-image bundle this stage was built from \
(`ImageBundle`, read-only)."""
        return self.detection.image


def _config_key(config: Any) -> str:
    """Serialize a task config to a stable cache key.

    Parameters
    ----------
    config : `lsst.pex.config.Config` or `None`
        Task config to key, or `None` to mean "use the task's default
        config".

    Returns
    -------
    key : `str`
        ``"<default>"`` when ``config`` is `None`; otherwise a JSON
        dump of ``config.toDict()`` with sorted keys.
    """
    if config is None:
        return "<default>"
    return json.dumps(config.toDict(), sort_keys=True, default=str)


def _build_psfs() -> tuple[tuple[GaussianPsf, ...], np.ndarray, np.ndarray]:
    """Build the per-band image PSFs and the narrow model PSF.

    Returns
    -------
    psfs : `tuple` [`lsst.afw.detection.GaussianPsf`]
        One `GaussianPsf` per entry in ``_PSF_SIGMAS``.
    modelPsf : `numpy.ndarray`
        Pixelized narrow model PSF, normalized.
    imagePsf : `numpy.ndarray`
        Pixelized per-band image PSFs, each normalized to unit sum.
    """
    modelPsf = scl.utils.integrated_circular_gaussian(
        sigma=_MODEL_PSF_SIGMA
    ).astype(np.float32)
    psfShape = (2 * _PSF_RADIUS + 1, 2 * _PSF_RADIUS + 1)
    psfs = tuple(
        GaussianPsf(psfShape[1], psfShape[0], sigma) for sigma in _PSF_SIGMAS
    )
    imagePsf = np.asarray(
        [psf.computeImage(psf.getAveragePosition()).array for psf in psfs]
    ).astype(np.float32)
    imagePsf /= imagePsf.sum(axis=(1, 2))[:, None, None]
    return psfs, modelPsf, imagePsf


# PSF objects built once at import time and shared across every scene.
_PSFS, _MODEL_PSF, _IMAGE_PSF = _build_psfs()


# Memoization caches, one per stage. Keys are tuples of all upstream
# config keys plus this stage's own, so reusing a scene + config tuple
# returns the same bundle without re-running the work.
_image_cache: dict[str, ImageBundle] = {}
_detection_cache: dict[tuple[str, str], DetectionBundle] = {}
_deconvolve_cache: dict[tuple[str, str, str], DeconvolveBundle] = {}
_deblend_cache: dict[tuple[str, str, str, str], DeblendBundle] = {}


def build_image(scene: Scene) -> ImageBundle:
    """Render the deconvolved, convolved, and noisy multiband image.

    Parameters
    ----------
    scene : `Scene`
        Scene whose models will be rendered.

    Returns
    -------
    bundle : `ImageBundle`
        Bundle holding both the truth (``deconvolved``) and the
        observation (``noisy_image`` / ``mCoadd``), keyed on
        ``scene.name``.
    """
    if scene.name in _image_cache:
        return _image_cache[scene.name]

    deconvolved, convolved = initData(scene.models, _MODEL_PSF, _IMAGE_PSF)

    rng = np.random.RandomState(_NOISE_SEED)
    noise = _NOISE_AMPLITUDE * (
        rng.rand(*convolved.shape).astype(np.float32) - 0.5
    )
    noisy_image = convolved.copy()
    noisy_image._data += noise

    masked = afwImage.MultibandMaskedImage.fromArrays(
        scene.bands, noisy_image.data, None, noise ** 2
    )
    coadds = [
        afwImage.Exposure(img, dtype=img.image.array.dtype) for img in masked
    ]
    mCoadd = afwImage.MultibandExposure.fromExposures(scene.bands, coadds)
    for b, coadd in enumerate(mCoadd):
        coadd.setPsf(_PSFS[b])

    bundle = ImageBundle(
        scene=scene,
        deconvolved=deconvolved,
        convolved=convolved,
        noise=noise,
        noisy_image=noisy_image,
        mCoadd=mCoadd,
        psfs=_PSFS,
        modelPsf=_MODEL_PSF,
        imagePsf=_IMAGE_PSF,
    )
    _image_cache[scene.name] = bundle
    return bundle


def detect(image: ImageBundle, config: Any = None) -> DetectionBundle:
    """Run :class:`SourceDetectionTask` on the r-band coadd.

    Parameters
    ----------
    image : `ImageBundle`
        Rendered-image bundle to detect on.
    config : `lsst.meas.algorithms.SourceDetectionConfig`, optional
        Detection-task config; the task's default is used when `None`.

    Returns
    -------
    bundle : `DetectionBundle`
        Bundle wrapping the catalog, schemas, and the task instance,
        keyed on ``(image.scene.name, config_key)``.
    """
    config_key = _config_key(config)
    key = (image.scene.name, config_key)
    if key in _detection_cache:
        return _detection_cache[key]

    inputSchema = SourceTable.makeMinimalSchema()
    table = SourceTable.make(inputSchema)
    detectionTask = SourceDetectionTask(schema=inputSchema, config=config)
    schemaMapper = SchemaMapper(inputSchema)
    schemaMapper.addMinimalSchema(inputSchema)
    schema = schemaMapper.getOutputSchema()

    detectionResult = detectionTask.run(table, image.mCoadd["r"])
    catalog_table = SourceCatalog.Table.make(schema)
    catalog = SourceCatalog(catalog_table)
    catalog.extend(detectionResult.sources, schemaMapper)

    bundle = DetectionBundle(
        image=image,
        catalog=catalog,
        schema=schema,
        inputSchema=inputSchema,
        schemaMapper=schemaMapper,
        detectionTask=detectionTask,
        detection_config_key=config_key,
    )
    _detection_cache[key] = bundle
    return bundle


def deconvolve(
    detection: DetectionBundle, config: Any = None
) -> DeconvolveBundle:
    """Run :class:`DeconvolveExposureTask` once per band.

    Parameters
    ----------
    detection : `DetectionBundle`
        Detection-stage bundle whose coadds are to be deconvolved.
    config : \
            `lsst.meas.extensions.scarlet.DeconvolveExposureConfig`, \
            optional
        Deconvolve-task config; the task's default is used when `None`.

    Returns
    -------
    bundle : `DeconvolveBundle`
        Bundle wrapping the per-band deconvolved exposures and the
        task instance, keyed on the upstream config keys plus this
        stage's config key.
    """
    config_key = _config_key(config)
    key = (
        detection.image.scene.name,
        detection.detection_config_key,
        config_key,
    )
    if key in _deconvolve_cache:
        return _deconvolve_cache[key]

    deconvolveTask = DeconvolveExposureTask(config=config)
    catalog = detection.catalog if deconvolveTask.config.useFootprints else None

    deconvolvedCoadds = []
    for coadd in detection.image.mCoadd:
        deconvolvedCoadd = deconvolveTask.run(coadd, catalog).deconvolved
        deconvolvedCoadds.append(deconvolvedCoadd)
    mDeconvolved = afwImage.MultibandExposure.fromExposures(
        detection.image.bands, deconvolvedCoadds
    )

    bundle = DeconvolveBundle(
        detection=detection,
        mDeconvolved=mDeconvolved,
        deconvolveTask=deconvolveTask,
        deconvolve_config_key=config_key,
    )
    _deconvolve_cache[key] = bundle
    return bundle


def deblend(
    deconvolved: DeconvolveBundle, config: Any = None
) -> DeblendBundle:
    """Run :class:`ScarletDeblendTask`.

    Parameters
    ----------
    deconvolved : `DeconvolveBundle`
        Deconvolve-stage bundle whose coadds and catalog are
        deblended.
    config : \
            `lsst.meas.extensions.scarlet.ScarletDeblendConfig`, \
            optional
        Deblend-task config; the task's default is used when `None`.

    Returns
    -------
    bundle : `DeblendBundle`
        Bundle wrapping the deblend result struct and the task
        instance, keyed on the upstream config keys plus this stage's
        config key.
    """
    config_key = _config_key(config)
    key = (
        deconvolved.image.scene.name,
        deconvolved.detection.detection_config_key,
        deconvolved.deconvolve_config_key,
        config_key,
    )
    if key in _deblend_cache:
        return _deblend_cache[key]

    deblendTask = ScarletDeblendTask(
        schema=deconvolved.detection.schema, config=config
    )
    result = deblendTask.run(
        deconvolved.image.mCoadd,
        deconvolved.mDeconvolved,
        deconvolved.detection.catalog,
    )
    bundle = DeblendBundle(
        deconvolved=deconvolved,
        result=result,
        deblendTask=deblendTask,
        deblend_config_key=config_key,
    )
    _deblend_cache[key] = bundle
    return bundle
