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

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
import logging
import json
from typing import Any, BinaryIO, cast
import zipfile

import numpy as np
from pydantic_core import from_json

import lsst.scarlet.lite as scl
from lsst.afw.detection import Footprint as afwFootprint
from lsst.afw.detection import HeavyFootprintF
from lsst.afw.geom import Span, SpanSet
from lsst.afw.image import Exposure, MaskedImage, MaskX, MultibandExposure
from lsst.afw.table import SourceCatalog, SourceRecord
import lsst.utils as lsst_utils
from lsst.daf.butler import StorageClassDelegate
from lsst.daf.butler import FormatterV2
from lsst.geom import Point2I, Extent2I, Box2I
from lsst.pipe.base import NoWorkFound
from lsst.resources import ResourceHandleProtocol
from lsst.scarlet.lite import (
    Box,
    FactorizedComponent,
    FixedParameter,
)

from .metrics import setDeblenderMetrics
from . import utils
from .footprint import scarletModelToHeavy

logger = logging.getLogger(__name__)


# The name of the band in an monochome blend.
# This is used as a placeholder since the band is not used in the
# monochromatic model.
monochromaticBand = "dummy"
monochromaticBands = (monochromaticBand,)


def monochromaticDataToScarlet(
    blendData: scl.io.ScarletBlendData,
    bandIndex: int,
    observation: scl.Observation,
):
    """Convert the storage data model into a scarlet lite blend

    Parameters
    ----------
    blendData:
        Persistable data for the entire blend.
    bandIndex:
        Index of model to extract.
    observation:
        Observation of region inside the bounding box.

    Returns
    -------
    blend : `scarlet.lite.LiteBlend`
        A scarlet blend model extracted from persisted data.
    """
    sources = []
    # Use a dummy band, since we are only extracting a monochromatic model
    # that will be turned into a HeavyFootprint.
    bands = monochromaticBands
    for sourceId, sourceData in blendData.sources.items():
        components: list[scl.Component] = []
        # There is no need to distinguish factorized components from regular
        # components, since there is only one band being used.
        for componentData in sourceData.components:
            if componentData.component_type == "component":
                bbox = Box(componentData.shape, origin=componentData.origin)
                model = scl.Image(
                    componentData.model[bandIndex][None, :, :], yx0=bbox.origin, bands=bands
                )
                component = scl.ComponentCube(
                    model=model,
                    peak=tuple(componentData.peak[::-1]),
                )
                components.append(component)
            else:
                bbox = Box(componentData.shape, origin=componentData.origin)
                # Add dummy values for properties only needed for
                # model fitting.
                spectrum = FixedParameter(componentData.spectrum)
                totalBands = len(spectrum.x)
                morph = FixedParameter(componentData.morph)
                factorized = FactorizedComponent(
                    bands=("dummy",) * totalBands,
                    spectrum=spectrum,
                    morph=morph,
                    peak=tuple(int(np.round(p)) for p in componentData.peak),  # type: ignore
                    bbox=bbox,
                    bg_rms=np.full((totalBands,), np.nan),
                )
                model = factorized.get_model().data[bandIndex][None, :, :]
                model = scl.Image(model, yx0=bbox.origin, bands=bands)
                component = scl.io.ComponentCube(
                    model=model,
                    peak=factorized.peak,
                )
                components.append(component)

        source = scl.Source(components=components, metadata=sourceData.metadata)
        sources.append(source)

    bbox = scl.Box(blendData.shape, origin=blendData.origin)
    blend = scl.Blend(sources=sources, observation=observation[:, bbox])
    return blend


def updateCatalogFootprints(
    modelData: scl.io.ScarletModelData,
    catalog: SourceCatalog,
    band: str,
    imageForRedistribution: MaskedImage | Exposure | None = None,
    removeScarletData: bool = True,
    updateFluxColumns: bool = True,
) -> None:
    """Use the scarlet models to set HeavyFootprints for modeled sources

    Parameters
    ----------
    catalog:
        The catalog missing heavy footprints for deblended sources.
    band:
        The name of the band that the catalog data describes.
    imageForRedistribution:
        The image that is the source for flux re-distribution.
        If `imageForRedistribution` is `None` then flux re-distribution is
        not performed.
    removeScarletData:
        Whether or not to remove `ScarletBlendData` for each blend
        in order to save memory.
    updateFluxColumns:
        Whether or not to update the `deblend_*` columns in the catalog.
        This should only be true when the input catalog schema already
        contains those columns.
    """
    # All of the blends should have the same PSF,
    # so we extract it from the first blend data.
    if len(modelData.blends) == 0:
        raise ValueError("Scarlet model data is empty")
    if modelData.metadata is None:
        raise ValueError("Scarlet model data does not contain metadata")
    bands = modelData.metadata["bands"]
    try:
        bandIndex = bands.index(band)
    except ValueError:
        raise NoWorkFound(f"Band '{band}' not found in scarlet model data")
    modelPsf = modelData.metadata["model_psf"]
    observedPsf = modelData.metadata["psf"][bandIndex][None, :, :]

    # Flux re-distribution may mix depth=1 blends, so we iterate over the
    # completely flux separated parents to ensure that the full models
    # are used for each source.
    parents = catalog[catalog["parent"] == 0]

    for n, parentRecord in enumerate(parents):
        if parentRecord["deblend_nChild"] == 0:
            # No children, so it is either an isolated source or failed
            # deblending.
            # Either way we can skip creating heavy child footprints.
            continue

        if updateFluxColumns and imageForRedistribution is not None:
            # Update the data coverage (1 - # of NO_DATA pixels/# of pixels)
            parentRecord["deblend_dataCoverage"] = calculateFootprintCoverage(
                parentRecord.getFootprint(), imageForRedistribution.mask
            )

        observation = buildMonochromaticObservation(
            modelPsf=modelPsf,
            observedPsf=observedPsf,
            footprint=parentRecord.getFootprint(),
            imageForRedistribution=imageForRedistribution,
        )

        updateBlendRecords(
            modelData=modelData,
            bandIndex=bandIndex,
            parent=parentRecord,
            catalog=catalog,
            observation=observation,
            updateFluxColumns=updateFluxColumns,
            imageForRedistribution=imageForRedistribution,
            removeScarletData=removeScarletData,
        )


def buildMonochromaticObservation(
    modelPsf: np.ndarray,
    observedPsf: np.ndarray,
    footprint: afwFootprint | None = None,
    imageForRedistribution: MaskedImage | Exposure | None = None,
) -> scl.Observation:
    """Create a single-band observation for the entire image

    Parameters
    ----------
    modelPsf :
        The 2D model of the PSF.
    observedPsf :
        The observed PSF model for the catalog.
    footprint :
        The footprint of the source, used for masking out the model.
    imageForRedistribution:
        The image that is the source for flux re-distribution.
        If `imageForRedistribution` is `None` then flux re-distribution is
        not performed.

    Returns
    -------
    observation : `scarlet.lite.Observation`
        The observation for the entire image
    """
    if footprint is None and imageForRedistribution is None:
        raise ValueError("Either footprint or imageForRedistribution must be provided")

    if footprint is None:
        bbox = imageForRedistribution.getBBox()
    else:
        bbox = footprint.getBBox()

    scarletBox = utils.bboxToScarletBox(bbox)

    if imageForRedistribution is not None:
        cutout = imageForRedistribution[bbox]

        # Mask the footprint
        weights = np.ones(cutout.image.array.shape, dtype=cutout.image.array.dtype)

        if footprint is not None:
            weights *= footprint.spans.asArray()

        observation = scl.Observation(
            images=cutout.image.array[None, :, :],
            variance=cutout.variance.array[None, :, :],
            weights=weights[None, :, :],
            psfs=observedPsf,
            model_psf=modelPsf[None, :, :],
            convolution_mode="real",
            bands=monochromaticBands,
            bbox=scarletBox,
        )
    else:
        observation = scl.Observation.empty(
            bands=monochromaticBands,
            psfs=observedPsf,
            model_psf=modelPsf[None, :, :],
            bbox=scarletBox,
            dtype=modelPsf.dtype,
        )
    return observation


def calculateFootprintCoverage(footprint: afwFootprint, maskImage: MaskX) -> np.floating:
    """Calculate the fraction of pixels with no data in a Footprint

    Parameters
    ----------
    footprint : `lsst.afw.detection.Footprint`
        The footprint to check for missing data.
    maskImage : `lsst.afw.image.MaskX`
        The mask image with the ``NO_DATA`` bit set.
    Returns
    -------
    coverage : `float`
        The fraction of pixels in `footprint` where the ``NO_DATA`` bit is set.
    """
    # Store the value of "NO_DATA" from the mask plane.
    noDataInt = 2 ** maskImage.getMaskPlaneDict()["NO_DATA"]

    # Calculate the coverage in the footprint
    bbox = footprint.getBBox()
    if bbox.area == 0:
        # The source has no footprint, so it has no coverage
        return 0
    spans = footprint.spans.asArray()
    totalArea = footprint.getArea()
    mask = maskImage[bbox].array & noDataInt
    noData = (mask * spans) > 0
    coverage = 1 - np.sum(noData) / totalArea
    return coverage


def updateBlendRecords(
    modelData: scl.io.ScarletModelData,
    bandIndex: int,
    parent: SourceRecord,
    catalog: SourceCatalog,
    observation: scl.Observation,
    updateFluxColumns: bool,
    imageForRedistribution: MaskedImage | Exposure | None = None,
    removeScarletData: bool = True,
):
    """Create footprints and update band-dependent columns in the catalog

    Parameters
    ----------
    modelData :
        Persistable data for the entire catalog.
    bandIndex :
        The number of the band to extract.
    parent :
        The parent source record.
    catalog :
        The catalog that is being updated.
    observation :
        The observation of the blend.
    updateFluxColumns :
        Whether or not to update the `deblend_*` columns in the catalog.
        This should only be true when the input catalog schema already
        contains those columns.
    imageForRedistribution :
        The image that is the source for flux re-distribution.
        If `imageForRedistribution` is `None` then flux re-distribution is
        not performed.
    removeScarletData :
        Whether or not to remove `ScarletBlendData` for each blend
        to save memory.
    """
    useFlux = imageForRedistribution is not None

    # Create a blend with the parent and all of its children.
    blendData = modelData.blends[parent.getId()]
    sources = []
    if isinstance(blendData, scl.io.HierarchicalBlendData):
        for blendId in blendData.children:
            _blendData = cast(scl.io.ScarletBlendData, blendData.children[blendId])
            blend = monochromaticDataToScarlet(_blendData, bandIndex, observation)
            sources.extend(blend.sources)

    if len(sources) == 0:
        # No sources to update, so we can skip the rest of the function.
        return

    blend = scl.Blend(
        sources=sources,
        observation=observation,
    )

    if useFlux:
        blend.conserve_flux()

    # Set the metrics for the blend.
    # TODO: remove this once DM-34558 runs all deblender metrics
    # in a separate task.
    if updateFluxColumns:
        setDeblenderMetrics(blend)

    # Update the HeavyFootprints for deblended sources
    # and update the band-dependent catalog columns.
    for source in blend.sources:
        sourceRecord = catalog.find(source.metadata["id"])

        peaks = parent.getFootprint().peaks
        peakIdx = np.where(peaks["id"] == source.metadata["peak_id"])[0][0]
        source.detectedPeak = peaks[peakIdx]
        # Set the Footprint
        heavy = scarletModelToHeavy(
            source=source,
            blend=blend,
            useFlux=useFlux,
        )

        if updateFluxColumns:
            if heavy.getArea() == 0:
                # The source has no flux after being weighted with the PSF
                # in this particular band (it might have flux in others).
                sourceRecord.set("deblend_zeroFlux", True)
                # Create a Footprint with a single pixel, set to zero,
                # to avoid breakage in measurement algorithms.
                center = Point2I(heavy.peaks[0]["i_x"], heavy.peaks[0]["i_y"])
                spanList = [Span(center.y, center.x, center.x)]
                footprint = afwFootprint(SpanSet(spanList))
                footprint.setPeakCatalog(heavy.peaks)
                heavy = HeavyFootprintF(footprint)
                heavy.getImageArray()[0] = 0.0
            else:
                sourceRecord.set("deblend_zeroFlux", False)
            sourceRecord.setFootprint(heavy)

            if useFlux:
                # Set the fraction of pixels with valid data.
                coverage = calculateFootprintCoverage(
                    heavy, imageForRedistribution.mask
                )
                sourceRecord.set("deblend_dataCoverage", coverage)

            # Set the flux of the scarlet model
            # TODO: this field should probably be deprecated,
            # since DM-33710 gives users access to the scarlet models.
            model = source.get_model().data[0]
            sourceRecord.set("deblend_scarletFlux", np.sum(model))

            # Set the flux at the center of the model
            peak = heavy.peaks[0]

            img = heavy.extractImage(fill=0.0)
            try:
                sourceRecord.set(
                    "deblend_peak_instFlux", img[Point2I(peak["i_x"], peak["i_y"])]
                )
            except Exception:
                srcId = sourceRecord.getId()
                x = peak["i_x"]
                y = peak["i_y"]
                logger.warning(
                    f"Source {srcId} at {x},{y} could not set the peak flux with error:",
                    exc_info=1,
                )
                sourceRecord.set("deblend_peak_instFlux", np.nan)
        else:
            sourceRecord.setFootprint(heavy)
    if removeScarletData:
        if parent.getId() in modelData.blends:
            del modelData.blends[parent.getId()]


def build_scarlet_model(zip_dict: dict[str, Any]) -> scl.io.ScarletModelData:
    """Build a ScarletModelData instance from a dictionary of files.

    Parameters
    ----------
    zip_dict : dict[str, Any]
        Dictionary mapping filenames to the desired file type.

    Returns
    -------
    model :
        ScarletModelData instance.
    """
    metadata = zip_dict.pop('metadata', None)
    if metadata is None:
        model_psf = zip_dict.pop('psf')
        psf_shape = zip_dict.pop('psf_shape')
        metadata = {
            'psf': model_psf,
            'psfShape': psf_shape,
        }
    blends = {}
    for key, value in zip_dict.items():
        blends[int(key)] = value
    return scl.io.ScarletModelData.parse_obj({
        'blends': blends,
        'metadata': metadata,
    })


def read_scarlet_model(path_or_stream: str, blend_ids: list[int] | None = None) -> scl.io.ScarletModelData:
    """Read a zip file and return a ScarletModelData instance.

    Parameters
    ----------
    path : `str`
        Path to the zip file.
    blend_ids : `list[int]`, optional
        List of blend IDs to extract from the zip file. If None,
        all blends in the dataset will be extracted.

    Returns
    -------
    model :
        ScarletModelData instance.
    """

    if blend_ids is not None:
        filenames = [str(f) for f in blend_ids]
    else:
        filenames = None

    with zipfile.ZipFile(path_or_stream, 'r') as zip_file:
        unzipped_files = {}
        if filenames is None:
            filenames = zip_file.namelist()
        # Attempt to read the metadata file first, if it exists.
        try:
            with zip_file.open('metadata') as f:
                metadata = from_json(f.read())
                unzipped_files['metadata'] = metadata
        except ValueError:
            # The metadata file is not present, so we will
            # assume that the model is in the legacy format.
            filenames += ['psf', 'psf_shape']
        for filename in filenames:
            with zip_file.open(filename) as f:
                unzipped_files[filename] = from_json(f.read())

        return build_scarlet_model(unzipped_files)


def scarlet_model_to_zip_json(model_data: scl.io.ScarletModelData) -> dict[str, Any]:
    """Convert a ScarletModelData instance to a dictionary of files.

    This is required to convert the model data into a format that
    can be insterted into a zip archive.

    Parameters
    ----------
    model_data : `lsst.scarelt.lite.io.ScarletModelData`
        ScarletModelData instance.

    Returns
    -------
    data : dict[str, Any]
        Dictionary mapping filenames to the desired file type.
    """
    json_model = model_data.as_dict()

    data = {
        str(blend_id): json.dumps(blend_data)
        for blend_id, blend_data in json_model['blends'].items()
    }
    # Support for legacy models
    if 'psf' in json_model:
        data.update({
            'psf_shape': json.dumps(json_model['psfShape']),
            'psf': json.dumps(json_model['psf']),
        })
    else:
        data.update({
            'metadata': json.dumps(json_model['metadata']),
        })
    return data


def write_scarlet_model(path_or_stream: str | BinaryIO, model_data: scl.io.ScarletModelData):
    """Write a ScarletModelData instance to a zip file.

    Parameters
    ----------
    model_data : `lsst.scarlet.lite.io.ScarletModelData`
        ScarletModelData instance.

    Returns
    -------
    zip_dict :
        Dictionary mapping filenames to the desired file type.
    """
    with zipfile.ZipFile(path_or_stream, 'w') as zf:
        zip_archive = scarlet_model_to_zip_json(model_data)
        for filename, data in zip_archive.items():
            zf.writestr(filename, data)


class ScarletModelFormatter(FormatterV2):
    """Read and write scarlet models.
    """

    default_extension = ".scarlet"
    unsupported_parameters = frozenset()
    can_read_from_stream = True
    can_read_from_local_file = True

    def read_from_local_file(self, path: str, component: str | None = None, expected_size: int = -1) -> Any:
        # Override of `FormatterV2.read_from_local_file`.
        return read_scarlet_model(path)

    def read_from_stream(
        self, stream: BinaryIO | ResourceHandleProtocol, component: str | None = None, expected_size: int = -1
    ) -> Any:
        # Override of `FormatterV2.read_from_stream`.
        if self.file_descriptor.parameters is not None and "blend_id" in self.file_descriptor.parameters:
            blend_ids = lsst_utils.iteration.ensure_iterable(self.file_descriptor.parameters["blend_id"])
        else:
            return NotImplemented

        return read_scarlet_model(stream, blend_ids=blend_ids)

    def to_bytes(self, in_memory_dataset: Any) -> bytes:
        # Override of `FormatterV2.to_bytes`.
        in_memory_zip = BytesIO()
        write_scarlet_model(in_memory_zip, in_memory_dataset)
        return in_memory_zip.getvalue()


class ScarletModelDelegate(StorageClassDelegate):
    """Delegate to extract a blend from an in-memory ScarletModelData object.
    """
    def can_accept(self, inMemoryDataset: Any) -> bool:
        return isinstance(inMemoryDataset, scl.io.ScarletModelData)

    def getComponent(self, composite: Any, componentName: str) -> Any:
        raise AttributeError(f"Unsupported component: {componentName}")

    def handleParameters(self, inMemoryDataset: Any, parameters: Mapping[str, Any] | None = None) -> Any:
        if "blend_id" in parameters:
            blend_ids = lsst_utils.iteration.ensure_iterable(parameters["blend_id"])
            blends = {blend_id: inMemoryDataset.blends[blend_id] for blend_id in blend_ids}
            inMemoryDataset.blends = blends
        elif parameters is not None:
            raise ValueError(f"Unsupported parameters: {parameters}")
        return inMemoryDataset


def loadBlend(blendData: scl.io.ScarletBlendData, model_psf: np.ndarray, mCoadd: MultibandExposure):
    """Load a blend from the persisted data

    Parameters
    ----------
    blendData:
        The persisted scarlet BlendData to load into the blend.
    model_psf:
        The psf of the model in each band. This should be 2D, as scarlet
        lite assumes that the PSF is the same for all bands.
    mCoadd:
        The coadd image to use for the observation attached to the blend.
        This is required in order to create a difference kernel to convolve
        the model into an observed seeing.

    Returns
    -------
    blend : `scarlet.lite.Blend`
        The blend object loaded from the persisted data.
    """
    psf, _ = utils.computePsfKernelImage(mCoadd, blendData.psf_center)
    bbox = Box(blendData.shape, origin=blendData.origin)
    afw_box = Box2I(Point2I(bbox.origin[::-1]), Extent2I(bbox.shape[::-1]))
    coadd = mCoadd[blendData.bands, afw_box]
    observation = scl.Observation(
        images=coadd.image.array,
        variance=coadd.variance.array,
        weights=np.ones(coadd.image.array.shape, dtype=np.float32),
        psfs=psf,
        model_psf=model_psf[None, :, :],
        convolution_mode='real',
        bands=mCoadd.bands,
        bbox=bbox,
    )
    return blendData.to_blend(observation), afw_box
