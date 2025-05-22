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

import logging

import lsst.scarlet.lite as scl
import numpy as np
from lsst.afw.detection import Footprint as afwFootprint
from lsst.afw.detection import HeavyFootprintF
from lsst.afw.geom import Span, SpanSet
from lsst.afw.image import Exposure, MaskedImage, MaskX
from lsst.afw.table import SourceCatalog, SourceRecord
from lsst.geom import Point2I

from .metrics import setDeblenderMetrics
from . import utils
from .footprint import scarletModelToHeavy, footprintsToNumpy

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
            bbox = scl.Box(componentData.shape, origin=componentData.origin)
            model = scl.io.Image(
                componentData.model[bandIndex][None, :, :], yx0=bbox.origin, bands=bands
            )
            component = scl.io.ComponentCube(
                bands=bands,
                model=model,
                peak=tuple(componentData.peak[::-1]),
                bbox=bbox,
            )
            components.append(component)

        for factorizedData in sourceData.factorized_components:
            bbox = scl.Box(factorizedData.shape, origin=factorizedData.origin)
            # Add dummy values for properties only needed for
            # model fitting.
            spectrum = scl.FixedParameter(factorizedData.spectrum)
            totalBands = len(spectrum.x)
            morph = scl.FixedParameter(factorizedData.morph)
            factorized = scl.FactorizedComponent(
                bands=monochromaticBands * totalBands,
                spectrum=spectrum,
                morph=morph,
                peak=tuple(int(np.round(p)) for p in factorizedData.peak),  # type: ignore
                bbox=bbox,
                bg_rms=np.full((totalBands,), np.nan),
            )
            model = factorized.get_model().data[bandIndex][None, :, :]
            model = scl.io.Image(model, yx0=bbox.origin, bands=bands)
            component = scl.io.ComponentCube(
                bands=bands,
                model=model,
                peak=factorized.peak,
                bbox=factorized.bbox,
            )
            components.append(component)

        source = scl.Source(components=components)
        source.record_id = sourceId
        source.peak_id = sourceData.peak_id
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
    refBlend = next(iter(modelData.blends.values()))
    bands = refBlend.bands
    bandIndex = bands.index(band)
    observedPsf = refBlend.psf[bandIndex][None, :, :]

    # Flux re-distribution may mix depth=1 blends, so we iterate over the
    # completely flux separated parents to ensure that the full models
    # are used for each source.
    parents = catalog[catalog["parent"] == 0]

    for parentRecord in parents:
        parentId = parentRecord.getId()

        children = catalog.getChildren(parentId)

        if len(children) == 0:
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
            modelPsf=modelData.psf,
            observedPsf=observedPsf,
            footprint=parentRecord.getFootprint(),
            imageForRedistribution=imageForRedistribution,
        )

        updateBlendRecords(
            modelData=modelData,
            bandIndex=bandIndex,
            parent=parentRecord,
            children=children,
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
    catalog :
        The catalog that is being updated.
    modelPsf :
        The 2D model of the PSF.
    observedPsf :
        The observed PSF model for the catalog.
    imageForRedistribution:
        The image that is the source for flux re-distribution.
        If `imageForRedistribution` is `None` then flux re-distribution is
        not performed.
    bbox :
        The bounding box of the image to create the weight image for.

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

        observation = scl.io.Observation(
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
        observation = scl.io.Observation.empty(
            bands=monochromaticBands,
            psfs=observedPsf,
            model_psf=modelPsf[None, :, :],
            bbox=bbox,
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
    children: SourceCatalog,
    catalog: SourceCatalog,
    observation: scl.Observation,
    updateFluxColumns: bool,
    imageForRedistribution: MaskedImage | Exposure | None = None,
    removeScarletData: bool = True,
):
    """Create footprints and update band-dependent columns in the catalog

    Parameters
    ----------
    blendData:
        Persistable data for the entire blend.
    catalog:
        The catalog that is being updated.
    modelPsf:
        The 2D model of the PSF.
    observedPsf:
        The observed PSF model for the catalog.
    imageForRedistribution:
        The image that is the source for flux re-distribution.
        If `imageForRedistribution` is `None` then flux re-distribution is
        not performed.
    bandIndex:
        The number of the band to extract.
    blendFootprint:
        The footprint of the blend, used for masking out the model
        when re-distributing flux.
    updateFluxColumns:
        Whether or not to update the `deblend_*` columns in the catalog.
        This should only be true when the input catalog schema already
        contains those columns.
    """
    useFlux = imageForRedistribution is not None

    # Create a blend with the parent and all of its children.
    if parent.getId() in modelData.blends:
        blendData = modelData.blends[parent.getId()]
        blend = monochromaticDataToScarlet(blendData, bandIndex, observation)
        sources = blend.sources
    else:
        sources = []

    for child in children:
        if child.getId() in modelData.blends:
            # The child is a blend, so we need to update the
            # parent with the child blend.
            blendData = modelData.blends[child.getId()]
            blend = monochromaticDataToScarlet(
                blendData, bandIndex, observation,
            )
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
        sourceRecord = catalog.find(source.record_id)

        peaks = parent.getFootprint().peaks
        peakIdx = np.where(peaks["id"] == source.peak_id)[0][0]
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
