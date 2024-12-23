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
from lsst.afw.table import SourceCatalog
from lsst.geom import Point2I

from .metrics import setDeblenderMetrics
from . import utils

logger = logging.getLogger(__name__)


# The name of the band in an monochome blend.
# This is used as a placeholder since the band is not used in the
# monochromatic model.
monochromeBand = "dummy"
monochromeBands = (monochromeBand,)


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
    bands = monochromeBands
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
                bands=monochromeBands * totalBands,
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
    bbox: scl.Box | None = None,
) -> dict[int, scl.Blend]:
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
    refBlend = next(iter(modelData.blends.values()))
    bands = refBlend.bands
    bandIndex = bands.index(band)
    observedPsf = refBlend.psf[bandIndex][None, :, :]
    blends = extraxctMonochrmaticBlends(
        modelData=modelData,
        catalog=catalog,
        modelPsf=modelData.psf,
        observedPsf=observedPsf,
        imageForRedistribution=imageForRedistribution,
        bandIndex=bandIndex,
        removeScarletData=removeScarletData,
        bbox=bbox,
    )

    for blendId, blend in blends.items():
        # Update the data coverage (1 - # of NO_DATA pixels/# of pixels)
        if imageForRedistribution is not None:
            parentRecord = catalog.find(blendId)
            calculateFootprintCoverage(
                parentRecord.getFootprint(), imageForRedistribution.mask
            )
        updateBlendRecords(
            blend=blend,
            catalog=catalog,
            imageForRedistribution=imageForRedistribution,
            updateFluxColumns=updateFluxColumns,
        )

    return blends


def buildMonochromeObservation(
    catalog: SourceCatalog,
    modelPsf: np.ndarray,
    observedPsf: np.ndarray,
    imageForRedistribution: MaskedImage | Exposure | None,
    bbox: scl.Box | None = None,
) -> scl.Observation:
    useFlux = imageForRedistribution is not None

    if useFlux:
        if bbox is None:
            bbox = utils.bboxToScarletBox(imageForRedistribution.getBBox())
        assert bbox is not None  # needed for typing
        parents = catalog[catalog["deblend_level"] == 0]
        footprintImage = utils.footprintsToNumpy(parents, bbox.shape, bbox.origin[::-1])
        # Extract the image array to re-distribute its flux
        images = scl.Image(
            imageForRedistribution.image.array[None, :, :],
            yx0=bbox.origin,
            bands=monochromeBands,
        )

        variance = scl.Image(
            imageForRedistribution.variance.array[None, :, :],
            yx0=bbox.origin,
            bands=monochromeBands,
        )

        weights = scl.Image(
            footprintImage[None, :, :],
            yx0=bbox.origin,
            bands=monochromeBands,
        )

        observation = scl.io.Observation(
            images=images,
            variance=variance,
            weights=weights,
            psfs=observedPsf,
            model_psf=modelPsf[None, :, :],
        )
    else:
        observation = scl.io.Observation.empty(
            bands=monochromeBands,
            psfs=observedPsf,
            model_psf=modelPsf[None, :, :],
            bbox=bbox,
            dtype=np.float32,
        )
    return observation


def extraxctMonochrmaticBlends(
    modelData: scl.ScarletModelData,
    catalog: SourceCatalog,
    modelPsf: np.ndarray,
    observedPsf: np.ndarray,
    imageForRedistribution: MaskedImage | Exposure | None,
    bandIndex: int,
    removeScarletData: bool = True,
    bbox: scl.Box | None = None,
) -> dict[int, scl.Blend]:
    """Extract the monochromatic blends from the scarlet model data

    Parameters
    ----------
    modelData:
        The scarlet model data.
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
    removeScarletData:
        Whether or not to remove `ScarletBlendData` for each blend
        in order to save memory.
    bbox:
        The bounding box of the image to create the weight image for.

    Returns
    -------
    blends :
        A dictionary of blends extracted from the model data.
    """
    blends = {}
    # Create an observation for the entire image
    observation = buildMonochromeObservation(
        catalog=catalog,
        modelPsf=modelPsf,
        observedPsf=observedPsf,
        imageForRedistribution=imageForRedistribution,
        bbox=bbox,
    )

    for blendId, blendData in modelData.blends.items():
        blend = monochromaticDataToScarlet(
            blendData=blendData,
            bandIndex=bandIndex,
            observation=observation,
        )
        blends[blendId] = blend
        # Save memory by removing the data for the blend
        if removeScarletData:
            del modelData.blends[blendId]

    if imageForRedistribution is not None:
        weightImage = createMonochromeWeightImage(list(blends.values()), observation)
        for blend in blends.values():
            # Re-distribute the flux using the ratio of the blends flux to
            # the flux in the image.
            blend.conserve_flux(weight_image=weightImage)

    return blends


def createMonochromeWeightImage(blends: list[scl.Blend], observation: scl.Observation) -> scl.Image:
    """Create a weight image from the scarlet model data

    Parameters
    ----------
    modelData:
        The scarlet model data.
    bbox :
        The bounding box of the image to create the weight image for.

    Returns
    -------
    weightImage : `lsst.afw.image.Image`
        The weight image with the same dimensions as the model data.
    """
    # Create a weight image with the same dimensions as the model data
    weightImage = scl.Image.from_box(observation.bbox, bands=monochromeBands)
    for blend in blends:
        for source in blend.sources:
            # Note that this is the deconvolvd source model
            model = source.get_model()
            weightImage += model

    weightImage = observation.convolve(weightImage)

    # Due to ringing in the PSF, the convolved model can have
    # negative values. We take the absolute value to avoid
    # negative fluxes in the flux weighted images.
    weightImage.data[:] = np.abs(weightImage.data)

    return weightImage


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
    blend: scl.Blend,
    catalog: SourceCatalog,
    imageForRedistribution: MaskedImage | Exposure | None,
    updateFluxColumns: bool,
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

    # Set the metrics for the blend.
    # TODO: remove this once DM-34558 runs all deblender metrics
    # in a separate task.
    if updateFluxColumns:
        setDeblenderMetrics(blend)

    # Update the HeavyFootprints for deblended sources
    # and update the band-dependent catalog columns.
    for source in blend.sources:
        sourceRecord = catalog.find(source.record_id)

        parent = catalog.find(sourceRecord["parent"])
        peaks = parent.getFootprint().peaks
        peakIdx = np.where(peaks["id"] == source.peak_id)[0][0]
        source.detectedPeak = peaks[peakIdx]
        # Set the Footprint
        heavy = utils.scarletModelToHeavy(
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
                sourceRecord.setFootprint(heavy)
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
