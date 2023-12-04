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

import numpy as np

from lsst.afw.table import SourceCatalog
from lsst.afw.image import MaskedImage, Exposure
from lsst.afw.detection import Footprint as afwFootprint
from lsst.geom import Box2I, Extent2I, Point2I
import lsst.scarlet.lite as scl
from lsst.scarlet.lite import Blend, Source, Box, Component, FixedParameter, FactorizedComponent, Image

from .metrics import setDeblenderMetrics
from .utils import scarletModelToHeavy


logger = logging.getLogger(__name__)


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
    bands = ("dummy", )
    for sourceId, sourceData in blendData.sources.items():
        components: list[Component] = []
        # There is no need to distinguish factorized components from regular
        # components, since there is only one band being used.
        for componentData in sourceData.components:
            bbox = Box(componentData.shape, origin=componentData.origin)
            model = scl.io.Image(componentData.model[bandIndex][None, :, :], yx0=bbox.origin, bands=bands)
            component = scl.io.ComponentCube(
                bands=bands,
                model=model,
                peak=tuple(componentData.peak[::-1]),
                bbox=bbox,
            )
            components.append(component)

        for factorizedData in sourceData.factorized_components:
            bbox = Box(factorizedData.shape, origin=factorizedData.origin)
            # Add dummy values for properties only needed for
            # model fitting.
            spectrum = FixedParameter(factorizedData.spectrum)
            totalBands = len(spectrum.x)
            morph = FixedParameter(factorizedData.morph)
            factorized = FactorizedComponent(
                bands=("dummy", ) * totalBands,
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

        source = Source(components=components)
        source.record_id = sourceId
        source.peak_id = sourceData.peak_id
        sources.append(source)

    return Blend(sources=sources, observation=observation)


def updateCatalogFootprints(
    modelData: scl.io.ScarletModelData,
    catalog: SourceCatalog,
    band: str,
    imageForRedistribution: MaskedImage | Exposure | None = None,
    removeScarletData: bool = True,
    updateFluxColumns: bool = True
):
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
    # Iterate over the blends, since flux re-distribution must be done on
    # all of the children with the same parent
    parents = catalog[catalog["parent"] == 0]

    for parentRecord in parents:
        parentId = parentRecord.getId()

        try:
            blendModel = modelData.blends[parentId]
        except KeyError:
            # The parent was skipped in the deblender, so there are
            # no models for its sources.
            continue

        parent = catalog.find(parentId)
        if updateFluxColumns and imageForRedistribution is not None:
            # Update the data coverage (1 - # of NO_DATA pixels/# of pixels)
            parentRecord["deblend_dataCoverage"] = calculateFootprintCoverage(
                parent.getFootprint(),
                imageForRedistribution.mask
            )

        if band not in blendModel.bands:
            peaks = parent.getFootprint().peaks
            # Set the footprint and coverage of the sources in this blend
            # to zero
            for sourceId, sourceData in blendModel.sources.items():
                sourceRecord = catalog.find(sourceId)
                footprint = afwFootprint()
                peakIdx = np.where(peaks["id"] == sourceData.peak_id)[0][0]
                peak = peaks[peakIdx]
                footprint.addPeak(peak.getIx(), peak.getIy(), peak.getPeakValue())
                sourceRecord.setFootprint(footprint)
                if updateFluxColumns:
                    sourceRecord["deblend_dataCoverage"] = 0
            continue

        # Get the index of the model for the given band
        bandIndex = blendModel.bands.index(band)

        updateBlendRecords(
            blendData=blendModel,
            catalog=catalog,
            modelPsf=modelData.psf,
            imageForRedistribution=imageForRedistribution,
            bandIndex=bandIndex,
            parentFootprint=parentRecord.getFootprint(),
            updateFluxColumns=updateFluxColumns,
        )

        # Save memory by removing the data for the blend
        if removeScarletData:
            del modelData.blends[parentId]


def calculateFootprintCoverage(footprint, maskImage):
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
    noDataInt = 2**maskImage.getMaskPlaneDict()["NO_DATA"]

    # Calculate the coverage in the footprint
    bbox = footprint.getBBox()
    if bbox.area == 0:
        # The source has no footprint, so it has no coverage
        return 0
    spans = footprint.spans.asArray()
    totalArea = footprint.getArea()
    mask = maskImage[bbox].array & noDataInt
    noData = (mask * spans) > 0
    coverage = 1 - np.sum(noData)/totalArea
    return coverage


def updateBlendRecords(
    blendData: scl.io.ScarletBlendData,
    catalog: SourceCatalog,
    modelPsf: np.ndarray,
    imageForRedistribution: MaskedImage | Exposure | None,
    bandIndex: int,
    parentFootprint: afwFootprint,
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
    parentFootprint:
        The footprint of the parent, used for masking out the model
        when re-distributing flux.
    updateFluxColumns:
        Whether or not to update the `deblend_*` columns in the catalog.
        This should only be true when the input catalog schema already
        contains those columns.
    """
    useFlux = imageForRedistribution is not None
    bands = ("dummy",)
    # Only use the PSF for the current image
    psfs = np.array([blendData.psf[bandIndex]])

    if useFlux:
        # Extract the image array to re-distribute its flux
        xy0 = Point2I(*blendData.origin[::-1])
        extent = Extent2I(*blendData.shape[::-1])
        bbox = Box2I(xy0, extent)

        images = Image(
            imageForRedistribution[bbox].image.array[None, :, :],
            yx0=blendData.origin,
            bands=bands,
        )

        variance = Image(
            imageForRedistribution[bbox].variance.array[None, :, :],
            yx0=blendData.origin,
            bands=bands,
        )

        weights = Image(
            parentFootprint.spans.asArray()[None, :, :],
            yx0=blendData.origin,
            bands=bands,
        )

        observation = scl.io.Observation(
            images=images,
            variance=variance,
            weights=weights,
            psfs=psfs,
            model_psf=modelPsf[None, :, :],
        )
    else:
        observation = scl.io.Observation.empty(
            bands=bands,
            psfs=psfs,
            model_psf=modelPsf[None, :, :],
            bbox=Box(blendData.shape, blendData.origin),
            dtype=np.float32,
        )

    blend = monochromaticDataToScarlet(
        blendData=blendData,
        bandIndex=bandIndex,
        observation=observation,
    )

    if useFlux:
        # Re-distribute the flux in the images
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
        parent = catalog.find(sourceRecord["parent"])
        peaks = parent.getFootprint().peaks
        peakIdx = np.where(peaks["id"] == source.peak_id)[0][0]
        source.detectedPeak = peaks[peakIdx]
        # Set the Footprint
        heavy = scarletModelToHeavy(
            source=source,
            blend=blend,
            useFlux=useFlux,
        )
        sourceRecord.setFootprint(heavy)

        if updateFluxColumns:
            if useFlux:
                # Set the fraction of pixels with valid data.
                coverage = calculateFootprintCoverage(heavy, imageForRedistribution.mask)
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
                sourceRecord.set("deblend_peak_instFlux", img[Point2I(peak["i_x"], peak["i_y"])])
            except Exception:
                srcId = sourceRecord.getId()
                x = peak["i_x"]
                y = peak["i_y"]
                logger.warning(
                    f"Source {srcId} at {x},{y} could not set the peak flux with error:",
                    exc_info=1
                )
                sourceRecord.set("deblend_peak_instFlux", np.nan)

            # Set the metrics columns.
            # TODO: remove this once DM-34558 runs all deblender metrics
            # in a separate task.
            sourceRecord.set("deblend_maxOverlap", source.metrics.maxOverlap[0])
            sourceRecord.set("deblend_fluxOverlap", source.metrics.fluxOverlap[0])
            sourceRecord.set("deblend_fluxOverlapFraction", source.metrics.fluxOverlapFraction[0])
            sourceRecord.set("deblend_blendedness", source.metrics.blendedness[0])


def oldScarletToData(blend: Blend, psfCenter: tuple[int, int], xy0: Point2I):
    """Convert a scarlet.lite blend into a persistable data object

    Note: This converts a blend from the old version of scarlet.lite,
    which is deprecated, to the persistable data format used in the
    new scarlet lite package.
    This is kept to compare the two scarlet versions,
    and can be removed once the new lsst.scarlet.lite package is
    used in production.

    Parameters
    ----------
    blend:
        The blend that is being persisted.
    psfCenter:
        The center of the PSF.
    xy0:
        The lower coordinate of the entire blend.
    Returns
    -------
    blendData : `ScarletBlendDataModel`
        The data model for a single blend.
    """
    from scarlet import lite
    yx0 = (xy0.y, xy0.x)

    sources = {}
    for source in blend.sources:
        components = []
        factorizedComponents = []
        for component in source.components:
            origin = tuple(component.bbox.origin[i+1] + yx0[i] for i in range(2))
            peak = tuple(component.center[i] + yx0[i] for i in range(2))

            if isinstance(component, lite.LiteFactorizedComponent):
                componentData = scl.io.ScarletFactorizedComponentData(
                    origin=origin,
                    peak=peak,
                    spectrum=component.sed,
                    morph=component.morph,
                )
                factorizedComponents.append(componentData)
            else:
                componentData = scl.io.ScarletComponentData(
                    origin=origin,
                    peak=peak,
                    model=component.get_model(),
                )
                components.append(componentData)
        sourceData = scl.io.ScarletSourceData(
            components=components,
            factorized_components=factorizedComponents,
            peak_id=source.peak_id,
        )
        sources[source.record_id] = sourceData

    blendData = scl.io.ScarletBlendData(
        origin=(xy0.y, xy0.x),
        shape=blend.observation.bbox.shape[-2:],
        sources=sources,
        psf_center=psfCenter,
        psf=blend.observation.psfs,
        bands=blend.observation.bands,
    )

    return blendData
