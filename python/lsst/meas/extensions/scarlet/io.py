from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
import logging
import numpy as np
from scarlet.bbox import Box, overlapped_slices
from scarlet.lite import LiteBlend, LiteFactorizedComponent, LiteObservation, LiteSource, LiteParameter
from scarlet.lite.measure import weight_sources

from lsst.geom import Box2I, Extent2I, Point2I, Point2D
from lsst.afw.detection.multiband import heavyFootprintToImage

from .source import liteModelToHeavy

__all__ = [
    "ScarletComponentData",
    "ScarletFactorizedComponentData",
    "ScarletSourceData",
    "ScarletBlendData",
    "ScarletModelData",
    "updateBlendRecords",
    "boundedDataToBox",
    "ComponentCube",
    "dataToScarlet",
    "scarletLiteToData",
    "scarletToData",
]

logger = logging.getLogger(__name__)


@dataclass
class ScarletComponentData:
    """Data for a component expressed as a 3D data cube

    For now this is used for scarlet main source models because
    their structure is too complex to persist in the same
    way that scarlet lite components can be persisted.

    Note that both `xy0` and `extent` use lsst ``(x, y)`` convention,
    not the scarlet/C++ ``(y, x)`` convention.

    Attributes
    ----------
    xy0 : `tuple` of `int`
        The lower bound of the components bounding box.
    extent : `tuple` of `int`
        The `(width, height)` of the component array.
    center : `tuple` of `int`
        The center of the component.
    model : `numpy.ndarray`
        The model for the component.
    """
    xy0: tuple[int, int]
    extent: tuple[int, int]
    center: tuple[float, float]
    model: np.ndarray

    def asDict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result : `dict`
            The object encoded as a JSON compatible dict
        """
        return {
            "xy0": self.xy0,
            "extent": self.extent,
            "center": self.extent,
            "model": tuple(self.model.flatten().astype(float))
        }

    @classmethod
    def fromDict(cls, data: dict) -> "ScarletComponentData":
        """Reconstruct `ScarletComponentData` from JSON compatible dict

        Parameters
        ----------
        data : `dict`
            Dictionary representation of the object

        Returns
        -------
        result : `ScarletComponentData`
            The reconstructed object
        """
        dataShallowCopy = dict(data)
        dataShallowCopy["xy0"] = tuple(data["xy0"])
        dataShallowCopy["extent"] = tuple(data["extent"])
        shape = dataShallowCopy['extent'][::-1]
        numBands = shape[0] * shape[1]
        dataShallowCopy['model'] = np.array(data['model']).reshape((numBands,) + shape).astype(np.float32)
        return cls(**dataShallowCopy)


@dataclass
class ScarletFactorizedComponentData:
    """Data for a factorized component

    Note that both `xy0` and `extent` use lsst ``(x, y)`` convention,
    not the scarlet/C++ ``(y, x)`` convention.

    Attributes
    ----------
    xy0 : `tuple` of `int`
        The lower bound of the components bounding box.
    extent : `tuple` of `int`
        The `(width, height)` of the component array.
    center : `tuple` of `int`
        The ``(x, y)`` center of the component.
        Note: once this is converted into a scarlet `LiteBlend` the source has
        the traditional c++ `(y, x)` ordering.
    sed : `numpy.ndarray`
        The SED of the component.
    morph : `numpy.ndarray`
        The 2D morphology of the component.
    """
    xy0: tuple[int, int]
    extent: tuple[int, int]
    center: tuple[float, float]
    sed: np.ndarray
    morph: np.ndarray

    def asDict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result : `dict`
            The object encoded as a JSON compatible dict
        """
        return {
            "xy0": self.xy0,
            "extent": self.extent,
            "center": self.center,
            "sed": tuple(self.sed.astype(float)),
            "morph": tuple(self.morph.flatten().astype(float))
        }

    @classmethod
    def fromDict(cls, data: dict) -> "ScarletFactorizedComponentData":
        """Reconstruct `ScarletFactorizedComponentData` from JSON compatible
        dict.

        Parameters
        ----------
        data : `dict`
            Dictionary representation of the object

        Returns
        -------
        result : `ScarletFactorizedComponentData`
            The reconstructed object
        """
        dataShallowCopy = dict(data)
        dataShallowCopy["xy0"] = tuple(data["xy0"])
        dataShallowCopy["extent"] = tuple(data["extent"])
        shape = dataShallowCopy['extent'][::-1]
        dataShallowCopy["sed"] = np.array(data["sed"]).astype(np.float32)
        dataShallowCopy['morph'] = np.array(data['morph']).reshape(shape).astype(np.float32)
        return cls(**dataShallowCopy)


@dataclass
class ScarletSourceData:
    """Data for a scarlet source

    Attributes
    ----------
    components : `list` of `ScarletComponentData`
        The components contained in the source that are not factorized.
    factorizedComponents : `list` of `ScarletFactorizedComponentData`
        The components contained in the source that are factorized.
    peakId : `int`
        The peak ID of the source in it's parent's footprint peak catalog.
    """
    components: list[ScarletComponentData]
    factorizedComponents: list[ScarletFactorizedComponentData]
    peakId: int

    def asDict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result : `dict`
            The object encoded as a JSON compatible dict
        """
        result = {
            "components": [],
            "factorized": [],
            "peakId": self.peakId,
        }
        for component in self.components:
            reduced = component.asDict()
            result["components"].append(reduced)

        for component in self.factorizedComponents:
            reduced = component.asDict()
            result["factorized"].append(reduced)
        return result

    @classmethod
    def fromDict(cls, data: dict) -> "ScarletSourceData":
        """Reconstruct `ScarletSourceData` from JSON compatible
        dict.

        Parameters
        ----------
        data : `dict`
            Dictionary representation of the object

        Returns
        -------
        result : `ScarletSourceData`
            The reconstructed object
        """
        dataShallowCopy = dict(data)
        del dataShallowCopy["factorized"]
        components = []
        for component in data['components']:
            component = ScarletComponentData.fromDict(component)
            components.append(component)
        dataShallowCopy['components'] = components

        factorized = []
        for component in data["factorized"]:
            component = ScarletFactorizedComponentData.fromDict(component)
            factorized.append(component)
        dataShallowCopy['factorizedComponents'] = factorized
        dataShallowCopy["peakId"] = int(data["peakId"])
        return cls(**dataShallowCopy)


@dataclass
class ScarletBlendData:
    """Data for an entire blend.

    Note that `xy0`, `extent`, and `psfCenter` use lsst ``(x, y)`` convention,
    not the scarlet/C++ ``(y, x)`` convention.

    Attributes
    ----------
    xy0 : `tuple` of `int`
        The lower bound of the components bounding box.
    extent : `tuple` of `int`
        The `(width, height)` of the component array.
    sources : `dict` of `int`: `ScarletSourceData`
        Data for the sources contained in the blend.
    psfCenter : `tuple` of `int`
        The location used for the center of the PSF for
        the blend.
    """
    xy0: tuple[int, int]
    extent: tuple[int, int]
    sources: dict[int, ScarletSourceData]
    psfCenter: tuple[float, float]

    def asDict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result : `dict`
            The object encoded as a JSON compatible dict
        """
        result: dict[str, Any] = {"xy0": self.xy0, "extent": self.extent, "psfCenter": self.psfCenter}
        result['sources'] = {id: source.asDict() for id, source in self.sources.items()}
        return result

    @classmethod
    def fromDict(cls, data: dict) -> "ScarletBlendData":
        """Reconstruct `ScarletBlendData` from JSON compatible
        dict.

        Parameters
        ----------
        data : `dict`
            Dictionary representation of the object

        Returns
        -------
        result : `ScarletBlendData`
            The reconstructed object
        """
        dataShallowCopy = dict(data)
        dataShallowCopy["xy0"] = tuple(data["xy0"])
        dataShallowCopy["extent"] = tuple(data["extent"])
        dataShallowCopy["psfCenter"] = tuple(data["psfCenter"])
        dataShallowCopy["sources"] = {int(id): ScarletSourceData.fromDict(source)
                                      for id, source in data['sources'].items()}
        return cls(**dataShallowCopy)


class ScarletModelData:
    """A container that propagates scarlet models for an entire `SourceCatalog`
    """
    def __init__(self, bands, psf, blends=None):
        """Initialize an instance

        Parameters
        ----------
        bands : `list` of `str`
            The names of the bands.
            The order of the bands must be the same as the order of
            the multiband model arrays, and SEDs.
        psf : `numpy.ndarray`
            The 2D array of the PSF in scarlet model space.
            This is typically a narrow Gaussian integrated over the
            pixels in the exposure.
        blends : `dict` of [`int`: `ScarletBlendData`]
            Initial `dict` that maps parent IDs from the source catalog
            to the scarlet model data for the parent blend.
        """
        self.bands = bands
        self.psf = psf
        if blends is None:
            blends = {}
        self.blends = blends

    def json(self) -> str:
        """Serialize the data model to a JSON formatted string

        Returns
        -------
        result : `str`
            The result of the object converted into a JSON format
        """
        result = {
            "bands": self.bands,
            "psfShape": self.psf.shape,
            "psf": list(self.psf.flatten()),
            "blends": {id: blend.asDict() for id, blend in self.blends.items()}
        }
        return json.dumps(result)

    @classmethod
    def parse_obj(cls, data: dict) -> "ScarletModelData":
        """Construct a ScarletModelData from python decoded JSON object.

        Parameters
        ----------
        inMemoryDataset : `Mapping`
            The result of json.load(s) on a JSON persisted ScarletModelData

        Returns
        -------
        result : `ScarletModelData`
            The `ScarletModelData` that was loaded the from the input object
        """
        dataShallowCopy = dict(data)
        modelPsf = np.array(
            dataShallowCopy["psf"]).reshape(dataShallowCopy.pop("psfShape")).astype(np.float32)
        dataShallowCopy["psf"] = modelPsf
        dataShallowCopy["blends"] = {
            int(id): ScarletBlendData.fromDict(blend)
            for id, blend in data['blends'].items()
        }
        return cls(**dataShallowCopy)

    def updateCatalogFootprints(self, catalog, band, psfModel, redistributeImage=None,
                                removeScarletData=True, updateFluxColumns=True):
        """Use the scarlet models to set HeavyFootprints for modeled sources

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            The catalog missing heavy footprints for deblended sources.
        band : `str`
            The name of the band that the catalog data describes.
        psfModel : `lsst.afw.detection.Psf`
            The observed PSF model for the catalog.
        redistributeImage : `lsst.afw.image.Image`
            The image that is the source for flux re-distribution.
            If `redistributeImage` is `None` then flux re-distribution is
            not performed.
        removeScarletData : `bool`
            Whether or not to remove `ScarletBlendData` for each blend
            in order to save memory.
        updateFluxColumns : `bool`
            Whether or not to update the `deblend_*` columns in the catalog.
            This should only be true when the input catalog schema already
            contains those columns.
        """
        # Iterate over the blends, since flux re-distribution must be done on
        # all of the children with the same parent
        parents = catalog[catalog["parent"] == 0]
        # Get the index of the model for the given band
        bandIndex = self.bands.index(band)

        for parentRecord in parents:
            parentId = parentRecord.getId()

            try:
                blendModel = self.blends[parentId]
            except KeyError:
                # The parent was skipped in the deblender, so there are
                # no models for its sources.
                continue
            updateBlendRecords(
                blendData=blendModel,
                catalog=catalog,
                modelPsf=self.psf,
                observedPsf=psfModel,
                redistributeImage=redistributeImage,
                bandIndex=bandIndex,
                parentFootprint=parentRecord.getFootprint(),
                updateFluxColumns=updateFluxColumns,
            )

            # Save memory by removing the data for the blend
            if removeScarletData:
                del self.blends[parentId]


def updateBlendRecords(blendData, catalog, modelPsf, observedPsf, redistributeImage, bandIndex,
                       parentFootprint, updateFluxColumns):
    """Create footprints and update band-dependent columns in the catalog

    Parameters
    ----------
    blendData : `ScarletBlendData`
        Persistable data for the entire blend.
    catalog : `lsst.afw.table.SourceCatalog`
        The catalog that is being updated.
    modelPsf : `numpy.ndarray`
        The 2D model of the PSF.
    observedPsf : `lsst.afw.detection.Psf`
        The observed PSF model for the catalog.
    redistributeImage : `lsst.afw.image.Image`
        The image that is the source for flux re-distribution.
        If `redistributeImage` is `None` then flux re-distribution is
        not performed.
    bandIndex : `int`
        The number of the band to extract.
    parentFootprint : `lsst.afw.Footprint`
        The footprint of the parent, used for masking out the model
        when re-distributing flux.
    updateFluxColumns : `bool`
        Whether or not to update the `deblend_*` columns in the catalog.
        This should only be true when the input catalog schema already
        contains those columns.
    """
    # We import here to avoid a circular dependency
    from .scarletDeblendTask import setDeblenderMetrics, getFootprintMask

    useFlux = redistributeImage is not None
    xy0 = Point2I(*blendData.xy0)

    blend = dataToScarlet(
        blendData=blendData,
        nBands=1,
        bandIndex=bandIndex,
        dtype=np.float32,
    )

    position = Point2D(*blendData.psfCenter)
    psfs = observedPsf.computeKernelImage(position).array[None, :, :]
    modelBox = Box((1,) + tuple(blendData.extent[::-1]), origin=(0, 0, 0))
    blend.observation = DummyObservation(
        psfs=psfs,
        model_psf=modelPsf[None, :, :],
        bbox=modelBox,
        dtype=np.float32,
    )

    # Set the metrics for the blend.
    # TODO: remove this once DM-34558 runs all deblender metrics
    # in a separate task.
    if updateFluxColumns:
        setDeblenderMetrics(blend)

    # Update the source models if the scarlet models are used as
    # templates to re-distribute flux from an observation
    if useFlux:
        # Extract the image array to re-distribute its flux
        extent = Extent2I(*blendData.extent)
        bbox = Box2I(xy0, extent)
        blend.observation.images = redistributeImage[bbox].array[None, :, :]
        blend.observation.weights = ~getFootprintMask(parentFootprint, None)[None, :, :]
        # Re-distribute the flux for each source in-place
        weight_sources(blend)

    # Update the HeavyFootprints for deblended sources
    # and update the band-dependent catalog columns.
    for source in blend.sources:
        sourceRecord = catalog.find(source.recordId)
        parent = catalog.find(sourceRecord["parent"])
        peaks = parent.getFootprint().peaks
        peakIdx = np.where(peaks["id"] == source.peakId)[0][0]
        source.detectedPeak = peaks[peakIdx]
        # Set the Footprint
        heavy = liteModelToHeavy(
            source=source,
            blend=blend,
            xy0=xy0,
            useFlux=useFlux,
        )
        sourceRecord.setFootprint(heavy)

        if updateFluxColumns:
            # Set the flux of the scarlet model
            # TODO: this field should probably be deprecated,
            # since DM-33710 gives users access to the scarlet models.
            model = source.get_model()[0]
            sourceRecord.set("deblend_scarletFlux", np.sum(model))

            # Set the flux at the center of the model
            peak = heavy.peaks[0]
            img = heavyFootprintToImage(heavy, fill=0.0)
            try:
                sourceRecord.set("deblend_peak_instFlux", img.image[Point2I(peak["i_x"], peak["i_y"])])
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


def boundedDataToBox(nBands, boundedData):
    """Convert bounds from the data storage format to a `scarlet.bbox.Box`

    Parameters
    ----------
    nBands : `int`
        The number of bands in the model.
    boundedData :
        The scarlet data object containing `xy0` and `extent`
        attributes giving bounding box information in the lsst format
        `(x, y)`.

    Returns
    -------
    bbox : `scarlet.bbox.Box`
        The scarlet bounding box generated by the bounds.
    """
    xy0 = (0, ) + boundedData.xy0[::-1]
    extent = (nBands, ) + boundedData.extent[::-1]
    bbox = Box(shape=extent, origin=xy0)
    return bbox


class ComponentCube:
    """Dummy component for scarlet main sources.

    This is duck-typed to a `scarlet.lite.LiteComponent` in order to
    generate a model from the component.

    If scarlet lite ever implements a component as a data cube,
    this class can be removed.
    """
    def __init__(self, model, center, bbox, model_bbox):
        """Initialization

        Parameters
        ----------
        model : `numpy.ndarray`
            The 3D (bands, y, x) model of the component.
        center : `tuple` of `int`
            The `(y, x)` center of the component.
        bbox : `scarlet.bbox.Box`
            The bounding box of the component.
        `model_bbox` : `scarlet.bbox.Box`
            The bounding box of the entire blend.
        """
        self.model = model
        self.center = center
        self.bbox = bbox

    def get_model(self, bbox=None):
        """Generate the model for the source

        Parameters
        ----------
        bbox : `scarlet.bbox.Box`
            The bounding box to insert the model into.
            If `bbox` is `None` then the model is returned in its own
            bounding box.

        Returns
        -------
        model : `numpy.ndarray`
            The model as a 3D `(band, y, x)` array.
        """
        model = self.model
        if bbox is not None:
            slices = overlapped_slices(bbox, self.bbox)
            _model = np.zeros(bbox.shape, model.dtype)
            _model[slices[0]] = model[slices[1]]
            model = _model
        return model


class DummyParameter(LiteParameter):
    """A parameter place holder

    Models in scarlet have parameters, not arrays,
    for their sed's and morphologies, so this wrapper for
    the SED and morphology arrays implements the required
    methods and attributes.
    """
    def __init__(self, x):
        self.x = x
        self.grad = None

    def update(self, it, input_grad, *args):
        pass

    def grow(self, new_shape, dist):
        pass

    def shrink(self, dist):
        pass


class DummyObservation(LiteObservation):
    """An observation that does not have any image data

    In order to reproduce a model in an observed seeing we make use of the
    scarlet `LiteObservation` class, but since we are not fitting the model
    to data we can use empty arrays for the image, variance, and weight data,
    and zero for the `noise_rms`.

    Parameters
    ----------
    psfs : `numpy.ndarray`
        The array of PSF images in each band
    psf_model : `numpy.ndarray`
        The image of the model PSF.
    bbox : `scarlet.bbox.Box`
    dtype : `numpy.dtype`
        The data type of the model that is generated.
    """
    def __init__(self, psfs, model_psf, bbox, dtype):
        dummyImage = np.zeros([], dtype=dtype)

        super().__init__(
            images=dummyImage,
            variance=dummyImage,
            weights=dummyImage,
            psfs=psfs,
            model_psf=model_psf,
            convolution_mode="real",
            noise_rms=0,
            bbox=bbox,
        )


def dataToScarlet(blendData, nBands=None, bandIndex=None, dtype=np.float32):
    """Convert the storage data model into a scarlet lite blend

    Parameters
    ----------
    blendData : `ScarletBlendData`
        Persistable data for the entire blend.
    nBands : `int`
        The number of bands in the image.
        If `bandIndex` is `None` then this parameter is ignored and
        the number of bands is set to 1.
    bandIndex : `int`
        Index of model to extract. If `bandIndex` is `None` then the
        full model is extracted.
    dtype : `numpy.dtype`
        The data type of the model that is generated.

    Returns
    -------
    blend : `scarlet.lite.LiteBlend`
        A scarlet blend model extracted from persisted data.
    """
    if bandIndex is not None:
        nBands = 1
    modelBox = Box((nBands,) + tuple(blendData.extent[::-1]), origin=(0, 0, 0))
    sources = []
    for sourceId, sourceData in blendData.sources.items():
        components = []
        for componentData in sourceData.components:
            bbox = boundedDataToBox(nBands, componentData)
            if bandIndex is None:
                model = componentData.model
            else:
                model = componentData.model[bandIndex][None, :, :]
            component = ComponentCube(
                model=model,
                center=tuple(componentData.center[::-1]),
                bbox=bbox,
            )
            components.append(component)
        for componentData in sourceData.factorizedComponents:
            bbox = boundedDataToBox(nBands, componentData)
            # Add dummy values for properties only needed for
            # model fitting.
            if bandIndex is None:
                sed = componentData.sed
            else:
                sed = componentData.sed[bandIndex:bandIndex+1]
            sed = DummyParameter(sed)
            morph = DummyParameter(componentData.morph)
            # Note: since we aren't fitting a model, we don't need to
            # set the RMS of the background.
            # We set it to NaN just to be safe.
            component = LiteFactorizedComponent(
                sed=sed,
                morph=morph,
                center=tuple(componentData.center[::-1]),
                bbox=bbox,
                model_bbox=modelBox,
                bg_rms=np.nan
            )
            components.append(component)

        source = LiteSource(components=components, dtype=dtype)
        source.recordId = sourceId
        source.peakId = sourceData.peakId
        sources.append(source)

    return LiteBlend(sources=sources, observation=None)


def scarletLiteToData(blend, psfCenter, xy0):
    """Convert a scarlet lite blend into a persistable data object

    Parameters
    ----------
    blend : `scarlet.lite.LiteBlend`
        The blend that is being persisted.
    psfCenter : `tuple` of `int`
        The center of the PSF.
    xy0 : `tuple` of `int`
        The lower coordinate of the entire blend.

    Returns
    -------
    blendData : `ScarletBlendDataModel`
        The data model for a single blend.
    """
    sources = {}
    for source in blend.sources:
        components = []
        for component in source.components:
            if isinstance(component, LiteFactorizedComponent):
                componentData = ScarletFactorizedComponentData(
                    xy0=tuple(int(x) for x in component.bbox.origin[1:][::-1]),
                    extent=tuple(int(x) for x in component.bbox.shape[1:][::-1]),
                    center=tuple(int(x) for x in component.center[::-1]),
                    sed=component.sed,
                    morph=component.morph,
                )
            else:
                componentData = ScarletComponentData(
                    xy0=tuple(int(x) for x in component.bbox.origin[1:][::-1]),
                    extent=tuple(int(x) for x in component.bbox.shape[1:][::-1]),
                    center=tuple(int(x) for x in component.center[::-1]),
                    model=component.get_model(),
                )
            components.append(componentData)
        sourceData = ScarletSourceData(
            components=[],
            factorizedComponents=components,
            peakId=source.peakId,
        )
        sources[source.recordId] = sourceData

    blendData = ScarletBlendData(
        xy0=(xy0.x, xy0.y),
        extent=blend.observation.bbox.shape[1:][::-1],
        sources=sources,
        psfCenter=psfCenter,
    )

    return blendData


def scarletToData(blend, psfCenter, xy0):
    """Convert a scarlet blend into a persistable data object

    Parameters
    ----------
    blend : `scarlet.Blend`
        The blend that is being persisted.
    psfCenter : `tuple` of `int`
        The center of the PSF.
    xy0 : `tuple` of `int`
        The lower coordinate of the entire blend.

    Returns
    -------
    blendData : `ScarletBlendDataModel`
        The data model for a single blend.
    """
    sources = {}
    for source in blend.sources:
        componentData = ScarletComponentData(
            xy0=tuple(int(x) for x in source.bbox.origin[1:][::-1]),
            extent=tuple(int(x) for x in source.bbox.shape[1:][::-1]),
            center=tuple(int(x) for x in source.center[::-1]),
            model=source.get_model(),
        )

        sourceData = ScarletSourceData(
            components=[componentData],
            factorizedComponents=[],
            peakId=source.peakId,
        )
        sources[source.recordId] = sourceData

    blendData = ScarletBlendData(
        xy0=(int(xy0.x), int(xy0.y)),
        extent=tuple(int(x) for x in blend.observation.bbox.shape[1:][::-1]),
        sources=sources,
        psfCenter=psfCenter,
    )

    return blendData
