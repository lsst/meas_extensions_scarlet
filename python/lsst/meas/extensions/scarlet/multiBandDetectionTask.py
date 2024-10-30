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

import logging
from typing import Sequence

import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.scarlet.lite as scl
import numpy as np
from lsst.afw.detection import Footprint as AfwFootprint
from lsst.afw.detection import PeakCatalog
from lsst.afw.detection.multiband import getSpanSetFromImages
from lsst.afw.table import IdFactory, SourceCatalog, SourceTable
from lsst.geom import Box2I, Point2I
from lsst.meas.base import SkyMapIdGeneratorConfig
from scipy.signal import convolve as scipy_convolve

from . import utils

log = logging.getLogger(__name__)

__all__ = [
    "MultiBandDetectionTask",
    "MultiBandDetectionConfig",
    "MultiBandDetectionConnections",
]


class MultiBandDetectionConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap"),
    defaultTemplates={"inputCoaddName": "deep"},
):
    """Connections for MultiBandDetectionTask"""

    coadds = cT.Input(
        doc="Exposure on which to run deblending",
        name="deconvolved_{inputCoaddName}_coadd",
        storageClass="ExposureF",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap"),
    )

    peaks = cT.Input(
        doc="Catalog of detected peak positions",
        name="{inputCoaddName}_coadd_multiband_peaks",
        storageClass="PeakCatalog",
        dimensions=("tract", "patch", "skymap"),
    )

    sources = cT.Output(
        doc="Output catalog of detected footprints",
        name="deconvolved_{inputCoaddName}_coadd_footprints",
        storageClass="SourceCatalog",
    )

    def __init__(self, *, config=None):
        if config.doDetectPeaks:
            # Deconvolution does not use input catalog
            self.inputs.remove("peaks")


class MultiBandDetectionConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=MultiBandDetectionConnections,
):
    """Configuration for MultiBandDetectionTask"""

    doDetectPeaks = pexConfig.Field[bool](
        doc="Detect peaks in the input images. "
        "If False then the multi-band peak catalog is used.",
        default=False,
    )
    minFootprintArea = pexConfig.Field[int](
        default=4,
        doc="Minimum area of a footprint to be considered detectable",
    )
    minPeakDistance = pexConfig.Field[int](
        default=4,
        doc="Minimum distance between peaks. "
        "Peaks closer than this distance to an existing peak will not be included",
    )
    minPeakSNR = pexConfig.Field[float](
        default=5,
        doc="Minimum signal-to-noise ratio for a peak to be considered detectable",
    )
    minFootprintSNR = pexConfig.Field[float](
        default=5,
        doc="Minimum signal-to-noise ratio for a pixel to be considered part of a footprint",
    )
    minPixelDetect = pexConfig.Field[int](
        default=2,
        doc="Minimum number of bands that a pixel must be detected in to be included in a footprint",
    )
    modelPsfSigma = pexConfig.Field[float](
        default=1.5,
        doc="Sigma of the circular Gaussian PSF used to detect peaks in the detection image. "
        "This is only used if doDetectPeaks is True",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()


class MultiBandDetectionTask(pipeBase.PipelineTask):
    """Detect footprints (and optionally peaks) in a multi-band image.

    This task is used to generate a set of deconvolved footprints that are
    intended to be used as initial blend models for the scarlet lite deblender.
    The task has the option to attach a set of previously determined peaks
    to the image or to detect peaks on the deconvolved models themselves
    (after convolving with a larger circular Gaussian PSF).
    """

    ConfigClass = MultiBandDetectionConfig
    _DefaultName = "multiBandDetection"

    def __init__(self, initInputs=None, **kwargs):
        if initInputs is None:
            initInputs = {}
        super().__init__(initInputs=initInputs, **kwargs)
        self.schema = SourceTable.makeMinimalSchema()

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["bands"] = [dRef.dataId["band"] for dRef in inputRefs.coadds]
        inputs["idFactory"] = self.config.idGenerator.apply(
            butlerQC.quantum.dataId
        ).make_table_id_factory()
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        coadds: list[afwImage.Exposure],
        bands: list[str],
        idFactory: IdFactory,
        peaks: PeakCatalog | None = None,
    ) -> pipeBase.Struct:
        """Run the detection task

        Parameters
        ----------
        coadds :
            List of coadds to detect footprints on.
        bands :
            List of band names corresponding to the coadds.
        idFactory :
            IdFactory to create source ids.
        peaks :
            Catalog of peaks to attach to the footprints.
            This is only necessary if config.doDetectPeaks is False.
        """
        if (not self.config.doDetectPeaks) and (peaks is None):
            raise ValueError("'peaks' must be provided if doDetectPeaks is False")
        # Create a MulitbandExposure from the list of coadds
        mCoadd = afwImage.MultibandExposure.fromExposures(bands, coadds)
        xmin, ymin = mCoadd.getBBox().getMin()
        # Detect footprints in the detection image
        footprints = scl.detect.detect_footprints(
            images=mCoadd.image.array,
            variance=mCoadd.variance.array,
            origin=(ymin, xmin),
            min_separation=self.config.minPeakDistance,
            min_area=self.config.minFootprintArea,
            peak_thresh=self.config.minPeakSNR,
            footprint_thresh=self.config.minFootprintSNR,
            find_peaks=False,
            remove_high_freq=False,
            min_pixel_detect=self.config.minPixelDetect,
        )

        # Create a source catalog from the detected footprints
        table = SourceTable.make(self.schema, idFactory)
        sources = SourceCatalog(table)
        for footprint in footprints:
            fp_ymin, fp_xmin = footprint.bbox.origin
            spans, _ = getSpanSetFromImages(
                footprint.data, xy0=Point2I((xmin + fp_xmin, ymin + fp_ymin))
            )
            _footprint = AfwFootprint(spans)
            source = sources.addNew()
            source.setFootprint(_footprint)

        if self.config.doDetectPeaks:
            # Detect peaks in the detection image
            # and update the footprints in place.
            peaks = self._detectPeaks(mCoadd, footprints)

        # Attach peaks from an external catalog to the footprints.
        self._attachExternalPeaks(peaks, footprints, sources, mCoadd.getBBox())

        return pipeBase.Struct(footprints=footprints, sources=sources)

    def _detectPeaks(
        self,
        mCoadd: afwImage.MultibandExposure,
        footprints: Sequence[scl.detect.Footprint],
    ):
        """Detect peaks in the detection image

        Parameters
        ----------
        mCoadd :
            Multiband exposure to detect peaks on.
        footprints :
            List of footprints already detected in the image.
        """
        # Build the noise weighted detection image.
        sigma = np.median(np.sqrt(mCoadd.variance.array), axis=(1, 2))
        detection = np.sum(mCoadd.image.array / sigma[:, None, None], axis=0)
        # Build a mask from the footprints to remove noise
        bbox = utils.bboxToScarletBox(mCoadd.getBBox())
        footprint_image = scl.detect.footprints_to_image(footprints, bbox).data
        detection *= footprint_image
        # Convolve the detection image with a larger circular Gaussian PSF
        model_psf = scl.utils.integrated_circular_gaussian(
            sigma=self.config.modelPsfSigma
        )
        convolved = scipy_convolve(detection, model_psf, mode="same")
        # Detect peaks in the convolved image
        xmin, ymin = mCoadd.getBBox().getMin()
        wide_footprints = scl.detect.detect_footprints(
            images=convolved,
            variance=mCoadd.variance.array,
            origin=(ymin, xmin),
            min_separation=self.config.minPeakDistance,
            min_area=self.config.minFootprintArea,
            peak_thresh=self.config.minPeakSNR,
            footprint_thresh=self.config.minFootprintSNR,
            find_peaks=True,
            remove_high_freq=False,
            min_pixel_detect=1,
        )
        peaks = utils.scarletFootprintsToPeakCatalog(wide_footprints)
        log.info("Total number of peaks detected:", len(peaks))
        return peaks

    def _attachExternalPeaks(
        self,
        peaks: PeakCatalog,
        footprints: Sequence[scl.detect.Footprint],
        sources: SourceCatalog,
        bbox: Box2I,
    ):
        """Attach peaks to the footprints in the source catalog

        Parameters
        ----------
        peaks :
            Catalog of peaks to attach to the footprints.
        footprints :
            List of footprints already detected in the image.
        sources :
            Source catalog to attach the peaks to.
        bbox :
            Bounding box of the entire image.
        """
        scarletBox = utils.bboxToScarletBox(bbox)
        xmin, ymin = bbox.getMin()
        footprint_image = scl.detect.footprints_to_image(footprints, scarletBox).data
        for peak in peaks:
            x = peak["i_x"] - xmin
            y = peak["i_y"] - ymin
            source_index = footprint_image[y, x]
            if source_index != 0:
                footprint = sources[source_index - 1].getFootprint()
                footprint.addPeak(peak["i_x"], peak["i_y"], peak["peakValue"])
            else:
                log.debug(f"no footprint at ({y}, {x})")
        return footprint_image
