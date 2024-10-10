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

import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.scarlet.lite as scl

from . import utils

log = logging.getLogger(__name__)

__all__ = [
    "MultiBandPeakDetectionTask",
    "MultiBandPeakDetectionConfig",
    "MultiBandPeakDetectionConnections",
]


class MultiBandPeakDetectionConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap"),
    defaultTemplates={"inputCoaddName": "deep"},
):
    """Connections for MultiBandPeakDetectionTask"""

    coadds = cT.Input(
        doc="Exposure on which to run deblending",
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap"),
    )

    peaks = cT.Output(
        doc="Output catalog of detected peak positions",
        name="{inputCoaddName}_coadd_multiband_peaks",
        storageClass="PeakCatalog",
        dimensions=("tract", "patch", "skymap"),
    )


class MultiBandPeakDetectionConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=MultiBandPeakDetectionConnections,
):
    """Configuration for MultiBandPeakDetectionTask"""

    waveletGeneration = pexConfig.Field(
        dtype=int,
        default=2,
        doc="Generation of the starlet wavelet used for peak detection (should be 1 or 2)",
    )
    waveletScales = pexConfig.Field(
        dtype=int,
        default=1,
        doc="Number of wavelet scales used for peak detection",
    )
    minFootprintArea = pexConfig.Field(
        dtype=int,
        default=4,
        doc="Minimum area of a footprint to be considered detectable",
    )
    minPeakDistance = pexConfig.Field(
        dtype=int,
        default=4,
        doc="Minimum distance between peaks. "
        "Peaks closer than this distance to an existing peak will not be included",
    )
    minPeakSNR = pexConfig.Field(
        dtype=float,
        default=5,
        doc="Minimum signal-to-noise ratio for a peak to be considered detectable",
    )
    minFootprintSNR = pexConfig.Field(
        dtype=float,
        default=5,
        doc="Minimum signal-to-noise ratio for a pixel to be considered part of a footprint",
    )


class MultiBandPeakDetectionTask(pipeBase.PipelineTask):
    """Detect peaks in a multi-band image

    This task performs a starlet (wavelet) decomposition of a multi-band image,
    removes the high frequency channel, and then detects peaks
    in the low frequency residual image.

    The detected peaks are returned as a `PeakCatalog`.
    """

    ConfigClass = MultiBandPeakDetectionConfig
    _DefaultName = "multibandPeakDetection"

    def __init__(self, initInputs=None, **kwargs):
        if initInputs is None:
            initInputs = {}
        super().__init__(initInputs=initInputs, **kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["filters"] = [dRef.dataId["band"] for dRef in inputRefs.coadds]
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, coadds: list[afwImage.Exposure], bands: list[str]) -> pipeBase.Struct:
        # Create a MulitbandExposure from the list of coadds
        mCoadd = afwImage.MultibandExposure.fromExposures(bands, coadds)
        xmin, ymin = mCoadd.getBBox().getMin()
        # Detect peaks
        footprints = scl.detect.detect_footprints(
            images=mCoadd.image.array,
            variance=mCoadd.variance.array,
            scales=self.config.waveletScales,
            generation=self.config.waveletGeneration,
            origin=(ymin, xmin),
            min_separation=self.config.minPeakDistance,
            min_area=self.config.minFootprintArea,
            peak_thresh=self.config.minPeakSNR,
            footprint_thresh=self.config.minFootprintSNR,
            find_peaks=True,
            remove_high_freq=True,
            min_pixel_detect=1,
        )
        peaks = utils.scarletFootprintsToPeakCatalog(footprints)
        return pipeBase.Struct(peaks=peaks)
