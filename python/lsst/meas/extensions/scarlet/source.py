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

import numpy as np
from scarlet.source import PointSource, ExtendedSource, SourceInitError
from scarlet.component import BlendFlag, Component

import lsst.afw.image as afwImage
from lsst.afw.geom import SpanSet
from lsst.geom import Point2I
import lsst.afw.detection as afwDet

__all__ = ["LsstSource", "LsstHistory"]


class BlankSource(Component):
    """Placeholder for a source that cannot be initialized

    In DM-19790 this class will be removed when skipped sources are
    not deblended at all.
    """
    def __init__(self, frame, center):
        self.pixel_center = center
        sed = np.zeros((frame.C,))
        morph = np.zeros((frame.Ny, frame.Nx))
        super().__init__(frame, sed=sed, morph=morph, fix_sed=True, fix_morph=True)

    def update(self):
        return self


class LsstSource(ExtendedSource):
    """LSST Base Source

    It is most likely that a source in LSST might be slightly different
    than the default scarlet sources, so this class allows us to define the
    default initialization and update constraints for general sources in
    LSST images.
    """
    def __init__(self, frame, peak, observation, bgRms, bbox,
                 thresh=1, symmetric=True, monotonic=True, centerStep=5,
                 pointSource=False, **componentKwargs):
        xmin = bbox.getMinX()
        ymin = bbox.getMinY()
        center = np.array([peak.getIy()-ymin, peak.getIx()-xmin], dtype=int)
        initialized = False
        if not pointSource:
            try:
                super().__init__(frame, center, observation, bgRms, thresh,
                                 symmetric, monotonic, centerStep, **componentKwargs)
                initialized = np.any(self.sed>0)
                if not initialized:
                    print("Could not initialize")
            # For now scarlet does not throw an error when it cannot initialize
            # So we catch all initialization errors here, even if scarlet
            # doesn't recognize it as a SourceInitError
            except Exception as e:
                # If the source is too faint for background detection,
                # initialize it as a PointSource
                pass
        self.skipped = False
        if not initialized:
            try:
                PointSource.__init__(self, frame, center, observation, symmetric, monotonic,
                                     centerStep, **componentKwargs)
            except SourceInitError:
                # There is really no flux to use for initializing this source,
                # so just use an empty placeholder.
                # TODO: In the future we should probably just strip blank
                # sources from the blend to save processing time, but that
                # will require keeping track of source indices.
                # DM-19790
                BlankSource.__init__(self, frame, center)
                self.skipped = True

        self.detectedPeak = peak

    def get_model(self, sed=None, morph=None, observation=None):
        model = super().get_model(sed, morph)
        if observation is not None:
            model = observation.render(model)
        return model

    def display_model(self, observation=None, ax=None, filters=None, Q=10, stretch=1, show=True):
        import matplotlib.pyplot as plt
        from astropy.visualization import make_lupton_rgb

        model = self.get_model(observation=observation)
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
        if filters is None:
            filters = [2, 1, 0]
        imgRgb = make_lupton_rgb(image_r=model[filters[0]],  # numpy array for the r channel
                                 image_g=model[filters[1]],  # numpy array for the g channel
                                 image_b=model[filters[2]],  # numpy array for the b channel
                                 stretch=stretch, Q=Q)  # parameters used to stretch and scale the values
        ax.imshow(imgRgb, interpolation='nearest')
        if show:
            plt.show()

    def morphToHeavy(self, peakSchema, xy0=Point2I()):
        """Convert the morphology to a `HeavyFootprint`
        """
        mask = afwImage.MaskX(np.array(self.morph > 0, dtype=np.int32), xy0=xy0)
        ss = SpanSet.fromMask(mask)

        if len(ss) == 0:
            return None

        tfoot = afwDet.Footprint(ss, peakSchema=peakSchema)
        cy, cx = self.pixel_center
        xmin, ymin = xy0
        peakFlux = self.morph[cy, cx]
        tfoot.addPeak(cx+xmin, cy+ymin, peakFlux)
        timg = afwImage.ImageF(self.morph, xy0=xy0)
        timg = timg[tfoot.getBBox()]
        heavy = afwDet.makeHeavyFootprint(tfoot, afwImage.MaskedImageF(timg))
        return heavy

    def modelToHeavy(self, filters, xy0=Point2I(), observation=None, dtype=np.float32):
        """Convert the model to a `MultibandFootprint`
        """
        model = self.get_model(observation=observation).astype(dtype)
        mHeavy = afwDet.MultibandFootprint.fromArrays(filters, model, xy0=xy0)
        peakCat = afwDet.PeakCatalog(self.detectedPeak.table)
        peakCat.append(self.detectedPeak)
        for footprint in mHeavy:
            footprint.setPeakCatalog(peakCat)
        return mHeavy


class LsstHistory(LsstSource):
    """LsstSource with attributes for traceback
    """
    def __init__(self, *args, **kwargs):
        self.float_history = []
        self.convergence_hist = []
        self.iterations = []
        super().__init__(*args, **kwargs)

    def update(self):
        if self._parent is None:
            it = 0
        else:
            it = self._parent.it
        self.update_history(it)
        return super().update()

    def update_history(self, it):
        self.float_history.append(self.pixel_center)
        self.iterations.append(it)
        _sed = self.flags & BlendFlag.SED_NOT_CONVERGED
        _morph = self.flags & BlendFlag.MORPH_NOT_CONVERGED
        if _sed and _morph:
            self.convergence_hist.append("black")
        elif _sed:
            self.convergence_hist.append("red")
        elif _morph:
            self.convergence_hist.append("cyan")
        else:
            self.convergence_hist.append("green")

    def plot_history(self):
        """Plot the position and convergence history of an object
        """
        import matplotlib.pyplot as plt
        history = np.array(self.float_history)
        hy = history[:, 0]
        hx = history[:, 1]
        fig = plt.figure(figsize=(12, 3))
        ax = [fig.add_subplot(1, 2, n+1) for n in range(2)]
        x = self.iterations
        ax[0].scatter(x, hy, c=self.convergence_hist, s=5)
        ax[0].set_title("py")
        ax[1].scatter(x, hx, c=self.convergence_hist, s=5)
        ax[1].set_title("px")
        ax[0].set_ylim([hy[-1]-1, hy[-1]+1])
        ax[1].set_ylim([hx[-1]-1, hx[-1]+1])
        plt.show()
