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

from scarlet.blend import Blend

__all__ = ["LsstBlend"]


class LsstBlend(Blend):
    """LSST Blend of sources

    It is possible that LSST blends might require different
    funtionality than those in scarlet, which is being designed
    for multiresolution blends. So this class exists for any
    LSST specific changes.
    """
    def get_model(self, seds=None, morphs=None, observation=None):
        model = super().get_model(seds, morphs)
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
