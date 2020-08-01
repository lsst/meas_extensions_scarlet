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

# TODO: DM-26194: Remove this file.

__all__ = ["deblend", "ScarletDeblendConfig", "ScarletDeblendTask"]

import warnings
from deprecated.sphinx import deprecated
from .scarletDeblendTask import deblend, ScarletDeblendConfig, ScarletDeblendTask

message = ['Importing from the lsst.meas.extensions.scarlet.deblend module is deprecated.',
           'Please import from lsst.meas.extensions.scarlet instead.',
           'This module will be removed after release 21.0.0.']
warnings.warn(" ".join(message), category=FutureWarning)

ScarletDeblendConfig = deprecated(" ".join(message), category=FutureWarning)(ScarletDeblendConfig)
ScarletDeblendTask = deprecated(" ".join(message), category=FutureWarning)(ScarletDeblendTask)

# This one can't come from lsst.meas.extensions.scarlet, since
# lsst.meas.extensions.scarlet.deblend is shadowed by this file!
deblend = deprecated(" ".join([message[0],
                               message[1].replace('scarlet', 'scarlet.scarletDeblendTask'),
                               message[2]]),
                     category=FutureWarning)(deblend)
