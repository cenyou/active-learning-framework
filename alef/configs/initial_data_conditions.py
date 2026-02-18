# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
define where the initial data is sampled from
format:
    oracle/dataset name: (lower_bound, box width)
"""
UNCONSTRAINED_BOXES = {}
CONSTRAINED_BOXES = {
    'BraninHoo': (0.7, 0.2),
    #'LSQ': (0.5, 0.3),
    'Simionescu': (0.4, 0.2),
    'Townsend': (0.4, 0.2),
    'LGBB': (0.4, 0.2),
    'CartPoleConstrained': (0.35, 0.3),
    'Engine3D': (0.4, 0.2),
    'PowerPlant': (0.4, 0.2),
    'GasTransmissionCompressorDesign': (0.4, 0.2),
    'HighPressureFluidSystemReduce': (0.4, 0.2),
    'HighPressureFluidSystemReduceGPInterpolated': (0.4, 0.2),
    'HighPressureFluidSystem': (0.4, 0.2),
}

INITIAL_OUTPUT_HIGHER_BY = 0.7 # 1.3 # sample the initial data with output constraint safety_threshold_lower + INITIAL_OUTPUT_HIGHER_BY
INITIAL_OUTPUT_LOWER_BY = 0.7 # 1.3 # sample the initial data with output constraint safety_threshold_upper - INITIAL_OUTPUT_LOWER_BY
OUTPUT_LOWER_BOUND = 0.5
