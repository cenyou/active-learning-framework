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

import math

__all__=[
    'OVERALL_VARIANCE',
    'AL_FUNCTION_VARIANCE_UPPERBOUND',
    'AL_FUNCTION_VARIANCE_LOWERBOUND',
    'SAFE_AL_FUNCTION_VARIANCE_UPPERBOUND',
    'SAFE_AL_FUNCTION_VARIANCE_LOWERBOUND',
    'MAX_DIMENSION',
]

from alef.configs.base_parameters import NOISE_VARIANCE_LOWER_BOUND

OVERALL_VARIANCE = 1.0 + 1e-4 # k + observation noise variance during amortized training
__AL_SN_RATIO = 5.0 # signal to noise ratio
__SAFE_AL_SN_RATIO = 5.0 # signal to noise ratio

SAFE_INTERVAL_SIZE = 0.2 # the ratio of interval so the final safe area is sth like (center - 1/2 size, center + 1/2 size)*domain_weight

######
AL_MEAN_VARIANCE = 0.0 # overall_variance = mean_var + kernel_var + noise_var
AL_FUNCTION_VARIANCE_UPPERBOUND = 1.0 # SNR = 100
AL_FUNCTION_VARIANCE_LOWERBOUND = OVERALL_VARIANCE * (__AL_SN_RATIO**2) / (__AL_SN_RATIO**2 + 1)
# function variance = mean_var + kernel_var
assert OVERALL_VARIANCE >= AL_MEAN_VARIANCE + AL_FUNCTION_VARIANCE_UPPERBOUND + NOISE_VARIANCE_LOWER_BOUND
assert AL_FUNCTION_VARIANCE_UPPERBOUND > AL_FUNCTION_VARIANCE_LOWERBOUND

SAFE_AL_MEAN_VARIANCE = 0.5 # overall_variance = mean_var + kernel_var + noise_var
SAFE_AL_FUNCTION_VARIANCE_UPPERBOUND = 1.0 # SNR = 100
SAFE_AL_FUNCTION_VARIANCE_LOWERBOUND = OVERALL_VARIANCE * (__SAFE_AL_SN_RATIO**2) / (__SAFE_AL_SN_RATIO**2 + 1)
assert OVERALL_VARIANCE >= SAFE_AL_FUNCTION_VARIANCE_UPPERBOUND + NOISE_VARIANCE_LOWER_BOUND
assert SAFE_AL_FUNCTION_VARIANCE_UPPERBOUND > SAFE_AL_FUNCTION_VARIANCE_LOWERBOUND

MAX_DIMENSION = 5


if __name__=='__main__':
    print(f'overall variance: {OVERALL_VARIANCE}')
    print(f'mean variance: {AL_MEAN_VARIANCE}')
    print(f'AL function variance interval: [{AL_FUNCTION_VARIANCE_LOWERBOUND}, {AL_FUNCTION_VARIANCE_UPPERBOUND}]')
    print(f'AL    noise variance interval: [{OVERALL_VARIANCE - AL_FUNCTION_VARIANCE_UPPERBOUND}, {OVERALL_VARIANCE - AL_FUNCTION_VARIANCE_LOWERBOUND}]')
    print('###')
    print(f'mean variance: {SAFE_AL_MEAN_VARIANCE}')
    print(f'SAL function variance interval: [{SAFE_AL_FUNCTION_VARIANCE_LOWERBOUND}, {SAFE_AL_FUNCTION_VARIANCE_UPPERBOUND}]')
    print(f'SAL    noise variance interval: [{OVERALL_VARIANCE - SAFE_AL_FUNCTION_VARIANCE_UPPERBOUND}, {OVERALL_VARIANCE - SAFE_AL_FUNCTION_VARIANCE_LOWERBOUND}]')