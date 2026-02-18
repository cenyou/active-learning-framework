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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from enum import Enum

from alef.utils.plot_utils import active_learning_1d_plot, active_learning_2d_plot, active_learning_nd_plot
from alef.enums.data_sets_enums import InputPreprocessingType, OutputPreprocessingType
from alef.utils.utils import normalize_data, min_max_normalize_data
from alef.data_sets.base_data_set import StandardDataSet
import os


class GEngine1(StandardDataSet):
    def __init__(self, base_path: str):
        super().__init__()
        self.training_file_path = os.path.join(base_path, 'gengine1_data_training.xlsx')
        self.test_file_path = os.path.join(base_path, 'gengine1_data_test.xlsx')
        self.load_from_test: bool=False
        # has the following sheets:
        #       'X_raw', 'X_processed', 'U_raw', 'U_processed', 'Y_raw', 'Y_processed'
        
        #self.input_preprocessing_type = InputPreprocessingType.NONE
        #self.output_preprocessing_type = OutputPreprocessingType.NONE
        self.name = "Engine1"

    def load_data_set(self):
        path = self.test_file_path if self.load_from_test else self.training_file_path
        X_df = pd.read_excel(
            path,
            sheet_name = 'U_processed',
            header=[0],
            index_col=[0]
        )
        Y_df = pd.read_excel(
            path,
            sheet_name = 'Y_processed',
            header=[0],
            index_col=[0]
        )
        # remove outlier (based on histogram)
        low = X_df.iloc[:, 7:].quantile(0.025)
        high  = X_df.iloc[:, 7:].quantile(0.975)
        mask_X = ( (X_df.iloc[:, 7:] < high) & (X_df.iloc[:, 7:] > low) ).product(axis=1).to_numpy()
        #
        high  = Y_df['HC'].quantile(0.95)
        mask_Y = (Y_df['HC'] < high).to_numpy()
        #
        low  = Y_df['T_EXM'].quantile(0.025)
        mask_Y *= (Y_df['T_EXM'] > low).to_numpy()

        X_df = X_df.loc[np.logical_and(mask_X, mask_Y), :]
        Y_df = Y_df.loc[np.logical_and(mask_X, mask_Y),:]

        self.x = X_df.to_numpy()
        self.y = Y_df.to_numpy()

        self.input_names = X_df.columns
        self.output_names = Y_df.columns
        self.length = self.x.shape[0]
        self.input_dimension = self.x.shape[1]
        self.output_dimension = self.y.shape[1]


class GEngine2(StandardDataSet):
    def __init__(self, base_path: str):
        super().__init__()
        self.training_file_path = os.path.join(base_path, 'gengine2_data_training.xlsx')
        self.test_file_path = os.path.join(base_path, 'gengine2_data_test.xlsx')
        self.load_from_test: bool=False
        # has the following sheets:
        #       'X_raw', 'X_processed', 'U_raw', 'U_processed', 'Y_raw', 'Y_processed'
        
        #self.input_preprocessing_type = InputPreprocessingType.NONE
        #self.output_preprocessing_type = OutputPreprocessingType.NONE
        self.name = "Engine1"

    def load_data_set(self):
        path = self.test_file_path if self.load_from_test else self.training_file_path
        X_df = pd.read_excel(
            path,
            sheet_name = 'U_processed',
            header=[0],
            index_col=[0]
        )
        Y_df = pd.read_excel(
            path,
            sheet_name = 'Y_processed',
            header=[0],
            index_col=[0]
        )
        # remove outlier (based on histogram)
        low = X_df.iloc[:, 7:10].quantile(0.05)
        mask_X = (X_df.iloc[:, 7:10] > low).product(axis=1).to_numpy()
        #
        high  = Y_df['HC'].quantile(0.975)
        mask_Y = (Y_df['HC'] < high).to_numpy()

        X_df = X_df.loc[np.logical_and(mask_X, mask_Y), :]
        Y_df = Y_df.loc[np.logical_and(mask_X, mask_Y),:]

        self.x = X_df.to_numpy()
        self.y = Y_df.to_numpy()

        self.input_names = X_df.columns
        self.output_names = Y_df.columns
        self.length = self.x.shape[0]
        self.input_dimension = self.x.shape[1]
        self.output_dimension = self.y.shape[1]
