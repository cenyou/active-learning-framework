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

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import os
import matplotlib.pyplot as plt
import matplotlib
from alef.enums.data_structure_enums import OutputType
from alef.utils.plotter import Plotter, PlotterPlotly
from alef.utils.plotter2D import Plotter2D
from alef.utils.histogram_plotter import HistogramPlotter
import seaborn as sns


def active_learning_nd_plot(x_data, y_data, save_plot=False, file_name=None, file_path=None):
    column_names = ["x" + str(i) for i in range(1, x_data.shape[1] + 1)] + ["y"]
    data = np.concatenate((x_data, y_data), axis=1)
    df = pd.DataFrame(data=data, columns=column_names)
    scatter_matrix(df, alpha=1.0, figsize=(6, 6), diagonal="kde")
    if save_plot:
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()
    else:
        plt.show()


def active_learning_1d_plot(
    x_grid,
    pred_mu_grid,
    pred_sigma_grid,
    x_data,
    y_data,
    x_query,
    y_query,
    gt_available=False,
    gt_x=None,
    gt_f=None,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    plotter_object = Plotter(1)
    if gt_available:
        plotter_object.add_gt_function(np.squeeze(gt_x), np.squeeze(gt_f), "black", 0)
    plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(y_data), "r", 0)
    plotter_object.add_datapoints(x_query, y_query, "green", 0)
    plotter_object.add_predictive_dist(np.squeeze(x_grid), np.squeeze(pred_mu_grid), np.squeeze(pred_sigma_grid), 0)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def active_learning_1d_plot_with_acquisition(
    x_grid,
    pred_mu_grid,
    pred_sigma_grid,
    acquisition_grid,
    x_data,
    y_data,
    x_query,
    y_query,
    gt_available=False,
    gt_x=None,
    gt_f=None,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    plotter_object = PlotterPlotly(2, width=1000, height_per_axis=400)
    plotter_object.add_predictive_dist(np.squeeze(x_grid), np.squeeze(pred_mu_grid), np.squeeze(pred_sigma_grid), 0)
    if gt_available:
        plotter_object.add_gt_function(np.squeeze(gt_x), np.squeeze(gt_f), "black", 0)
    plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(y_data), "red", 0)
    plotter_object.add_datapoints(x_query, y_query, "lime", 0)
    plotter_object.add_gt_function(np.squeeze(x_grid), np.squeeze(acquisition_grid), "black", 1)
    x_max = max(np.max(x_grid), np.max(x_data), x_query)
    x_min = max(np.min(x_grid), np.min(x_data), x_query)
    plotter_object.set_x_axes(x_min, x_max)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def active_learning_1d_plot_multioutput(
    x_grid, pred_mu_grid, pred_sigma_grid, x_data, y_data, x_query, y_query, save_plot=False, file_name=None, file_path=None
):
    output_dim = y_data.shape[1]
    assert output_dim == pred_mu_grid.shape[1]
    plotter_object = Plotter(output_dim)
    for m in range(0, output_dim):
        plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(y_data[:, m]), "r", m)
        plotter_object.add_datapoints(x_query, y_query[m], "green", m)
        plotter_object.add_predictive_dist(np.squeeze(x_grid), np.squeeze(pred_mu_grid[:, m]), np.squeeze(pred_sigma_grid[:, m]), m)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def active_learning_1d_plot_with_true_safety_measure(
    x_grid,
    pred_mu_grid,
    pred_sigma_grid,
    x_data,
    y_data,
    z_data,
    x_query,
    y_query,
    z_query,
    safety_thresholds_lower,
    safety_thresholds_upper,
    gt_available=False,
    gt_x=None,
    gt_f=None,
    gt_fz=None,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    plotter_object = Plotter(1, v_axes=2)
    if gt_available:
        plotter_object.add_gt_function(np.squeeze(gt_x), np.squeeze(gt_f), "black", 0, v_ax=0)
        plotter_object.add_gt_function(np.squeeze(gt_x), np.squeeze(gt_fz), "black", 0, v_ax=1)
    if hasattr(safety_thresholds_lower, '__len__'):
        plotter_object.add_multiple_hline(safety_thresholds_lower, 0, v_ax=1)
    else:
        plotter_object.add_hline(safety_thresholds_lower, "black", 0, v_ax=1)
    if hasattr(safety_thresholds_upper, '__len__'):
        plotter_object.add_multiple_hline(safety_thresholds_upper, 0, v_ax=1)
    else:
        plotter_object.add_hline(safety_thresholds_upper, "black", 0, v_ax=1)
    plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(y_data), "r", 0, v_ax=0)
    plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(z_data), "r", 0, v_ax=1)
    plotter_object.add_datapoints(x_query, y_query, "green", 0, v_ax=0)
    plotter_object.add_datapoints(x_query, z_query, "green", 0, v_ax=1)
        
    plotter_object.add_predictive_dist(np.squeeze(x_grid), np.squeeze(pred_mu_grid), np.squeeze(pred_sigma_grid), 0, v_ax=0)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def active_learning_2d_plot(
    x_grid, acquisition_values_grid, pred_mu_grid, y_over_grid, x_data, x_query, save_plot=False, file_name=None, file_path=None
):
    plotter_object = Plotter2D(3)
    plotter_object.add_gt_function(x_grid, np.squeeze(acquisition_values_grid), "RdBu_r", 14, 0)
    plotter_object.add_datapoints(x_data, "black", 0)
    if len(x_query.shape) == 1:
        x_query = np.expand_dims(x_query, axis=0)

    plotter_object.add_datapoints(x_query, "green", 0)
    min_y = np.min(y_over_grid)
    max_y = np.max(y_over_grid)
    levels = np.linspace(min_y, max_y, 100)
    plotter_object.add_gt_function(x_grid, np.squeeze(y_over_grid), "seismic", levels, 1)
    plotter_object.add_datapoints(x_data, "black", 1)
    plotter_object.add_datapoints(x_query, "green", 1)
    plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu_grid), "seismic", levels, 2)
    plotter_object.add_datapoints(x_data, "black", 2)
    plotter_object.add_datapoints(x_query, "green", 2)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def active_learning_2d_plot_with_true_safety_measure(
    x_grid,
    acquisition_values_grid,
    safety_score,
    pred_mu_grid,
    y_over_grid,
    z_over_grid,
    safety_bool_over_grid,
    x_data,
    x_query,
    save_plot=False,
    file_name=None,
    file_path=None
):
    plotter_object = Plotter2D(4, v_axes=2)
    if safety_score is not None:
        plotter_object.add_gt_function(x_grid, np.squeeze(acquisition_values_grid) + np.squeeze(safety_score), "RdBu_r", 14, 0, v_ax=0)
    else:
        plotter_object.add_gt_function(x_grid, np.squeeze(acquisition_values_grid), "RdBu_r", 14, 0, v_ax=0)
    levels = np.array([-0.5, 0.5, 1.5, 2.5])
    plotter_object.add_gt_function(x_grid, np.squeeze(safety_bool_over_grid), "YlGn", levels, 0, v_ax=1)
    if len(x_query.shape) == 1:
        x_query = np.expand_dims(x_query, axis=0)
    for i in range(2):
        plotter_object.add_datapoints(x_data, "black", 0, v_ax=i)
        plotter_object.add_datapoints(x_query, "green", 0, v_ax=i)
    plotter_object.configure_axes(0, v_ax=0, ax_title='acq_score + safety_score')
    plotter_object.configure_axes(0, v_ax=1, ax_title='safe region')

    plotter_object.add_gt_function(x_grid, np.squeeze(acquisition_values_grid), "RdBu_r", 14, 1, v_ax=0)
    if safety_score is not None:
        plotter_object.add_gt_function(x_grid, np.squeeze(safety_score), "RdBu_r", 14, 1, v_ax=1)
    for i in range(2):
        plotter_object.add_datapoints(x_data, "black", 1, v_ax=i)
        plotter_object.add_datapoints(x_query, "green", 1, v_ax=i)
    plotter_object.configure_axes(1, v_ax=0, ax_title='acq_score')
    plotter_object.configure_axes(1, v_ax=1, ax_title='safety_score')

    min_y = np.min(y_over_grid)
    max_y = np.max(y_over_grid)
    levels = np.linspace(min_y, max_y, 100)
    plotter_object.add_gt_function(x_grid, np.squeeze(y_over_grid), "seismic", levels, 2, v_ax=0)
    min_z = np.min(z_over_grid)
    max_z = np.max(z_over_grid)
    levels = np.linspace(min_z, max_z, 100)
    plotter_object.add_gt_function(x_grid, np.squeeze(z_over_grid), "seismic", levels, 2, v_ax=1)
    for i in range(2):
        plotter_object.add_datapoints(x_data, "black", 2, v_ax=i)
        plotter_object.add_datapoints(x_query, "green", 2, v_ax=i)
    plotter_object.configure_axes(2, v_ax=0, ax_title='function')
    plotter_object.configure_axes(2, v_ax=1, ax_title='constraint')

    plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu_grid), "seismic", levels, 3, v_ax=0)
    plotter_object.add_datapoints(x_data, "black", 3, v_ax=0)
    plotter_object.add_datapoints(x_query, "green", 3, v_ax=0)
    plotter_object.configure_axes(3, v_ax=0, ax_title='predictive mean')
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def active_learning_2d_plot_without_gt(
    x_grid, acquisition_values_grid, pred_mu_grid, x_data, x_query, save_plot=False, file_name=None, file_path=None
):
    plotter_object = Plotter2D(2)
    plotter_object.add_gt_function(x_grid, np.squeeze(acquisition_values_grid), "RdBu_r", 14, 0)
    plotter_object.add_datapoints(x_data, "black", 0)
    if len(x_query.shape) == 1:
        x_query = np.expand_dims(x_query, axis=0)

    plotter_object.add_datapoints(x_query, "green", 0)
    min_y = np.min(pred_mu_grid)
    max_y = np.max(pred_mu_grid)
    levels = np.linspace(min_y, max_y, 100)
    plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu_grid), "seismic", levels, 1)
    plotter_object.add_datapoints(x_data, "black", 1)
    plotter_object.add_datapoints(x_query, "green", 1)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def safe_active_learning_1d_plot(
    x_grid,
    pred_mu,
    pred_sigma,
    safety_mu,
    safety_sigma,
    safe_grid,
    x_data,
    y_data,
    safety_data,
    x_query,
    y_query,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    plotter_object = Plotter(2)
    plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(y_data), "r", 0)
    plotter_object.add_datapoints(x_query, y_query, "green", 0)
    plotter_object.add_predictive_dist(np.squeeze(x_grid), np.squeeze(pred_mu), np.squeeze(pred_sigma), 0)
    plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(safety_data), "r", 1)
    plotter_object.add_predictive_dist(np.squeeze(x_grid), np.squeeze(safety_mu), np.squeeze(safety_sigma), 1)
    plotter_object.add_safety_region(safe_grid, 1)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def safe_active_learning_2d_plot(
    x_grid,
    pred_mu,
    pred_sigma,
    safety_mu,
    safety_sigma,
    safety_estimate_over_grid,
    true_safety_over_grid,
    y_over_grid,
    x_data,
    x_query,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    plotter_object = Plotter2D(5)
    plotter_object.add_gt_function(x_grid, np.squeeze(pred_sigma), "RdBu_r", 14, 0)
    plotter_object.add_datapoints(x_data, "red", 0)
    plotter_object.add_datapoints(np.expand_dims(x_query, axis=0), "green", 0)
    levels_safety = np.linspace(-1.0, 1.02, 500)
    plotter_object.add_gt_function(x_grid, safety_estimate_over_grid, "RdBu_r", levels_safety, 1)
    plotter_object.add_datapoints(x_data, "red", 1)
    plotter_object.add_datapoints(np.expand_dims(x_query, axis=0), "green", 1)

    plotter_object.add_gt_function(x_grid, np.squeeze(true_safety_over_grid), "RdBu_r", levels_safety, 2)
    plotter_object.add_datapoints(x_data, "red", 2)
    plotter_object.add_datapoints(np.expand_dims(x_query, axis=0), "green", 2)

    min_y = np.min(y_over_grid)
    max_y = np.max(y_over_grid)
    max_abs = max(np.abs(max_y), np.abs(min_y))
    max_y = max_y + 0.5 * max_abs
    min_y = min_y - 0.5 * max_abs

    levels = np.linspace(min_y, max_y, 100)
    plotter_object.add_gt_function(x_grid, np.squeeze(y_over_grid), "seismic", levels, 3)
    plotter_object.add_datapoints(x_data, "red", 3)
    plotter_object.add_datapoints(np.expand_dims(x_query, axis=0), "green", 3)
    plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu), "seismic", levels, 4)
    plotter_object.add_datapoints(x_data, "red", 4)
    plotter_object.add_datapoints(np.expand_dims(x_query, axis=0), "green", 4)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def safe_bayesian_optimization_1d_plot(
    output_type: OutputType,
    x_grid,
    acq_score,
    mu,
    std,
    safety_thresholds_lower,
    safety_thresholds_upper,
    safety_mask_grid,
    x_data,
    y_data,
    safety_data,
    x_query,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    if output_type == OutputType.SINGLE_OUTPUT:
        n_row = 1 if mu.shape[1] == 1 else 2
        plotter_object = Plotter(1, v_axes=n_row, share_x="all", share_y="all")
        safe_bayesian_optimization_1d_plot_1task(
            plotter_object,
            0, 0,
            x_grid,
            acq_score,
            mu,
            std,
            safety_thresholds_lower,
            safety_thresholds_upper,
            safety_mask_grid,
            x_data,
            y_data,
            safety_data,
            x_query,
        )
    elif output_type == OutputType.MULTI_OUTPUT_FLATTENED:
        p = np.unique(x_data[:, -1])
        n_row = 1 if mu.shape[1] == 1 else 2

        if len(p) == 2:
            plotter_object = Plotter(1, v_axes=n_row, share_x="all", share_y="all")
        else:
            plotter_object = Plotter(len(p), v_axes=n_row, share_x="all", share_y="all")
        for i, p_idx in enumerate(p):
            plug_in_x_query = x_query[..., :-1] if x_query[..., -1] == p_idx else None

            safe_bayesian_optimization_1d_plot_1task(
                plotter_object,
                0 if len(p)==2 else i,
                0,
                x_grid[x_grid[:, -1] == p_idx, :-1],
                acq_score[x_grid[:, -1] == p_idx],
                mu[x_grid[:, -1] == p_idx],
                std[x_grid[:, -1] == p_idx],
                safety_thresholds_lower,
                safety_thresholds_upper,
                safety_mask_grid[x_grid[:, -1] == p_idx],
                x_data[x_data[:, -1] == p_idx, :-1],
                y_data[x_data[:, -1] == p_idx],
                safety_data[x_data[:, -1] == p_idx],
                plug_in_x_query,
                point_color= "black" if p_idx == len(p)-1 else "y"
            )

    elif output_type == OutputType.MULTI_OUTPUT:
        raise EnvironmentError("unfinished work")

    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def safe_bayesian_optimization_1d_plot_1task(
    plotter_object,
    ax_num,
    v_ax_num_init,
    x_grid,
    acq_score,
    mu,
    std,
    safety_thresholds_lower,
    safety_thresholds_upper,
    safety_mask_grid,
    x_data,
    y_data,
    safety_data,
    x_query,
    point_color="black"
):
    plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(y_data), point_color, ax_num, v_ax=v_ax_num_init)
    if x_query is not None:
        plotter_object.add_vline(x_query, "green", ax_num, v_ax=v_ax_num_init)
    if x_grid.shape[0] > 0:
        plotter_object.add_confidence_bound(np.squeeze(x_grid), np.squeeze(mu[:,0]), np.squeeze(std[:,0]), ax_num, v_ax=v_ax_num_init)
        plotter_object.add_acquisition_score(np.squeeze(x_grid), np.squeeze(acq_score), "c", ax_num, v_ax=v_ax_num_init)
    if mu.shape[1] == 1:  # safety constrained directly on the modeling function
        # plot safety on the same plot as above
        plotter_object.add_hline(safety_thresholds_lower[0], "C0", ax_num, v_ax=v_ax_num_init)
        plotter_object.add_hline(safety_thresholds_upper[0], "C0", ax_num, v_ax=v_ax_num_init)
        plotter_object.add_safety_region(x_grid[safety_mask_grid >= 1], ax_num, v_ax=v_ax_num_init)
        plotter_object.add_query_region(x_grid[safety_mask_grid >= 2], ax_num, v_ax=v_ax_num_init)
    elif len(safety_thresholds_lower) == (mu.shape[1] - 1):  # safety constrained on addition functions
        # plot safety on the next row
        plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(safety_data), point_color, ax_num, v_ax=v_ax_num_init+1)
        if x_query is not None:
            plotter_object.add_vline(x_query, "green", ax_num, v_ax=v_ax_num_init+1)
        if x_grid.shape[0] > 0:
            plotter_object.add_multiple_confidence_bound(np.squeeze(x_grid), mu[:,1:], std[:,1:], ax_num, v_ax=v_ax_num_init+1)
            plotter_object.add_multiple_hline(safety_thresholds_lower, ax_num, v_ax=v_ax_num_init+1)
            plotter_object.add_multiple_hline(safety_thresholds_upper, ax_num, v_ax=v_ax_num_init+1)
            plotter_object.add_safety_region(x_grid[safety_mask_grid >= 1], ax_num, v_ax=v_ax_num_init+1)
            plotter_object.add_query_region(x_grid[safety_mask_grid >= 2], ax_num, v_ax=v_ax_num_init+1)


def safe_bayesian_optimization_2d_plot(
    output_type: OutputType,
    x_grid,
    acquisition_values_grid,
    pred_mu_grid,
    y_over_grid,
    safety_mask_grid,
    x_data,
    y_data,
    x_query,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    if output_type == OutputType.SINGLE_OUTPUT:
        plotter_object = Plotter2D(4, share_x="all", share_y="all")
        safe_bayesian_optimization_2d_plot_1task(
            plotter_object, 0, x_grid, acquisition_values_grid, pred_mu_grid, y_over_grid, safety_mask_grid, x_data, y_data, x_query
        )
    elif output_type == OutputType.MULTI_OUTPUT_FLATTENED:
        p = np.unique(x_data[:, -1])
        plotter_object = Plotter2D(4, len(p), share_x="all", share_y="all")
        for i, p_idx in enumerate(p):
            plug_in_x_query = x_query[..., :-1] if x_query[..., -1] == p_idx else None
            safe_bayesian_optimization_2d_plot_1task(
                plotter_object,
                i,
                x_grid[x_grid[:, -1] == p_idx, :-1],
                acquisition_values_grid[x_grid[:, -1] == p_idx],
                pred_mu_grid[x_grid[:, -1] == p_idx],
                y_over_grid[x_grid[:, -1] == p_idx],
                safety_mask_grid[x_grid[:, -1] == p_idx],
                x_data[x_data[:, -1] == p_idx, :-1],
                y_data[x_data[:, -1] == p_idx],
                plug_in_x_query,
            )

    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def safe_bayesian_optimization_2d_plot_1task(
    plotter_object, vax, x_grid, acquisition_values_grid, pred_mu_grid, y_over_grid, safety_mask_grid, x_data, y_data, x_query
):
    if x_grid.shape[0] > 0:
        plotter_object.add_gt_function(x_grid, np.squeeze(acquisition_values_grid), "RdBu_r", 14, 0, vax)
    plotter_object.add_datapoints(x_data, "black", 0, vax)
    if x_query is not None:
        plotter_object.add_datapoints(np.atleast_2d(x_query), "green", 0, vax)
        plotter_object.configure_axes(0, vax, f"{x_data.shape[0]} + 1 points")
    else:
        plotter_object.configure_axes(0, vax, f"{x_data.shape[0]} points")

    if x_grid.shape[0] > 0:
        min_y = np.min(y_over_grid)
        max_y = np.max(y_over_grid)
        levels = np.linspace(min_y - (max_y - min_y) * 0.05, max_y + (max_y - min_y) * 0.05, 100)
        plotter_object.add_gt_function(x_grid, np.squeeze(y_over_grid), "seismic", levels, 1, vax)
    plotter_object.add_datapoints(x_data, "black", 1, vax)
    if x_query is not None:
        plotter_object.add_datapoints(np.atleast_2d(x_query), "green", 1, vax)

    if x_grid.shape[0] > 0:
        plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu_grid), "seismic", levels, 2, vax)
    else:
        min_y = np.min(y_data)
        max_y = np.max(y_data)
        levels = np.linspace(min_y - (max_y - min_y) * 0.05, max_y + (max_y - min_y) * 0.05, 100)
        plotter_object.add_gt_function(x_data, np.squeeze(y_data), "seismic", levels, 2, vax)
    plotter_object.add_datapoints(x_data, "black", 2, vax)
    if x_query is not None:
        plotter_object.add_datapoints(np.atleast_2d(x_query), "green", 2, vax)

    if x_grid.shape[0] > 0:
        levels = np.array([-0.5, 0.5, 1.5, 2.5])
        plotter_object.add_gt_function(x_grid, np.squeeze(safety_mask_grid), "YlGn", levels, 3, vax)
    plotter_object.add_datapoints(x_data, "black", 3, vax)
    plotter_object.add_datapoints(x_query, "green", 3, v_ax=vax)
    if x_query is not None:
        plotter_object.add_datapoints(np.atleast_2d(x_query), "red", 3, vax)


def safe_bayesian_optimization_2d_plot_without_gt(
    output_type: OutputType,
    x_grid,
    acquisition_values_grid,
    pred_mu_grid,
    safety_mask_grid,
    x_data,
    y_data,
    x_query,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    """
    note that x_query have at most 1 raw
    note that we did not set xlim and ylim
    """
    if output_type == OutputType.SINGLE_OUTPUT:
        plotter_object = Plotter2D(3, share_x="all", share_y="all")
        safe_bayesian_optimization_2d_plot_without_gt_1task(
            plotter_object, 0, x_grid, acquisition_values_grid, pred_mu_grid, safety_mask_grid, x_data, y_data, x_query
        )
    elif output_type == OutputType.MULTI_OUTPUT_FLATTENED:
        p = np.unique(x_data[:, -1])
        plotter_object = Plotter2D(3, len(p), share_x="all", share_y="all")
        for i, p_idx in enumerate(p):
            plug_in_x_query = x_query[..., :-1] if x_query[..., -1] == p_idx else None
            safe_bayesian_optimization_2d_plot_without_gt_1task(
                plotter_object,
                i,
                x_grid[x_grid[:, -1] == p_idx, :-1],
                acquisition_values_grid[x_grid[:, -1] == p_idx],
                pred_mu_grid[x_grid[:, -1] == p_idx],
                safety_mask_grid[x_grid[:, -1] == p_idx],
                x_data[x_data[:, -1] == p_idx, :-1],
                y_data[x_data[:, -1] == p_idx],
                plug_in_x_query,
            )

    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def safe_bayesian_optimization_2d_plot_without_gt_1task(
    plotter_object, vax, x_grid, acquisition_values_grid, pred_mu_grid, safety_mask_grid, x_data, y_data, x_query
):
    if x_grid.shape[0] > 0:
        plotter_object.add_gt_function(x_grid, np.squeeze(acquisition_values_grid), "RdBu_r", 14, 0, vax)
    plotter_object.add_datapoints(x_data, "black", 0, vax)
    if x_query is not None:
        plotter_object.add_datapoints(np.atleast_2d(x_query), "green", 0, vax)
        plotter_object.configure_axes(0, vax, f"{x_data.shape[0]} + 1 points")
    else:
        plotter_object.configure_axes(0, vax, f"{x_data.shape[0]} points")

    if x_grid.shape[0] > 0:
        min_y = np.min(pred_mu_grid)
        max_y = np.max(pred_mu_grid)
        levels = np.linspace(min_y - (max_y - min_y) * 0.05, max_y + (max_y - min_y) * 0.05, 100)
        plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu_grid), "seismic", levels, 1, vax)
    else:
        min_y = np.min(y_data)
        max_y = np.max(y_data)
        levels = np.linspace(min_y - (max_y - min_y) * 0.05, max_y + (max_y - min_y) * 0.05, 100)
        plotter_object.add_gt_function(x_data, np.squeeze(y_data), "seismic", levels, 1, vax)
    plotter_object.add_datapoints(x_data, "black", 1, vax)
    if x_query is not None:
        plotter_object.add_datapoints(np.atleast_2d(x_query), "green", 1, vax)

    if x_grid.shape[0] > 0:
        levels = np.array([-0.5, 0.5, 1.5, 2.5])
        plotter_object.add_gt_function(x_grid, np.squeeze(safety_mask_grid), "YlGn", levels, 2, vax)
    plotter_object.add_datapoints(x_data, "black", 2, vax)
    plotter_object.add_datapoints(x_query, "green", 2, v_ax=vax)
    if x_query is not None:
        plotter_object.add_datapoints(np.atleast_2d(x_query), "red", 2, vax)


def safe_bayesian_optimization_2d_plot_without_gt_with_only_safepoints(
    output_type: OutputType,
    x_grid,
    acquisition_values_grid,
    pred_mu_grid,
    safety_mask_grid,
    x_data,
    y_data,
    x_query,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    """
    note that x_query have at most 1 raw
    note that we did not set xlim and ylim
    """
    acq_score = acquisition_values_grid.mean() * np.ones(x_grid.shape[0])
    acq_score[safety_mask_grid.astype(bool)] = acquisition_values_grid

    if output_type == OutputType.SINGLE_OUTPUT:
        plotter_object = Plotter2D(2, share_x="all", share_y="all")
        safe_bayesian_optimization_2d_plot_without_gt_1task_with_only_safepoints(
            plotter_object, 0, x_grid, acq_score, pred_mu_grid, safety_mask_grid, x_data, y_data, x_query
        )
    elif output_type == OutputType.MULTI_OUTPUT_FLATTENED:
        p = np.unique(x_data[:, -1])
        plotter_object = Plotter2D(2, len(p), share_x="all", share_y="all")
        for i, p_idx in enumerate(p):
            plug_in_x_query = x_query[..., :-1] if x_query[..., -1] == p_idx else None

            safe_bayesian_optimization_2d_plot_without_gt_1task_with_only_safepoints(
                plotter_object,
                i,
                x_grid[x_grid[:, -1] == p_idx, :-1],
                acq_score[x_grid[:, -1] == p_idx],
                pred_mu_grid[x_grid[:, -1] == p_idx],
                safety_mask_grid[x_grid[:, -1] == p_idx],
                x_data[x_data[:, -1] == p_idx, :-1],
                y_data[x_data[:, -1] == p_idx],
                plug_in_x_query,
            )

    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def safe_bayesian_optimization_2d_plot_without_gt_1task_with_only_safepoints(
    plotter_object, vax, x_grid, acquisition_values_grid, pred_mu_grid, safety_mask_grid, x_data, y_data, x_query
):
    if x_grid.shape[0] > 0:
        mask = np.where(safety_mask_grid, False, True)
        plotter_object.add_gt_function_with_mask(x_grid, np.squeeze(acquisition_values_grid), mask, "RdBu_r", 14, 0, vax)

    plotter_object.add_datapoints(x_data, "black", 0, vax)
    if x_query is not None:
        plotter_object.add_datapoints(np.atleast_2d(x_query), "green", 0, vax)
        plotter_object.configure_axes(0, vax, f"{x_data.shape[0]} + 1 points")
    else:
        plotter_object.configure_axes(0, vax, f"{x_data.shape[0]} points")

    if x_grid.shape[0] > 0:
        min_y = np.min(pred_mu_grid)
        max_y = np.max(pred_mu_grid)
        levels = np.linspace(min_y - (max_y - min_y) * 0.05, max_y + (max_y - min_y) * 0.05, 100)
        plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu_grid), "seismic", levels, 1, vax)
    else:
        min_y = np.min(y_data)
        max_y = np.max(y_data)
        levels = np.linspace(min_y - (max_y - min_y) * 0.05, max_y + (max_y - min_y) * 0.05, 100)
        plotter_object.add_gt_function(x_data, np.squeeze(y_data), "seismic", levels, 1, vax)
    plotter_object.add_datapoints(x_data, "black", 1, vax)
    if x_query is not None:
        plotter_object.add_datapoints(np.atleast_2d(x_query), "green", 1, vax)

def safety_function_2d_plot(
    output_type: OutputType,
    x_grid, pred_mu_grid, pred_conf_bound_grid,
    safety_thresholds_lower, safety_thresholds_upper,
    x_data, z_data,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    
    J = z_data.shape[1]
    plotter_object = Plotter2D(J, 3, share_x="all", share_y="all")

    if output_type== OutputType.SINGLE_OUTPUT:
        safety_function_2d_1task_plot(
            plotter_object,
            0,
            x_grid, pred_mu_grid, pred_conf_bound_grid,
            safety_thresholds_lower, safety_thresholds_upper,
            x_data, z_data
        )
    elif output_type==OutputType.MULTI_OUTPUT_FLATTENED:
        p = np.max(x_grid[:, -1])
        safety_function_2d_1task_plot(
            plotter_object,
            0,
            x_grid[x_grid[:, -1] == p, :-1], pred_mu_grid[x_grid[:, -1] == p, :], pred_conf_bound_grid[x_grid[:, -1] == p, :],
            safety_thresholds_lower, safety_thresholds_upper,
            x_data[x_data[:, -1] == p, :-1], z_data[z_data[:, -1] == p, :]
        )

    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()

def safety_function_2d_1task_plot(
    plotter_object,
    v_ax,
    x_grid, pred_mu_grid, pred_conf_bound_grid,
    safety_thresholds_lower, safety_thresholds_upper,
    x_data, z_data
):
    J = z_data.shape[1]
    mask = np.logical_and(
        pred_mu_grid - pred_conf_bound_grid >= safety_thresholds_lower,    
        pred_mu_grid + pred_conf_bound_grid <= safety_thresholds_upper
    )
    min_z = min(pred_mu_grid.reshape(-1))
    max_z = max(pred_mu_grid.reshape(-1))
    levels = np.linspace(min_z - (max_z - min_z) * 0.05, max_z + (max_z - min_z) * 0.05, 100)
    levels_safety = np.array([-0.5, 0.5, 1.5, 2.5])
    for model_idx in range(J):
        plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu_grid[..., model_idx]), "seismic", levels, model_idx, v_ax)
        plotter_object.add_datapoints(x_data, "black", model_idx, v_ax)
        plotter_object.configure_axes(model_idx, v_ax, f"model {model_idx} mean, {x_data.shape[0]} points")

        plotter_object.add_gt_function(x_grid, np.squeeze(pred_conf_bound_grid[..., model_idx]), "RdBu_r", 14, model_idx, v_ax+1)
        plotter_object.add_datapoints(x_data, "black", model_idx, v_ax+1)
        plotter_object.configure_axes(model_idx, v_ax+1, f"model {model_idx} std, {x_data.shape[0]} points")

        plotter_object.add_gt_function(x_grid, np.squeeze(mask[:, model_idx]), "YlGn", levels_safety, model_idx, v_ax+2)
        plotter_object.add_datapoints(x_data, "black", model_idx, v_ax=v_ax+2)


def safety_histogram(
    output_type: OutputType, x, z, safe_threshold_lower, safe_threshold_upper, save_plot=False, file_name=None, file_path=None
):
    if output_type == OutputType.SINGLE_OUTPUT:
        safety_dim = z.shape[1]
        plotter_object = HistogramPlotter(safety_dim)
        bins = 30
        for q in range(safety_dim):
            plotter_object.add_histogram(z[:, q], bins, color="blue", label=None, ax_num=q)
            plotter_object.add_vline(safe_threshold_lower[q], color="red", ax_num=q)
            plotter_object.add_vline(safe_threshold_upper[q], color="red", ax_num=q)

    elif output_type == OutputType.MULTI_OUTPUT_FLATTENED:
        p = np.unique(x[:, -1])
        safety_dim = z.shape[1]

        plotter_object = HistogramPlotter(safety_dim, len(p))
        bins = 30
        for i, p_idx in enumerate(p):
            z_p = z[x[:, -1] == p_idx]
            for q in range(safety_dim):
                plotter_object.add_histogram(z_p[:, q], bins, color="blue", label=None, ax_num=q, v_ax=i)
                plotter_object.add_vline(safe_threshold_lower[q], color="red", ax_num=q, v_ax=i)
                plotter_object.add_vline(safe_threshold_upper[q], color="red", ax_num=q, v_ax=i)

    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def plot_matrix_with_text(ax, matrix, x_name=None, y_name=None, title="", x_axis=None, y_axis=None, x_ticks_angle=0):
    if x_name is None:
        x_name = np.arange(np.shape(matrix)[1])
    if y_name is None:
        y_name = np.arange(np.shape(matrix)[0])

    plt.sca(ax)

    for i in range(len(y_name)):
        for j in range(len(x_name)):
            if matrix.dtype == int:
                text = "%d" % (matrix[i, j])
            else:
                text = "%.2f" % (matrix[i, j])
            ax.text(j, i, text, ha="center", va="center", color="w" if matrix[i, j] <= 0.8 else "k")

    ax.imshow(matrix)

    ax.set_xticks(np.arange(len(x_name)))
    ax.set_yticks(np.arange(len(y_name)))
    ax.set_xticklabels(x_name)
    ax.set_yticklabels(y_name)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    plt.setp(ax.get_xticklabels(), rotation=x_ticks_angle, ha="right", rotation_mode="anchor")
    plt.title(title)


def set_font_sizes(font_size, only_axis=True):
    if only_axis:
        matplotlib.rc("xtick", labelsize=font_size)
        matplotlib.rc("ytick", labelsize=font_size)
    else:
        font = {"family": "normal", "size": font_size}

        matplotlib.rc("font", **font)



def create_box_plot_from_dict(
    dictionary: Union[Dict[str, Union[np.array, List[float]]], Dict[str, Dict[str, Union[np.array, List[float]]]]],
    y_name: str,
    key_level1_name: str,
    key_level2_name: Optional[str] = None,
    swap_key: bool = False,
    save_fig: bool = False,
    file_path: str = "",
    file_name: str = "boxplot.png",
    return_fig: bool = False,
    bar_plot: bool = True,
    log_y: bool = False,
    showfliers: bool = True,
):
    num_levels = 1
    if isinstance(dictionary[list(dictionary.keys())[0]], dict):
        num_levels = 2
        second_layer_keys = list(dictionary[list(dictionary.keys())[0]].keys())
    if num_levels == 1:
        data_frame = pd.DataFrame.from_dict(dictionary)
        if bar_plot:
            ax = sns.barplot(data=data_frame, estimator="median", errorbar=("pi", 60))
        else:
            ax = sns.boxplot(data=data_frame, showfliers=showfliers)
        ax.set_ylabel(y_name)
    elif num_levels == 2:
        list_of_dfs = []
        for key in dictionary:
            data_frame = pd.DataFrame.from_dict(dictionary[key])
            for inner_key in dictionary[key]:
                df_inner = data_frame[inner_key].to_frame(y_name).assign(**{key_level1_name: key}).assign(**{key_level2_name: inner_key})
                list_of_dfs.append(df_inner)

        cdf = pd.concat(list_of_dfs)
        print(cdf.head())
        if swap_key:
            if bar_plot:
                ax = sns.barplot(x=key_level2_name, y=y_name, hue=key_level1_name, data=cdf, estimator="median", errorbar=("pi", 60))
            else:
                ax = sns.boxplot(x=key_level2_name, y=y_name, hue=key_level1_name, data=cdf, showfliers=showfliers)
        else:
            if bar_plot:
                ax = sns.barplot(
                    x=key_level1_name, y=y_name, hue=key_level2_name, data=cdf, estimator="median", errorbar=("pi", 60)
                )  # RUN PLOT
            else:
                ax = sns.boxplot(x=key_level1_name, y=y_name, hue=key_level2_name, data=cdf, showfliers=showfliers)
        if log_y:
            ax.set_yscale("log")
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()
    elif return_fig:
        return plt.gcf()
    else:
        plt.show()
        plt.clf()
        plt.close()
    return None


if __name__ == "__main__":
    df_dict = {}
    df_dict["data1"] = {}
    df_dict["data1"]["method1"] = np.random.randn(100)
    df_dict["data1"]["method2"] = np.random.randn(100)
    df_dict["data2"] = {}
    df_dict["data2"]["method1"] = np.random.randn(100)
    df_dict["data2"]["method2"] = np.random.randn(100)
    df_dict["data3"] = {}
    df_dict["data3"]["method1"] = np.random.randn(100)
    df_dict["data3"]["method2"] = np.random.randn(100)
    df_dict["data4"] = {}
    df_dict["data4"]["method1"] = np.random.randn(100)
    df_dict["data4"]["method2"] = np.random.randn(100)
    df_dict["data5"] = {}
    df_dict["data5"]["method1"] = np.random.randn(100)
    df_dict["data5"]["method2"] = np.random.randn(100)
    create_box_plot_from_dict(df_dict, "random", "data", "method", swap_key=True)
