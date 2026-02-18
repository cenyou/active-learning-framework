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

from datetime import datetime
import pickle
from typing import Tuple, Union, Sequence, List
import json
import os
from typing import List, Union
import numpy as np
from scipy import integrate
from scipy.stats import norm
from sklearn.cluster import KMeans
import math


def gmm_density(y, mus, sigmas, weights):
    normalized_weights = weights / np.sum(weights)
    density = 0.0
    for i in range(0, mus.shape[0]):
        density += normalized_weights[i] * norm.pdf(y, mus[i], sigmas[i])
    return density


def gmm_entropy_integrand(y, mus, sigmas, weights):
    p = gmm_density(y, mus, sigmas, weights)
    # print(p)
    if math.isclose(p, 0.0):
        return 0.0
    else:
        return -1 * p * np.log(p)


def entropy_of_gmm(mus, sigmas, weights, uniform_weights):
    if weights is None or uniform_weights:
        weights = np.repeat(1.0, mus.shape[0])
    f = lambda y: gmm_entropy_integrand(y, mus, sigmas, weights)
    int_f = integrate.quad(f, -np.infty, np.infty)
    return int_f[0]


def calculate_multioutput_rmse(pred_y, y):
    M = pred_y.shape[1]
    assert y.shape[1] == M
    rmses = []
    for m in range(0, M):
        rmse = np.sqrt(np.mean(np.power(pred_y[:, m] - np.squeeze(y[:, m]), 2.0)))
        rmses.append(rmse)
    return np.array(rmses)


def create_grid(a, b, n_per_dim, dimensions):
    grid_points = np.linspace(a, b, n_per_dim)
    n = int(np.power(n_per_dim, dimensions))
    X = np.zeros((n, dimensions))
    for i in range(0, dimensions):
        repeats_per_item = int(np.power(n_per_dim, i))
        block_size = repeats_per_item * n_per_dim
        block_repeats = int(n / block_size)
        for block in range(0, block_repeats):
            for j in range(0, n_per_dim):
                point = grid_points[j]
                for l in range(0, repeats_per_item):
                    index = block * block_size + j * repeats_per_item + l
                    X[index, i] = point
    return X


def create_grid_multi_bounds(a: Sequence[float], b: Sequence[float], n_per_dim: Sequence[float]):
    aa = np.reshape(a, -1)
    bb = np.reshape(b, -1)
    nn = np.reshape(n_per_dim, -1)
    assert len(aa) == len(bb)
    assert len(bb) == len(nn)

    n = int(np.prod(nn))
    dimensions = len(nn)
    X = np.zeros((n, dimensions))

    for i in range(dimensions):
        grids = np.linspace(aa[i], bb[i], nn[i])
        n_repeat = np.prod(nn[:i])  # np.prod([]) is 1
        n_tile = np.prod(nn[i + 1 :])

        X[:, i] = np.tile(np.repeat(grids, n_repeat), n_tile)
    return X


def filter_safety(X, y, safety_threshold, safety_is_upper_bound):
    if safety_is_upper_bound:
        safe_indexes = np.squeeze(y) < safety_threshold
    else:
        safe_indexes = np.squeeze(y) > safety_threshold
    return X[safe_indexes], y[safe_indexes]


def filter_nan(X, y):
    r"""
    get input data pair X, y and return data pair without nan values
    input:
        X [N, D] array
        y [N, 1] array
    output:
        X [M, D] array, M <= N
        y [M, 1] array
    """
    mask = ~np.isnan(y).reshape(-1)
    return np.atleast_2d(X)[mask], np.atleast_2d(y)[mask]


def one_fold_cross_validation(model, x_data, y_data, only_use_subset=False, subset_indexes=[]):
    n = len(x_data)
    true_ys = []
    pred_ys = []
    if only_use_subset:
        val_indexes = subset_indexes
    else:
        val_indexes = list(range(0, n))
    for val_index in val_indexes:
        train_indexes = list(range(0, n))
        train_indexes.pop(val_index)
        train_data_x = x_data[train_indexes]
        train_data_y = y_data[train_indexes]
        test_point_x = np.expand_dims(x_data[val_index], axis=0)
        test_point_y = y_data[val_index]
        true_ys.append(test_point_y)
        model.reset_model()
        print(train_data_x.shape)
        print(train_data_y.shape)
        model.infer(train_data_x, train_data_y)
        predicted_y, _ = model.predictive_dist(test_point_x)
        pred_ys.append(predicted_y)
    true_ys = np.array(true_ys)
    pred_ys = np.array(pred_ys)
    rmse = np.sqrt(np.mean(np.power(pred_ys - np.squeeze(true_ys), 2.0)))
    return rmse


def calculate_rmse(y_pred, y_true):
    return np.sqrt(np.mean(np.power(np.squeeze(y_pred) - np.squeeze(y_true), 2.0)))


def calculate_nll_normal(y_test, pred_mus, pred_sigmas):
    return -1 * norm.logpdf(np.squeeze(y_test), np.squeeze(pred_mus), np.squeeze(pred_sigmas))


def extract_grid_lines(grid, point):
    dimensions = grid.shape[1]
    lines = []
    for i in range(0, dimensions):
        grid_buffer = grid.copy()
        for j in range(0, dimensions):
            if i != j:
                grid_buffer = grid_buffer[grid_buffer[:, j] == point[j]]
        lines.append(grid_buffer)

    return lines


def normal_entropy(sigma):
    entropy = np.log(sigma * np.sqrt(2 * np.pi * np.exp(1)))
    return entropy


def string2bool(b):
    if isinstance(b, bool):
        return b
    if b.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif b.lower() in ("no", "false", "f", "n", "0"):
        return False


def normalize_data(x: np.array):
    assert len(x.shape) == 2
    x_normalized = (x - np.expand_dims(np.mean(x, axis=0), axis=0)) / np.expand_dims(np.std(x, axis=0), axis=0)
    return x_normalized


def min_max_normalize_data(x: np.array):
    assert len(x.shape) == 2
    x_normalized = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    return x_normalized


def scale_data(x: np.array):
    assert len(x.shape) == 1
    x_scaled = (x - np.min(x)) / np.max(x - np.min(x))
    return x_scaled


def k_means(num_clusters: int, x_data: np.array):
    assert len(x_data.shape) == 2
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto').fit(x_data)
    return kmeans.cluster_centers_


def twod_array_to_list_over_arrays(array):
    list_over_arrays = [array[i, :] for i in range(0, array.shape[0])]
    return list_over_arrays


def draw_from_hp_prior_and_assign(kernel):
    print("-Draw from hyperparameter prior")
    for parameter in kernel.trainable_parameters:
        new_value = parameter.prior.sample()
        parameter.assign(new_value)


def row_wise_compare(x, y):
    r"""

    :param x: [N1, D] array
    :param y: [N2, D] array
    :return: [N1, ] boolean array, True if the row in x is contained in y
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    d1 = x.shape[1]
    d2 = y.shape[1]  # if d2 != d1, the result should be all False, so no need to waste time on checking

    struct_x = x.view(x.dtype.descr * d1)
    struct_y = y.view(y.dtype.descr * d2)

    return np.in1d(struct_x, struct_y)


def row_wise_unique(x):
    r"""
    :param x: [N1, D] array
    :return:
        output [N, D] array, N <= N1, the unique rows of x (sorted)
        idx    [N, ] array, idx, output = x[idx]
    """
    x = np.atleast_2d(x)

    d = x.shape[1]

    struct_x = x.view(x.dtype.descr * d)
    _, idx = np.unique(struct_x, return_index=True)

    return x[idx], idx


def check1Dlist(variables, dimension: int):
    r"""
    variables: a value or an array
    dimension: integer

    return a 1D list of 'dimension' num of element, could be duplicate of variables if it is a value or variables if it is of len dimension
    """
    if hasattr(variables, "__len__"):
        output = np.reshape(variables, -1).tolist()
        if len(output) == 1:  # duplicate to d elements list
            output = output * dimension
        assert len(output) == dimension
    else:
        output = [variables] * dimension
    return output


def write_list_to_file(alist: List[Union[float, int, str]], file_path: str):
    with open(file_path, "w") as f:
        for line in alist:
            f.write(f"{line}\n")


def write_dict_to_json(dictionary, file_path: str):
    with open(file_path, "w") as f:
        json.dump(dictionary, f)


def pickle_object(obj, file_path: str):
    with open(file_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickled_object(file_path: str):
    with open(file_path, "rb") as handle:
        obj = pickle.load(handle)
    return obj


def read_list_from_file(file_path: str):
    with open(file_path, "r") as f:
        alist = f.read().splitlines()
    return alist


def read_dict_from_json(file_path: str):
    with open(file_path, "r") as fp:
        dictionary = json.load(fp)
    return dictionary


def get_datetime_as_string():
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    return date_time


if __name__ == "__main__":
    print(get_datetime_as_string())
