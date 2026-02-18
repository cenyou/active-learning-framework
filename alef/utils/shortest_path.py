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

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial import distance_matrix, distance

def shortest_path(X: np.ndarray):
    """
    X: [N, D] array
    
    return:
    int: length,
    [N,] array: idices of the found path
    
    X[returned_output] is a path with shortest dist. under L2 norm
    this is just a greedy approximation
    """
    assert X.ndim == 2
    N = X.shape[0]
    dist = distance.cdist(X, X, 'euclidean')
    dist = pd.DataFrame(dist)
    # start path from a point farthest to all others
    path_points = [dist.mean(axis=0).argmax()]
    path_length = 0
    for _ in range(N - 1):
        closest_dist = dist.loc[path_points[-1], ~dist.index.isin(path_points)].min()
        closest_idx = dist.loc[path_points[-1], ~dist.index.isin(path_points)].idxmin()
        path_points.append(closest_idx)
        path_length += closest_dist
    return path_length, path_points

if __name__=='__main__':
    from matplotlib import pyplot as plt
    X = np.random.standard_normal([100, 2])
    l, idx = shortest_path(X)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(X[:,0], X[:,1], 'o', color='blue')
    ax.plot(X[idx,0], X[idx,1], '-', color='b')
    ax.set_title(f'path length: {l}')
    plt.show()
    