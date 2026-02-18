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

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Union, Sequence
from functools import partial
from scipy import interpolate
from gpflow.utilities import tabulate_module_summary, leaf_components

from alef.utils.utils import create_grid
from alef.oracles.base_oracle import Standard2DOracle

class MOGP2DOracle(Standard2DOracle):
    def __init__(
        self,
        kernel_list,
        W: np.ndarray,
        observation_noise: float,
        specified_output: int
    ):
        if not kernel_list is None:
            assert np.all([kern.input_dimension == 2 for kern in kernel_list])
        super().__init__(observation_noise, -np.inf, np.inf)
        self.observation_noise = observation_noise
        self.W = W
        self.kernel_list = kernel_list
        self.__specified_output = specified_output
    def set_specified_output(self, specified_output:int):
        self.__specified_output = specified_output

    def draw_from_hyperparameter_prior(self):
        print("-Draw from hyperparameter prior")
        for i, kernel in enumerate(self.kernel_list):
            for parameter in kernel.trainable_parameters:
                # old_value=parameter.numpy()
                # print("Old value")
                # print(old_value)
                print(parameter)
                new_value = parameter.prior.sample()
                parameter.assign(new_value)
            print(f'Latent kernel {i}')
            print_summary(kernel)

    def initialize(self, a, b, n, normalized_output: bool=False):
        self._export_kernel()
        obs_noise = self.observation_noise
        super().__init__(obs_noise, a, b)
        grid, function_values = self._gp_sample(a, b, n, normalized_output=normalized_output)
        self.__f_list = [
            interpolate.interp2d(grid[:, 0], grid[:, 1], fvs, kind="linear") for fvs in function_values.T
        ]
    
    def initialize_from_txt(self, file_path, output_name='function_values.txt'):
        obs_noise = self.observation_noise
        a, b = np.loadtxt(os.path.join(file_path, 'box_bounds.txt'))
        super().__init__(obs_noise, a, b)

        grid = np.loadtxt(os.path.join(file_path, 'grid.txt'))
        function_values = np.loadtxt(os.path.join(file_path, output_name))
        self.__f_list = [
            interpolate.interp2d(grid[:, 0], grid[:, 1], fvs, kind="linear") for fvs in function_values.T
        ]
    
    def save_gp_initialization_to_txt(self, a, b, n, file_path, output_name='function_values.txt', normalized_output: bool=False):
        self._export_kernel(path=os.path.join(file_path, 'kernel_documentation.json'))
        
        grid, function_values = self._gp_sample(a, b, n, normalized_output=normalized_output)

        np.savetxt(os.path.join(file_path, 'box_bounds.txt'), [a, b])
        np.savetxt(os.path.join(file_path, 'grid.txt'), grid)
        np.savetxt(os.path.join(file_path, output_name), function_values)

    def _export_kernel(self, path: str=None):
        if path is None:
            print(f'W: {self.W}\n')
            print('\nlatent kernels')
            for LatKern in self.kernel_list:
                #print(f'name: {LatKern.__class__.__name__}\n')
                print(tabulate_module_summary(LatKern, None))
        else:
            if path.endswith('.txt'):
                with open(path, mode='w') as fp:
                    print(f'W: {self.W}\n', file=fp)
                    print('\nlatent kernels', file=fp)
                    for LatKern in self.kernel_list:
                        #print(f'name: {LatKern.__class__.__name__}\n', file=fp)
                        print(tabulate_module_summary(LatKern, None), file=fp)
            elif path.endswith('.json'):
                kernel = {
                    'W': self.W.tolist(),
                    'latent_k': {
                        i:{
                            var_name: var_value.numpy().tolist() for var_name, var_value in leaf_components(k).items()
                        } for i, k in enumerate(self.kernel_list)
                    }
                }
                with open(path, 'w') as f:
                    json.dump(kernel, f)

    def _gp_sample(self, a, b, n, normalized_output: bool=False): 
        """
        return grid (inputs), gp samples of the grid
        """
        grid = create_grid(a, b, n, self.get_dimension())
        K_list = [kern(grid, grid).numpy() for kern in self.kernel_list]
        n = grid.shape[0]
        L, P, R = self.W.shape
        
        Chol_stack = [np.kron(
            np.linalg.cholesky( self.W[l, :, :] @ self.W[l,:,:].T ),
            np.linalg.cholesky( K_list[l] )
            ) for l in range(L)] # L elements, each [Pn, Pn]
        
        samples = np.random.standard_normal(size=[P*n, L])
        F_latent = np.concatenate(
            [ np.reshape(Chol_stack[l] @ samples[:, l, None], [P, n, 1]) for l in range(L) ],
            axis=-1
        ) # [P, n, L]
        F = F_latent.sum(axis=-1)
        F = F.T
        
        if normalized_output:
            return grid, (F - F.mean(axis=0))/ np.sqrt(F.var(axis=0) )
        else:
            return grid, F        

    def f(self, *args, **kwargs):
        p = self.__specified_output
        if p >= 0:
            return self.__f_list[p](*args, **kwargs)
        else:
            return np.concatenate([fi(*args, **kwargs) for fi in self.__f_list])

    def query(self, x, noisy=True):
        p = self.__specified_output
        if p >=0:
            function_value = np.squeeze(self.f(x[0], x[1]))
            if noisy:
                epsilon = np.random.normal(0, self.observation_noise, 1)[0]
                function_value += epsilon
        else:
            function_value = np.reshape(self.f(x[0], x[1]), -1)
            if noisy:
                epsilon = np.random.normal(0, self.observation_noise, len(function_value))
                function_value += np.reshape(epsilon, -1)
        return function_value
    
    def _plot_3D(self):
        xs, ys = self.get_random_data(2000, noisy=False)
        
        P = ys.shape[1]

        fig = plt.figure()
        for p in range(P):
            ax = fig.add_subplot(1, P, p+1, projection="3d")
            ax.scatter(xs[:, 0], xs[:, 1], ys[:,p], marker=".")
        return fig

    def _plot_color_map(self, safe_lower_bound: float=0.0, safe_upper_bound: float=np.inf):
        xs, ys = self.get_random_data(2000, noisy=False)
        """
        notice that if self.__specified_output < 0, this plot will only plot f0
        """
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(1,2,1)
        contour = ax.tricontourf(xs[:,0],xs[:,1],ys[:,0], levels=np.linspace(-2, 2, 50), cmap='seismic')
        fig.colorbar(contour, ax=ax)
        ax = fig.add_subplot(1,2,2)
        contour = ax.tricontourf(xs[:,0],xs[:,1],(ys[:,0] >= safe_lower_bound) * (ys[:,0] <= safe_upper_bound), levels=[-0.5, 0.5, 1.5], cmap='YlGn')
        fig.colorbar(contour, ax=ax)
        return fig


if __name__ == "__main__":
    import gpflow
    from gpflow import kernels
    from alef.utils.plotter2D import Plotter2D
    from alef.kernels.kernel_factory import KernelFactory
    
    from alef.configs.kernels.linear_configs import LinearWithPriorConfig
    from alef.configs.kernels.periodic_configs import PeriodicWithPriorConfig
    from alef.configs.kernels.rbf_configs import RBFWithPriorConfig
    from alef.configs.kernels.additive_kernel_configs import BasicAdditiveKernelConfig, AdditiveKernelWithPriorConfig
    from alef.kernels.kernel_grammar.kernel_grammar import ElementaryKernelGrammarExpression, KernelGrammarExpression, KernelGrammarOperator
    from alef.kernels.periodic_kernel import PeriodicKernel
    from alef.kernels.hierarchical_hyperplane_kernel import HierarchicalHyperplaneKernel as HyperplaneDividedKernelGeneral
    from alef.kernels.warped_multi_index_kernel import WarpedMultiIndexKernel
    from alef.kernels.additive_kernel import Partition
    
    from gpflow.utilities import print_summary, set_trainable, to_default_float
    gpflow.config.set_default_float(np.float64)
    gpflow.config.set_default_jitter(1e-4)

    f64 = gpflow.utilities.to_default_float


    base_expression_1 = ElementaryKernelGrammarExpression(KernelFactory.build(RBFWithPriorConfig(input_dimension=2)))
    base_expression_2 = ElementaryKernelGrammarExpression(KernelFactory.build(LinearWithPriorConfig(input_dimension=2)))
    base_expression_3 = ElementaryKernelGrammarExpression(KernelFactory.build(LinearWithPriorConfig(input_dimension=2)))
    base_expression_4 = ElementaryKernelGrammarExpression(KernelFactory.build(RBFWithPriorConfig(input_dimension=2)))
    expression1 = KernelGrammarExpression(base_expression_1, base_expression_2, operator=KernelGrammarOperator.SPLIT_CH)
    expression2 = KernelGrammarExpression(base_expression_3, base_expression_4, operator=KernelGrammarOperator.SPLIT_CH)
    expression3 = KernelGrammarExpression(expression1, expression2, operator=KernelGrammarOperator.SPLIT_CH)

    for i in range(0, 10):

        # partition = Partition(2)
        # partition.add_partition_element([0])
        # partition.add_partition_element([1])
        # kernel_config = AdditiveKernelWithPriorConfig(input_dimension=2,partition=partition)
        # kernel = KernelFactory().build(kernel_config)
        # kernel = (gpflow.kernels.Matern32()+gpflow.kernels.Linear())*gpflow.kernels.Matern32(lengthscales=[0.5,2.0])
        # kernel = gpflow.kernels.periodic.Periodic(gpflow.kernels.Matern32(),period=3)
        # kernel_config = DeepKernelWithPriorConfig(input_dimension=2,layer_size_list=[3,2])
        # kernel = KernelFactory.build(kernel_config)
        # kernel = gpflow.kernels.RBF()
        # kernel_config = FullyDependendInvertibleResnetKernelConfig(input_dimension=2)
        # kernel_config = PeriodicWithPriorConfig(input_dimension=2)
        kernel_list = [expression3.get_kernel() for _ in range(3)]
        gp_oracle = MOGP2DOracle(kernel_list, np.array([[0.9, 0.1, 0.1], [-0.1, 0.9, 0.1]]), 0.01, -1)
        gp_oracle.draw_from_hyperparameter_prior()
        gp_oracle.initialize(-2, 2, 50)

        X, f = gp_oracle.get_random_data(1000, noisy=False)
        print(X.shape)
        print(f.shape)
        f_0 = gp_oracle.query([0.0, 0.0])
        print(f_0)

        #gp_oracle.plot()
        plotter_object = Plotter2D(2)
        plotter_object.add_gt_function(X, np.squeeze(f[:,0]), "seismic", 30, 0)
        plotter_object.add_gt_function(X, np.squeeze(f[:,1]), "seismic", 30, 1)
        plotter_object.show()
