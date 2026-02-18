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
import os
import logging
import gym
from typing import List, Union
from alef.oracles.base_oracle import StandardOracle
from alef.oracles.base_constrained_oracle import StandardConstrainedOracle
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from alef.utils.custom_logging import getLogger
logger = getLogger(__name__)

try:
    from gym_brt.envs import QubeSwingupEnv
    from gym_brt.control import flip_and_hold_policy
    USE_GYM_BRT = True
except:
    logger.warning('Cannot import gym_brt, use gym instead')
    USE_GYM_BRT = False

MAX_STEPS = 1000
FREQUENCY = 250

class SwingUp(StandardOracle):
    def __init__(
        self,
        observation_noise: float=0.1,
    ):
        """
        :param observation_noise: standard deviation of the observation noise (Gaussian noise)
        """
        D = 2
        super().__init__(observation_noise, 0.0, 1.0, D)
        if USE_GYM_BRT:
            self._env = QubeSwingupEnv(use_simulator=True, frequency=FREQUENCY, batch_size=MAX_STEPS)
        else:
            self._env = gym.make('Pendulum-v1', g=9.81)

    def x_scale(self, x: np.ndarray):
        r"""
        rescale x=[x1, x2] as if we are considering x in
            [-6, 0] x
            [-6, 0]
        """
        assert x.shape[-1] == self.get_dimension(), x.shape
        desire_bound = np.array([
            [-6, 0],
            [-6, 0]
        ])
        a, b = self.get_box_bounds()
        return (x - a) / (b - a) * (desire_bound[:,1] - desire_bound[:,0]) + desire_bound[:,0]

    def cartesian2angle(self, state):
        x, y, alpha = state
        if x == 0:
            theta = 0.0
        else:
            theta = - np.arcsin(x) if y >= 0 else np.pi + np.arcsin(x)
        if theta > np.pi:
            theta -= 2 * np.pi
        return np.array([theta, alpha])

    def controller(self, x:np.ndarray, state):
        if USE_GYM_BRT:
            return flip_and_hold_policy(state, x[0], x[1])
        else:
            theta, alpha = self.cartesian2angle(state)
            u = x[0] * theta + x[1] * alpha
            return u

    def _run_gym_brt_episode(self, x, *, noisy:bool):
        #state = self._env.reset_in_state(np.array([0, 0.1, 0.2, 0.], dtype=np.float64))
        state = self._env.reset()
        thetas = []
        alphas = []
        alphas_dot = []
        thetas_dot = []
        actions = []
        step = 0
        terminated = False
        while not terminated and step <= MAX_STEPS:
            u, using_pd_controller = self.controller(self.x_scale(x), state)
            state, reward, terminated, info = self._env.step(u)
            theta, alpha, alpha_dot, theta_dot = state
            thetas.append(theta)
            alphas.append(alpha)
            alphas_dot.append(alpha_dot)
            thetas_dot.append(theta_dot)
            assert False, u
            actions.append(u)
            step += 1
        thetas = np.array(thetas)
        alphas = np.array(alphas)
        actions = np.array(actions)
        if noisy:
            return (
                thetas + np.random.normal(0, self.observation_noise, [step]),
                alphas + np.random.normal(0, self.observation_noise, [step]),
                actions
            )
        else:
            return (thetas, alphas, actions)

    def _run_gym_episode(self, x, *, noisy:bool):
        state, info = self._env.reset()
        thetas = []
        alphas = []
        actions = []
        step = 0
        terminated = False
        truncated = False
        while not terminated and not truncated and step <= MAX_STEPS:
            u = self.controller(self.x_scale(x), state)
            state, reward, terminated, truncated, info = self._env.step([u])
            theta, alpha = self.cartesian2angle(state)
            thetas.append(theta)
            alphas.append(alpha)
            actions.append(u)
            step += 1
        thetas = np.array(thetas)
        alphas = np.array(alphas)
        actions = np.array(actions)
        if noisy:
            return (
                thetas + np.random.normal(0, self.observation_noise, [step]),
                alphas + np.random.normal(0, self.observation_noise, [step]),
                actions
            )
        else:
            return (thetas, alphas, actions)

    def run_episode(self, *args, **kwargs):
        if USE_GYM_BRT:
            return self._run_gym_brt_episode(*args, **kwargs)
        else:
            return self._run_gym_episode(*args, **kwargs)

    def query(self, x, noisy=True):
        thetas, alphas, actions = self.run_episode(x, noisy=noisy)
        return self.loss(thetas, alphas, actions)

    def loss(self, thetas, alphas, actions):
        loss = np.mean(
            1 - np.exp( - np.abs(thetas) )
        ) # [0, 1]
        return loss

    def c(self, thetas, alphas, actions):
        max_alpha = np.max(alphas)
        return 1 - max_alpha / np.pi


class SwingUpConstrained(StandardConstrainedOracle):
    def __init__(
        self,
        observation_noise: Union[float, List[float]]=0.1
    ):
        """
        :param observation_noise: standard deviation of the observation noise (Gaussian noise)
            specify all functions by passing a float or specify each individually by passing a list of 2 floats
        """
        if hasattr(observation_noise, '__len__'):
            if len(observation_noise) == 1:
                observation_noise = [observation_noise[0]] * 2
            elif len(observation_noise) == 2:
                assert np.isclose(*observation_noise)
            else:
                assert False, f'passing incorrect number of observation_noise to {self.__class__.__name__}.__init__ method'
        else:
            observation_noise = [observation_noise] * 2

        main = SwingUp(observation_noise[0])
        super().__init__(main, main)

    def query(self, x, noisy: bool = True):
        thetas, alphas, actions = self.oracle.run_episode(x, noisy=noisy)
        loss = self.oracle.loss(thetas, alphas, actions)
        c = self.oracle.c(thetas, alphas, actions)
        return loss, c[None]

    def batch_query(self, X, noisy: bool = True):
        Y = []
        Z = []
        for i in range(X.shape[-2]):
            y, z = self.query(X[..., i,:], noisy)
            Y.append(y)
            Z.append(z.reshape(-1, 1))
        return np.array(Y), np.vstack(Z)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=2)
    np.random.seed(12345)
    oracle = SwingUpConstrained(0.001)

    N = 2000
    X, Y, Z = oracle.get_random_data(N, noisy=True)
    print('X:', X.mean(), X.var())
    print('Y:', Y.mean(), Y.var())
    print('Z:', Z.mean(), Z.var())
    fig, ax = plt.subplots(1,1)
    ax.tricontourf(
        np.squeeze(X[:, 0]), np.squeeze(X[:, 1]), np.squeeze(Z>=0), cmap="YlGn", levels = np.array([-0.5, 0.5, 1.5, 2.5])
    )
    plt.show()
