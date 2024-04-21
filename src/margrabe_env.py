from typing import Tuple

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from jax import vmap
from jaxfin.models.gbm import MultiGeometricBrownianMotion

from .base.spread_env import SpreadHedgingEnvBase
from .math import margrabe, margrabe_deltas


def flatten(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs).flatten()

    return wrapper


class MargrabeEnvBase(SpreadHedgingEnvBase):
    metadata = {"render.modes": ["human"]}
    action_space: spaces.Space
    observation_space: spaces.Space

    def __init__(
        self,
        s1_0: float,
        s2_0: float,
        expiry: float,
        r: float,
        mu_1: float,
        mu_2: float,
        sigma_1: float,
        sigma_2: float,
        corr: float,
        n_steps: int,
        tick_size: float = 0.01,
    ):
        super().__init__(
            s1_0=s1_0,
            s2_0=s2_0,
            strike=0.0,
            expiry=expiry,
            r=r,
            mu_1=mu_1,
            mu_2=mu_2,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            corr=corr,
            n_steps=n_steps,
            tick_size=tick_size,
        )

    def _generate_stock_path(self) -> Tuple[np.ndarray, np.ndarray]:
        s0 = jnp.array([self.s1_0, self.s2_0])
        means = jnp.array([self.mu_1, self.mu_2])
        vols = jnp.array([self.sigma_1, self.sigma_2])
        corr_matrix = jnp.array([[1, self.corr], [self.corr, 1]])
        gbm = MultiGeometricBrownianMotion(s0, means, vols, corr_matrix)

        stock_paths = gbm.sample_paths(self.expiry, self.n_steps, 1)

        return np.asarray(stock_paths[:, :, 0]), np.asarray(stock_paths[:, :, 1])

    def _get_current_stock_vols(self, step: int) -> Tuple[float, float]:
        return self.sigma_1, self.sigma_2

    @flatten
    def _get_spread_prices(self) -> np.ndarray:
        return np.asarray(
            [
                margrabe(
                    self._stock_path_2[i],
                    self._stock_path_1[i],
                    self.expiry - i * self.dt,
                    self.sigma_1,
                    self.sigma_2,
                    self.corr,
                )
                for i in range(self.n_steps)
            ]
        )

    def _get_deltas(self) -> Tuple[np.ndarray, np.ndarray]:
        deltas = np.asarray(
            [
                margrabe_deltas(
                    self._stock_path_2[i, 0],
                    self._stock_path_1[i, 0],
                    self.expiry - i * self.dt,
                    self.sigma_1,
                    self.sigma_2,
                    self.corr,
                )
                for i in range(self.n_steps)
            ]
        )
        print(deltas.shape)
        return deltas[:, 0], deltas[:, 1]


class MargrabeEnvCont(MargrabeEnvBase):

    def __init__(
        self,
        s1_0: float,
        s2_0: float,
        expiry: float,
        r: float,
        mu_1: float,
        mu_2: float,
        sigma_1: float,
        sigma_2: float,
        corr: float,
        n_steps: int,
        tick_size: float = 0.01,
    ):
        super().__init__(
            s1_0=s1_0,
            s2_0=s2_0,
            expiry=expiry,
            r=r,
            mu_1=mu_1,
            mu_2=mu_2,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            corr=corr,
            n_steps=n_steps,
            tick_size=tick_size,
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        self.observation_space = Box(
            low=np.array(
                [-np.inf, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32
            ),
            high=np.array(
                [np.inf, 2.0, 2.0, np.inf, 1.0, 1.0, np.inf, 1.0, 1.0], dtype=np.float32
            ),
        )
