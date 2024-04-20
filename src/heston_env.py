import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from jax import vmap
from jaxfin.models.heston import UnivHestonModel
from jaxfin.price_engine.fft import delta_call_fourier, fourier_inv_call

from .base.call_env import HedgingEnvBase

v_delta_call_fourier = vmap(
    delta_call_fourier, in_axes=(0, None, None, None, None, None, None, None, None)
)
v_fouirer_inv_call = vmap(
    fourier_inv_call, in_axes=(0, None, None, None, None, None, None, None, None)
)


def flatten(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs).flatten()

    return wrapper


class HestonEnvBase(HedgingEnvBase):
    metadata = {"render.modes": ["human"]}
    action_space: spaces.Space
    observation_space: spaces.Space

    def __init__(
        self,
        s0: float,
        strike: float,
        expiry: float,
        r: float,
        mu: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        n_steps: int,
        ticksize: float = 0.01,
    ):
        super().__init__(
            s0=s0,
            strike=strike,
            expiry=expiry,
            r=r,
            mu=mu,
            sigma=theta,
            n_steps=n_steps,
            ticksize=ticksize,
        )
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.vol_of_vol = sigma
        self._variance_process = np.asarray([])

    def _generate_stock_path(self) -> np.ndarray:
        heston_process = UnivHestonModel(
            s0=self.s0,
            v0=self.v0,
            mean=self.mu,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.vol_of_vol,
            rho=self.rho,
        )

        paths, variance_p = heston_process.sample_paths(self.expiry, self.n_steps, 1)
        paths = np.asarray(paths)
        variance_p = np.asarray(variance_p).squeeze()
        self._variance_process = variance_p

        return paths

    def _get_current_stock_vol(self, step: int) -> float:
        return np.sqrt(self._variance_process[step])

    @flatten
    def _get_call_prices(self) -> np.ndarray:
        return np.asarray(
            [
                fourier_inv_call(
                    s0=self._stock_path[i],
                    K=self.strike,
                    T=self.expiry - i * self.dt,
                    v0=self.v0,
                    mu=self.mu,
                    kappa=self.kappa,
                    theta=self.theta,
                    sigma=self.vol_of_vol,
                    rho=self.rho,
                )
                for i in range(self.n_steps)
            ]
        )

    @flatten
    def _get_deltas(self) -> np.ndarray:
        return np.asarray(
            [
                v_delta_call_fourier(
                    self._stock_path[i],
                    self.strike,
                    self.expiry - i * self.dt,
                    self.v0,
                    self.mu,
                    self.theta,
                    self.vol_of_vol,
                    self.kappa,
                    self.rho,
                )
                for i in range(self.n_steps)
            ]
        )


class HestonEnvCont(HestonEnvBase):

    def __init__(
        self,
        s0: float,
        strike: float,
        expiry: float,
        r: float,
        mu: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        n_steps: int,
        ticksize: float = 0.01,
    ):
        super().__init__(
            s0=s0,
            strike=strike,
            expiry=expiry,
            r=r,
            mu=mu,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            n_steps=n_steps,
            ticksize=ticksize,
        )
        self.action_space = Box(low=-1, high=0.0, shape=(1,))
        self.observation_space = Box(
            low=np.array([-np.inf, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([np.inf, 2.0, np.inf, 1.0, np.inf, 1.0], dtype=np.float32),
            shape=(6,),
        )

    def step(self, action: np.ndarray):
        new_hedge = action[0]
        return super().step(new_hedge)


class HestonEnvDis(HestonEnvBase):

    def __init__(
        self,
        s0: float,
        strike: float,
        expiry: float,
        r: float,
        mu: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        n_steps: int,
        ticksize: float = 0.01,
    ):
        super().__init__(
            s0=s0,
            strike=strike,
            expiry=expiry,
            r=r,
            mu=mu,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            n_steps=n_steps,
            ticksize=ticksize,
        )
        self.action_space = Discrete(100)
        self.observation_space = Box(
            low=np.array([-np.inf, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([np.inf, 2.0, np.inf, 1.0, np.inf, 1.0], dtype=np.float32),
            shape=(6,),
        )

    def step(self, action: float):
        return super().step(-(action / 100))
