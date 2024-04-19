import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from jax import vmap
from jaxfin.models.gbm import UnivGeometricBrownianMotion
from jaxfin.price_engine.black_scholes import delta_european, european_price

from .base import HedgingEnvBase

v_delta_european = vmap(delta_european, in_axes=(0, None, None, None, None))


def flatten(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs).flatten()

    return wrapper


class BlackScholesEnvBase(HedgingEnvBase):
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
        sigma: float,
        n_steps: int,
    ):
        super().__init__(
            s0=s0,
            strike=strike,
            expiry=expiry,
            r=r,
            mu=mu,
            sigma=sigma,
            n_steps=n_steps,
        )

    def _generate_stock_path(self, seed=None) -> np.ndarray:
        if seed:
            seed = 0

        gbm = UnivGeometricBrownianMotion(self.s0, self.mu, self.sigma)

        return np.asarray(gbm.sample_paths(self.expiry, self.n_steps, 1))

    def _get_current_stock_vol(self, step: int) -> float:
        return self.sigma

    @flatten
    def _get_call_prices(self) -> np.ndarray:
        return np.asarray(
            [
                european_price(
                    self._stock_path[i],
                    self.strike,
                    self.expiry - i * self.dt,
                    self.sigma,
                    self.r,
                )
                for i in range(self.n_steps)
            ]
        )

    @flatten
    def _get_deltas(self) -> np.ndarray:
        return np.asarray(
            [
                v_delta_european(
                    self._stock_path[i],
                    self.strike,
                    self.expiry - i * self.dt,
                    self.sigma,
                    self.r,
                )
                for i in range(self.n_steps)
            ]
        )


class BlackScholesEnvCont(BlackScholesEnvBase):

    def __init__(
        self,
        s0: float,
        strike: float,
        expiry: float,
        r: float,
        mu: float,
        sigma: float,
        n_steps: int,
    ):
        super().__init__(
            s0=s0,
            strike=strike,
            expiry=expiry,
            r=r,
            mu=mu,
            sigma=sigma,
            n_steps=n_steps,
        )
        self.action_space = Box(low=-1.0, high=0.0, shape=(1,))
        self.observation_space = Box(
            low=np.array([-np.inf, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([np.inf, 2.0, np.inf, 1.0, np.inf, 1.0], dtype=np.float32),
            shape=(6,),
        )

    def step(self, action: np.ndarray):
        new_hedge = action[0]
        return super().step(new_hedge)


class BlackScholesEnvDis(BlackScholesEnvBase):

    def __init__(
        self,
        s0: float,
        strike: float,
        expiry: float,
        r: float,
        mu: float,
        sigma: float,
        n_steps: int,
    ):
        super().__init__(
            s0=s0,
            strike=strike,
            expiry=expiry,
            r=r,
            mu=mu,
            sigma=sigma,
            n_steps=n_steps,
        )
        self.action_space = Discrete(100)
        self.observation_space = Box(
            low=np.array([-np.inf, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([np.inf, 2.0, np.inf, 1.0, np.inf, 1.0], dtype=np.float32),
            shape=(6,),
        )

    def step(self, action: float):
        return super().step(-(action / 100))


if __name__ == "__main__":
    import pprint

    pp = pprint.PrettyPrinter(indent=4)

    SEED = 0

    env = BlackScholesEnvCont(100, 100, 1, 0.02, 0.05, 0.2, 252)
    obs, info = env.reset(seed=SEED)

    np.random.seed(SEED)

    n_steps = 270
    for i in range(n_steps):
        if i == 252:
            pass

        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)

        print(f"Step: {i}")
        print("Action:", action)
        print("Observations:")
        pp.pprint(obs)
        print("Info:")
        pp.pprint(info)
        print("Reward:", reward, end="\n\n")

        if done:
            obs, info = env.reset(seed=0)
            print("Resetting environment...")
