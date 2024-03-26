import numpy as np

from jax import vmap

from gymnasium import Env
from gymnasium.spaces import Box, Dict

from jaxfin.price_engine.black_scholes import european_price, delta_european
from jaxfin.models.gbm import UnivGeometricBrownianMotion

v_delta_european = vmap(delta_european, in_axes=(0, None, None, None, None))

def flatten(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs).flatten()

    return wrapper

class BlackScholesEnv(Env):
    metadata = {"render.modes": ["human"]}

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
        self.s0 = s0
        self.strike = strike
        self.expiry = expiry
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = expiry / n_steps
        self._current_step = 0

        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = Box(
            low=np.array([-np.inf, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0, np.inf, 1.0], dtype=np.float32),
            shape=(5,)
        )
        self.render_mode = "human"

        self._stock_path = np.array([])
        self._call_prices = np.array([])
        self._deltas = np.array([])

        self._hedging_portfolio_value = 0.0
        self._current_hedging_delta = 0.0
        self._back_account_value = 0.0

        self._epsilon = 1
        self._lambda = 0.1
        self._tick_size = 0.01

    def reset(self, seed=None):
        super().reset(seed=seed)

        self._current_step = 0
        self._stock_path = self._generate_stock_path(seed=seed)
        self._call_prices = self._get_call_prices()
        self._deltas = self._get_deltas()
        self._hedging_portfolio_value = self._call_prices[0]
        self._current_hedging_delta = -self._deltas[0]
        self._back_account_value = self._hedging_portfolio_value - self.current_hedging_delta * self.s0

        observations = self._get_observations()
        infos = self._get_infos()

        return observations, infos

    def step(self, action):
        self._current_step += 1

        done = self._current_step == (self.n_steps - 1)
        reward = self._calculate_reward(action[0])

        self._hedging_portfolio_value = self._calculate_hedging_portfolio_value(action[0])

        observations = self._get_observations()
        infos = self._get_infos()
        self.current_hedging_delta = action

        return observations, reward, done, False, infos

    def render(self, mode="human"):
        pass

    def _calculate_reward(self, new_delta: float) -> float:
        pnl = self._calculate_pnl(new_delta)
        return pnl - self._lambda / 2 * pnl**2

    def _calculate_pnl(self, new_delta: float) -> float:
        dv = (
            self._call_prices[self._current_step]
            - self._call_prices[self._current_step - 1]
        )
        ds = (
            self._stock_path[self._current_step]
            - self._stock_path[self._current_step - 1]
        )[0]
        ddelta = (new_delta - self.current_hedging_delta)

        if self._current_step == (self.n_steps - 1):
            liquidation_value = self._get_transaction_costs(new_delta)
            return dv + new_delta * ds - self._get_transaction_costs(ddelta) - liquidation_value

        return dv + new_delta * ds - self._get_transaction_costs(ddelta)

    def _get_transaction_costs(self, ddelta):
        return self._epsilon * self._tick_size * (ddelta + 0.01 * ddelta**2)

    def _get_observations(self):
        log_price_strike = self._get_log_ratio()
        time_to_expiration = self.expiry
        bs_delta = self._deltas[self._current_step]
        call_price = self._call_prices[self._current_step]
        return np.asarray([log_price_strike, time_to_expiration, bs_delta, call_price, self.current_hedging_delta], dtype=np.float32)

    def _get_infos(self):
        return {
            "price": self._call_prices[self._current_step],
            "time_to_expiration": self.expiry - self._current_step * self.dt,
            "bs_delta": self._deltas[self._current_step],
            "stock_price": self._get_current_stock_price(),
            "current_delta": self.current_hedging_delta,
            "log(S/K)": self._get_log_ratio(),
            "hedge_portfolio_value": self._hedging_portfolio_value,
            "bank_account": self._back_account_value
        }

    @property
    def current_hedging_delta(self):
        return self._current_hedging_delta

    @current_hedging_delta.setter
    def current_hedging_delta(self, new_hedge: np.ndarray):
        self._current_hedging_delta = new_hedge[0]

    def _get_log_ratio(self):
        if self._current_step == -1:
            return np.log(self.s0 / self.strike)
        return np.log(self._stock_path[self._current_step, 0] / self.strike)

    def _generate_stock_path(self, seed=None) -> np.ndarray:
        if seed:
            seed = 0

        gbm = UnivGeometricBrownianMotion(self.s0, self.mu, self.sigma)

        return np.asarray(gbm.sample_paths(0, self.expiry, self.n_steps, 1))

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

    def _calculate_hedging_portfolio_value(self, new_delta: float):
        if self._current_step == 0:
            self._hedging_portfolio_value = self._call_prices[0]
            self._back_account_value = (
                self._hedging_portfolio_value
                - self._current_hedging_delta * self.s0
            )
            return self._hedging_portfolio_value

        new_hedging_port_value = (
            self._back_account_value
            + self._current_hedging_delta * self._get_current_stock_price()
        )
        self._back_account_value = (
            new_hedging_port_value
            - new_delta * self._get_current_stock_price()
        )

        return new_hedging_port_value

    def _get_current_stock_price(self):
        return self._stock_path[self._current_step, 0]


if __name__ == '__main__':
    import pprint

    pp = pprint.PrettyPrinter(indent=4)

    SEED = 0

    env = BlackScholesEnv(100, 100, 1, 0.02, 0.05, 0.2, 252)
    obs, info = env.reset(seed=SEED)

    np.random.seed(SEED)

    n_steps = 270
    for i in range(n_steps):
        if i == 252:
            pass

        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        
        print(f'Step: {i}')
        print('Action:', action)
        print('Observations:')
        pp.pprint(obs)
        print('Info:')
        pp.pprint(info)
        print('Reward:', reward)

        if done:
            obs, info = env.reset(seed=0)
            print('Resetting environment...')
