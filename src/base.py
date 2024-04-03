from abc import ABCMeta, abstractmethod

import numpy as np
from gymnasium import Env, spaces


class HedgingEnvBase(Env):
    metadata = {"render.modes": ["human"]}
    action_space: spaces.Space
    observation_space: spaces.Space

    __metaclass__ = ABCMeta

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

        self.render_mode = "human"

        self._stock_path = np.array([])
        self._call_prices = np.array([])
        self._deltas = np.array([])

        self._hedging_portfolio_value = 0.0
        self._current_hedging_delta = 0.0
        self._previous_hedging_delta = 0.0
        self._back_account_value = 0.0
        self._ddelta = 0.0

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
        self._back_account_value = (
            self._hedging_portfolio_value - self.current_hedging_delta * self.s0
        )

        observations = self._get_observations()
        infos = self._get_infos()

        return observations, infos

    def step(self, action: float):
        self._current_step += 1

        done = self._current_step == (self.n_steps - 1)
        reward = self._calculate_reward()

        self._hedging_portfolio_value = self._calculate_hedging_portfolio_value(action)

        observations = self._get_observations()
        infos = self._get_infos()
        self._update_delta(action)

        return observations, reward, done, False, infos

    def render(self, mode="human"):
        pass

    def _update_delta(self, new_delta: float):
        self._ddelta = new_delta - self.current_hedging_delta
        self._previous_hedging_delta = self.current_hedging_delta
        self.current_hedging_delta = new_delta

    def _calculate_reward(self) -> float:
        if self._current_step == 1:
            return -self._get_transaction_costs(self.current_hedging_delta)

        pnl = self._calculate_pnl()
        return pnl - self._lambda / 2 * pnl**2

    def _calculate_pnl(self) -> float:
        dv = (
            self._call_prices[self._current_step]
            - self._call_prices[self._current_step - 1]
        )
        ds = (
            self._stock_path[self._current_step]
            - self._stock_path[self._current_step - 1]
        )[0]
        ddelta = self._ddelta

        if self._current_step == (self.n_steps - 1):
            liquidation_value = self._get_transaction_costs(self.current_hedging_delta)
            return (
                dv
                + self.current_hedging_delta * ds
                - self._get_transaction_costs(ddelta)
                - liquidation_value
            )

        return (
            dv + self.current_hedging_delta * ds - self._get_transaction_costs(ddelta)
        )

    def _get_transaction_costs(self, ddelta: float):
        return self._epsilon * self._tick_size * (np.abs(ddelta) + 0.01 * ddelta**2)

    def _get_observations(self):
        log_price_strike = self._get_log_ratio()
        time_to_expiration = self.expiry
        bs_delta = self._deltas[self._current_step]
        call_price = self._call_prices[self._current_step]
        return np.asarray(
            [
                log_price_strike,
                self.sigma,
                time_to_expiration,
                bs_delta,
                call_price / self._call_prices[0],
                self.current_hedging_delta,
            ],
            dtype=np.float32,
        )

    def _get_infos(self):
        return {
            "price": self._call_prices[self._current_step],
            "time_to_expiration": self.expiry - self._current_step * self.dt,
            "bs_delta": self._deltas[self._current_step],
            "stock_price": self._get_current_stock_price(),
            "current_delta": self.current_hedging_delta,
            "log(S/K)": self._get_log_ratio(),
            "hedge_portfolio_value": self._hedging_portfolio_value,
            "bank_account": self._back_account_value,
        }

    @property
    def current_hedging_delta(self):
        return self._current_hedging_delta

    @current_hedging_delta.setter
    def current_hedging_delta(self, new_hedge: float):
        self._current_hedging_delta = new_hedge

    def _get_log_ratio(self):
        if self._current_step == -1:
            return np.log(self.s0 / self.strike)
        return np.log(self._stock_path[self._current_step, 0] / self.strike)

    @abstractmethod
    def _generate_stock_path(self, seed=None) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _get_call_prices(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _get_deltas(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def _calculate_hedging_portfolio_value(self, new_delta: float):
        if self._current_step == 0:
            self._hedging_portfolio_value = self._call_prices[0]
            self._back_account_value = (
                self._hedging_portfolio_value - self._current_hedging_delta * self.s0
            )
            return self._hedging_portfolio_value

        new_hedging_port_value = (
            self._back_account_value
            + self._current_hedging_delta * self._get_current_stock_price()
        )
        self._back_account_value = (
            new_hedging_port_value - new_delta * self._get_current_stock_price()
        )

        return new_hedging_port_value

    def _get_current_stock_price(self):
        return self._stock_path[self._current_step, 0]
