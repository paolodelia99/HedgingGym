from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
from gymnasium import Env, spaces


class SpreadHedgingEnvBase(Env):
    metadata = {"render.modes": ["human"]}
    action_space: spaces.Space
    observation_space: spaces.Space

    __metaclass__ = ABCMeta

    def __init__(
        self,
        s1_0: float,
        s2_0: float,
        strike: float,
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
        self.s1_0 = s1_0
        self.s2_0 = s2_0
        self.strike = strike
        self.expiry = expiry
        self.r = r
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.corr = corr
        self.n_steps = n_steps
        self.dt = expiry / n_steps
        self._current_step = 0

        self.render_mode = "human"

        self._stock_path_1 = np.array([])
        self._stock_path_2 = np.array([])
        self._spread_prices = np.array([])
        self._deltas_1 = np.array([])
        self._deltas_2 = np.array([])

        self._hedging_portfolio_value = 0.0
        self._current_hedging_delta_1 = 0.0
        self._current_hedging_delta_2 = 0.0
        self._previous_hedging_delta_1 = 0.0
        self._previous_hedging_delta_2 = 0.0
        self._back_account_value = 0.0
        self._ddelta_1 = 0.0
        self._ddelta_2 = 0.0

        self._epsilon = 1
        self._lambda = 0.1
        self._tick_size = tick_size

    def reset(self, seed=None):
        super().reset(seed=seed)

        self._current_step = 0
        self._stock_path_1, self._stock_path_2 = self._generate_stock_path()
        self._spread_prices = self._get_spread_prices()
        self._deltas_1, self._deltas_2 = self._get_deltas()
        self._hedging_portfolio_value = self._spread_prices[0]
        self._current_hedging_delta_1 = -self._deltas_1[0]
        self._current_hedging_delta_2 = -self._deltas_2[0]
        self._back_account_value = (
            self._hedging_portfolio_value
            - self.current_hedging_delta_1 * self.s1_0
            - self.current_hedging_delta_2 * self.s2_0
        )

        observations = self._get_observations()
        infos = self._get_infos()

        return observations, infos

    def step(self, action: np.ndarray):
        self._current_step += 1
        delta_1 = action[0]
        delta_2 = action[1]

        done = self._current_step == (self.n_steps - 1)
        reward = self._calculate_reward()

        self._hedging_portfolio_value = self._calculate_hedging_portfolio_value(
            delta_1, delta_2
        )

        observations = self._get_observations()
        infos = self._get_infos()
        self._update_deltas(delta_1, delta_2)

        return observations, reward, done, False, infos

    def render(self, mode="human"):
        pass

    def _update_deltas(self, new_delta_1: float, new_delta_2: float):
        self._ddelta_1 = new_delta_1 - self.current_hedging_delta_1
        self._previous_hedging_delta_1 = self.current_hedging_delta_1
        self.current_hedging_delta_1 = new_delta_1
        self._ddelta_2 = new_delta_2 - self.current_hedging_delta_2
        self._previous_hedging_delta_2 = self.current_hedging_delta_2
        self.current_hedging_delta_2 = new_delta_2

    def _calculate_reward(self) -> float:
        if self._current_step == 1:
            return -self._get_transaction_costs(
                self.current_hedging_delta_1, self.current_hedging_delta_2
            )

        pnl = self._calculate_pnl()
        return pnl - self._lambda / 2 * pnl**2

    def _calculate_pnl(self) -> float:
        dv = (
            self._spread_prices[self._current_step]
            - self._spread_prices[self._current_step - 1]
        )
        ds_1 = (
            self._stock_path_1[self._current_step]
            - self._stock_path_1[self._current_step - 1]
        )[0]
        ds_2 = (
            self._stock_path_2[self._current_step]
            - self._stock_path_2[self._current_step - 1]
        )[0]
        ddelta_1 = self._ddelta_1
        ddelta_2 = self._ddelta_2

        if self._current_step == (self.n_steps - 1):
            liquidation_value = self._get_transaction_costs(
                self.current_hedging_delta_1, self.current_hedging_delta_2
            )
            return (
                dv
                + self.current_hedging_delta_1 * ds_1
                + self.current_hedging_delta_2 * ds_2
                - self._get_transaction_costs(ddelta_1, ddelta_2)
                - liquidation_value
            )

        return (
            dv
            + self.current_hedging_delta_1 * ds_1
            + self.current_hedging_delta_2 * ds_2
            - self._get_transaction_costs(ddelta_1, ddelta_2)
        )

    def _get_transaction_costs(self, ddelta_1: float, ddelta_2: float):
        return self._epsilon * self._tick_size * (
            np.abs(ddelta_1) + 0.01 * ddelta_1**2
        ) + self._epsilon * self._tick_size * (np.abs(ddelta_2) + 0.01 * ddelta_2**2)

    def _get_observations(self):
        log_price_strike = self._get_log_ratio()
        time_to_expiration = self.expiry
        vol_1, vol_2 = self._get_current_stock_vols(self._current_step)
        m_delta_1 = self._deltas_1[self._current_step]
        m_delta_2 = self._deltas_2[self._current_step]
        spread_price = self._spread_prices[self._current_step]
        return np.asarray(
            [
                log_price_strike,
                vol_1,
                vol_2,
                time_to_expiration,
                m_delta_1,
                m_delta_2,
                spread_price / self._spread_prices[0],
                self.current_hedging_delta_1,
                self.current_hedging_delta_2,
            ],
            dtype=np.float32,
        )

    def _get_infos(self):
        s1_t, s2_t = self._get_current_stock_prices()
        return {
            "price": self._spread_prices[self._current_step],
            "time_to_expiration": self.expiry - self._current_step * self.dt,
            "m_delta_1": self._deltas_1[self._current_step],
            "m_delta_2": self._deltas_2[self._current_step],
            "stock_1": s1_t,
            "stock_2": s2_t,
            "current_delta_1": self.current_hedging_delta_1,
            "current_delta_2": self.current_hedging_delta_2,
            "log(S_2/S_1)": self._get_log_ratio(),
            "hedge_portfolio_value": self._hedging_portfolio_value,
            "bank_account": self._back_account_value,
        }

    @property
    def current_hedging_delta_1(self):
        return self._current_hedging_delta_1

    @current_hedging_delta_1.setter
    def current_hedging_delta_1(self, new_hedge: float):
        self._current_hedging_delta_1 = new_hedge

    @property
    def current_hedging_delta_2(self):
        return self._current_hedging_delta_2

    @current_hedging_delta_2.setter
    def current_hedging_delta_2(self, new_hedge: float):
        self._current_hedging_delta_2 = new_hedge

    def _get_log_ratio(self):
        if self._current_step == -1:
            return np.log(self.s2_0 / self.s1_0)
        return np.log(
            self._stock_path_1[self._current_step, 0]
            / self._stock_path_2[self._current_step, 0]
        )

    @abstractmethod
    def _get_current_stock_vols(self, step: int) -> Tuple[float, float]:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _generate_stock_path(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _get_spread_prices(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _get_deltas(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclasses must implement this method")

    def _calculate_hedging_portfolio_value(
        self, new_delta_1: float, new_delta_2: float
    ):
        stock_1, stock_2 = self._get_current_stock_prices()

        new_hedging_port_value = (
            self._back_account_value
            + self._current_hedging_delta_1 * stock_1
            + self._current_hedging_delta_2 * stock_2
        )
        self._back_account_value = (
            new_hedging_port_value - new_delta_1 * stock_1 - new_delta_2 * stock_2
        )

        return -new_hedging_port_value

    def _get_current_stock_prices(self) -> Tuple[float, float]:
        return (
            self._stock_path_1[self._current_step, 0],
            self._stock_path_2[self._current_step, 0],
        )
