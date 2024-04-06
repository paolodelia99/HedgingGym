import numpy as np
import pytest

from src.heston_env import HestonEnvCont, HestonEnvDis
from src.utils.env_checker import check_env

s0 = 100.0
strike = 100.0
expiry = 1.0
r = 0.0
mu = 0.0
v0 = 0.2
kappa = 2.0
theta = 0.3
sigma = 0.1
rho = -0.5
n_steps = 252
dt = expiry / n_steps

SEED = 0


def test_check_env_cont():
    env = HestonEnvCont(
        s0, strike, expiry, r, mu, v0, kappa, theta, sigma, rho, n_steps
    )

    try:
        check_env(env)
    except Exception as e:
        assert False, f"check_env raised an exception: {e}"

    assert True


def test_check_env_dis():
    env = HestonEnvDis(s0, strike, expiry, r, mu, v0, kappa, theta, sigma, rho, n_steps)

    try:
        check_env(env)
    except Exception as e:
        assert False, f"check_env raised an exception: {e}"

    assert True


def test_reset_cont():
    env = HestonEnvCont(
        s0, strike, expiry, r, mu, v0, kappa, theta, sigma, rho, n_steps
    )
    obs, info = env.reset(seed=SEED)

    bs_delta_0 = 0.6048372
    call_price_0 = 19.924908

    expected_info = {
        "price": call_price_0,
        "time_to_expiration": 1.0,
        "bs_delta": bs_delta_0,
        "stock_price": s0,
        "current_delta": -bs_delta_0,
        "log(S/K)": np.log(s0 / strike),
        "hedge_portfolio_value": call_price_0,
        "bank_account": 80.40862560272217,
    }
    expected_obs = np.array(
        [0.0, 0.4472136, 1.0, bs_delta_0, 1.0, -bs_delta_0], dtype=np.float32
    )

    assert np.array_equal(obs, expected_obs)
    for key in expected_info:
        assert np.isclose(info[key], expected_info[key])


def test_reset_dis():
    env = HestonEnvDis(s0, strike, expiry, r, mu, v0, kappa, theta, sigma, rho, n_steps)
    obs, info = env.reset(seed=SEED)

    bs_delta_0 = 0.6048372
    call_price_0 = 19.924908

    expected_info = {
        "price": call_price_0,
        "time_to_expiration": 1.0,
        "bs_delta": bs_delta_0,
        "stock_price": s0,
        "current_delta": -bs_delta_0,
        "log(S/K)": np.log(s0 / strike),
        "hedge_portfolio_value": call_price_0,
        "bank_account": 80.40862560272217,
    }
    expected_obs = np.array(
        [0.0, 0.4472136, 1.0, bs_delta_0, 1.0, -bs_delta_0], dtype=np.float32
    )

    assert np.array_equal(obs, expected_obs)
    for key in expected_info:
        assert np.isclose(info[key], expected_info[key])


@pytest.mark.skip(reason="Skipping this test for now")
def test_step():
    env = HestonEnvCont(
        s0, strike, expiry, r, mu, v0, kappa, theta, sigma, rho, n_steps
    )
    obs, info = env.reset(seed=SEED)

    bs_delta_0 = 0.6048372
    call_price_0 = 19.924908

    action = np.array([bs_delta_0], dtype=np.float32)
    obs, reward, done, info = env.step(action)

    expected_obs = np.array(
        [0.0, 1.0 - dt, bs_delta_0, call_price_0, call_price_0], dtype=np.float32
    )
