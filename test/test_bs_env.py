import numpy as np
from jaxfin.price_engine.black_scholes import delta_european, european_price

from src.bs_env import BlackScholesEnvCont, BlackScholesEnvDis
from src.utils.env_checker import check_env

s0 = 100.0
strike = 100.0
expiry = 1.0
r = 0.0
mu = 0.0
sigma = 0.2
n_steps = 252
dt = expiry / n_steps

SEED = 0


def test_check_env_cont():
    env = BlackScholesEnvCont(s0, strike, expiry, r, mu, sigma, n_steps)

    try:
        check_env(env)
    except Exception as e:
        assert False, f"check_env raised an exception: {e}"

    assert True


def test_check_env_dis():
    env = BlackScholesEnvDis(s0, strike, expiry, r, mu, sigma, n_steps)

    try:
        check_env(env)
    except Exception as e:
        assert False, f"check_env raised an exception: {e}"

    assert True


def test_reset_cont():
    env = BlackScholesEnvCont(s0, strike, expiry, r, mu, sigma, n_steps)
    obs, info = env.reset(seed=SEED)

    bs_delta_0 = delta_european(s0, strike, expiry, sigma, r)
    call_price_0 = european_price(s0, strike, expiry, sigma, r)

    expected_info = {
        "price": call_price_0,
        "time_to_expiration": 1.0,
        "bs_delta": bs_delta_0,
        "stock_price": s0,
        "current_delta": -bs_delta_0,
        "log(S/K)": np.log(s0 / strike),
        "hedge_portfolio_value": call_price_0,
        "bank_account": 46.017221450805664,
    }
    expected_obs = np.array(
        [0.0, sigma, 1.0, bs_delta_0, 1.0, -bs_delta_0], dtype=np.float32
    )

    assert np.array_equal(obs, expected_obs)
    for key in expected_info:
        assert np.isclose(info[key], expected_info[key])


def test_reset_dis():
    env = BlackScholesEnvDis(s0, strike, expiry, r, mu, sigma, n_steps)
    obs, info = env.reset(seed=SEED)

    bs_delta_0 = 0.5398278
    call_price_0 = 7.965561

    expected_info = {
        "price": call_price_0,
        "time_to_expiration": 1.0,
        "bs_delta": bs_delta_0,
        "stock_price": s0,
        "current_delta": -bs_delta_0,
        "log(S/K)": np.log(s0 / strike),
        "hedge_portfolio_value": call_price_0,
        "bank_account": 46.017221450805664,
    }
    expected_obs = np.array(
        [0.0, sigma, 1.0, bs_delta_0, 1.0, -bs_delta_0], dtype=np.float32
    )

    assert np.array_equal(obs, expected_obs)
    for key in expected_info:
        assert np.isclose(info[key], expected_info[key])


def test_step():
    env = BlackScholesEnvCont(s0, strike, expiry, r, mu, sigma, n_steps, 0.00)
    obs, info = env.reset(seed=SEED)
    dt = expiry / n_steps
    stock_path = env._stock_path[:, 0]

    expected_call_prices = np.asarray(
        [
            european_price(s, strike, expiry - i * dt, sigma, r)
            for i, s in enumerate(stock_path)
        ]
    )
    expected_deltas = np.asarray(
        [
            delta_european(s, strike, expiry - i * dt, sigma, r)
            for i, s in enumerate(stock_path)
        ]
    )

    call_prices = [info["price"]]
    bs_deltas = [info["bs_delta"]]

    for i in range(252):
        action = env.action_space.sample()

        obs, rewards, done, _, info = env.step(action)

        call_prices.append(info["price"])
        bs_deltas.append(info["bs_delta"])

        if done:
            break

    np.allclose(np.asarray(call_prices), expected_call_prices)
    np.allclose(np.asarray(bs_deltas), expected_deltas)
