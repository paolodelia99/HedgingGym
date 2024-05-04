import numpy as np
from jaxfin.price_engine.fft import delta_call_fourier, fourier_inv_call

from hedging_gym.envs.heston_env import HestonEnvCont, HestonEnvDis
from hedging_gym.utils.env_checker import check_env

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
        "bank_account": 40.564895188662994,
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
        "bank_account": 40.564895188662994,
    }
    expected_obs = np.array(
        [0.0, 0.4472136, 1.0, bs_delta_0, 1.0, -bs_delta_0], dtype=np.float32
    )

    assert np.array_equal(obs, expected_obs)
    for key in expected_info:
        assert np.isclose(info[key], expected_info[key])


def test_step():
    env = HestonEnvCont(
        s0, strike, expiry, r, mu, v0, kappa, theta, sigma, rho, n_steps
    )
    obs, info = env.reset(seed=SEED)
    dt = expiry / n_steps
    stock_path = env._stock_path[:, 0]
    var_process = env._variance_process

    expected_call_prices = np.asarray(
        [
            fourier_inv_call(
                s, strike, expiry - i * dt, var_process[i], mu, theta, sigma, kappa, rho
            )
            for i, s in enumerate(stock_path)
        ]
    )
    expected_deltas = np.asarray(
        [
            delta_call_fourier(
                s, strike, expiry - i * dt, var_process[i], mu, theta, sigma, kappa, rho
            )
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

    assert np.allclose(np.asarray(call_prices), expected_call_prices)
    assert np.allclose(np.asarray(bs_deltas), expected_deltas)
