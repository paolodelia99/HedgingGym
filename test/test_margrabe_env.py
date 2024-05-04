import numpy as np

from hedging_gym.envs.margrabe_env import MargrabeEnvCont
from hedging_gym.math import margrabe, margrabe_deltas
from hedging_gym.utils.env_checker import check_env

s1_0 = 100.0
s2_0 = 100.0
expiry = 1.0
r = 0.0
mu_1 = 0.0
mu_2 = 0.0
sigma_1 = 0.2
sigma_2 = 0.2
corr = 0.5
n_steps = 252
dt = expiry / n_steps

SEED = 0

np.random.seed(SEED)


def test_check_env_cont():
    env = MargrabeEnvCont(
        s1_0,
        s2_0,
        expiry,
        r,
        mu_1,
        mu_2,
        sigma_1,
        sigma_2,
        corr,
        n_steps,
        0.01,
    )

    try:
        check_env(env)
    except Exception as e:
        assert False, f"check_env raised an exception: {e}"

    assert True


def test_reset_cont():
    env = MargrabeEnvCont(
        s1_0,
        s2_0,
        expiry,
        r,
        mu_1,
        mu_2,
        sigma_1,
        sigma_2,
        corr,
        n_steps,
        0.01,
    )
    obs, info = env.reset(seed=SEED)

    deltas = margrabe_deltas(s1_0, s2_0, expiry, sigma_1, sigma_2, corr)
    m_delta_1, m_delta_2 = deltas[0], deltas[1]
    spread_price = margrabe(s1_0, s2_0, expiry, sigma_1, sigma_2, corr)
    log_ratio = np.log(s2_0 / s1_0)

    expected_info = {
        "price": spread_price,
        "time_to_expiration": 1.0,
        "m_delta_1": m_delta_1,
        "m_delta_2": m_delta_2,
        "stock_1": s1_0,
        "stock_2": s2_0,
        "current_delta_1": m_delta_1,
        "current_delta_2": m_delta_2,
        "log(S_2/S_1)": log_ratio,
        "hedge_portfolio_value": spread_price,
        "bank_account": -15.921075323618034,
    }
    expected_obs = np.array(
        [
            log_ratio,
            sigma_1,
            sigma_2,
            1.0,
            m_delta_1,
            m_delta_2,
            1.0,
            m_delta_1,
            m_delta_2,
        ],
        dtype=np.float32,
    )

    assert np.array_equal(obs, expected_obs)
    for key in expected_info:
        assert np.isclose(info[key], expected_info[key])


def test_margrabe_step():
    env = MargrabeEnvCont(
        s1_0,
        s2_0,
        expiry,
        r,
        mu_1,
        mu_2,
        sigma_1,
        sigma_2,
        corr,
        n_steps,
        0.01,
    )
    obs, info = env.reset(seed=SEED)
    dt = expiry / n_steps
    stock_path_1, stock_path_2 = env._stock_path_1, env._stock_path_2

    expected_spread_prices = np.asarray(
        [
            margrabe(
                stock_path_2[i],
                stock_path_1[i],
                expiry - i * dt,
                sigma_1,
                sigma_2,
                corr,
            )
            for i in range(n_steps)
        ]
    ).flatten()
    expected_deltas = np.asarray(
        [
            margrabe_deltas(
                stock_path_2[i, 0],
                stock_path_1[i, 0],
                expiry - i * dt,
                sigma_1,
                sigma_2,
                corr,
            )
            for i in range(n_steps)
        ]
    )
    expected_delta_1 = expected_deltas[:, 0]
    expected_delta_2 = expected_deltas[:, 1]

    spread_prices = [info["price"]]
    deltas_1 = [info["m_delta_1"]]
    deltas_2 = [info["m_delta_2"]]

    for _ in range(252):
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, done, _, info = env.step(action)

        spread_prices.append(info["price"])
        deltas_1.append(info["m_delta_1"])
        deltas_2.append(info["m_delta_2"])

        if done:
            break

    assert np.allclose(np.asarray(spread_prices), expected_spread_prices)
    assert np.allclose(np.asarray(deltas_1), expected_delta_1)
    assert np.allclose(np.asarray(deltas_2), expected_delta_2)
