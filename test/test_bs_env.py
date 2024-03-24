import numpy as np

from src.bs_env import BlackScholesEnv
from src.utils.env_checker import check_env

s0 = 100.0
strike = 100.0
expiry = 1.0
r = 0.0
mu = 0.0
sigma = 0.2
n_steps = 252

SEED = 0

def test_check_env():
    env = BlackScholesEnv(s0, strike, expiry, r, mu, sigma, n_steps)
    
    try:
        check_env(env)
    except Exception as e:
        assert False, f"check_env raised an exception: {e}"

    assert True


def test_reset():
    env = BlackScholesEnv(s0, strike, expiry, r, mu, sigma, n_steps)
    obs, info = env.reset(seed=SEED)

    bs_delta_0 = 0.5398278
    call_price_0 = 7.965561

    expected_info = {
        "price": call_price_0,
        "time_to_expiration": 1.0,
        "bs_delta": bs_delta_0,
        "stock_price": s0,
        "current_delta": bs_delta_0,
        "log(S/K)": np.log(s0 / strike),
        "hedge_portfolio_value": call_price_0,
        "bank_account": -46.017221450805664
    }
    expected_obs = np.array([0.0, 1.0, bs_delta_0, call_price_0, call_price_0], dtype=np.float32)

    assert np.array_equal(obs, expected_obs)
    for key in expected_info:
        assert np.isclose(info[key], expected_info[key])
