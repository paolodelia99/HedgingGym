# Hedging Gym

Hedging Gym is a reinforcement learning environment for training and testing hedging strategies. It is built on top of OpenAI's Gym framework and is designed to simulate real-world financial markets.

## Features

- Simulates Black-Scholes environment for call option hedging.
- Supports both discrete and continuous action spaces.
- Simulates Heston environment for option hedging, for both discrete and continuous action spaces.

## Installation

To install Hedging Gym, run the following command:

```bash
pip install hedging-gym
```

## Usage

Here is an example of how to create a Hedging Gym environment:

```python
from hedging_gym import BlackScholesEnvCont

# Initialize the environment
env = BlackScholesEnvCont(s0, strike, expiry, r, mu, sigma, n_steps)

# Reset the environment
obs, info = env.reset(seed=SEED)

# Take a step
action = np.array([bs_delta_0], dtype=np.float32)
obs, reward, done, info = env.step(action)
```