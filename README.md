# Hedging Gym

Hedging Gym is a reinforcement learning environment for training and testing hedging strategies. It is built on top of OpenAI's Gym framework and is designed to simulate real-world financial markets.

The hedging environment specifics are based on the following paper: [Delta hedging with Deep Reinforcement Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3847272) by Giurca and Borovkova (2021).

The rewards function is defined as 

\[
    r_t(s_t, a_t) = PL_t(s_t, a_t) - \frac{\lambda}{2} PL_t(s_t, a_t)^2
\]

where the profit and loss function is defined as

\[
    PL_t(s_t, a_t) = \delta V_t + \Delta (\delta S_t) - c(\delta \Delta_t)
\]

where \(V_t\) is the option value, \(S_t\) is the stock price, \(\Delta\) is the delta used to hedged the position, and \(c\) is the cost function that represents the transaction costs for the underlying between \( t - 1 \) and \( t \). The cost function is defined as 

\[
    c(\delta \Delta_t) = \xi \times \textit{Ticksize} \times (\delta \Delta_t + 0.01(\delta \Delta)^2)
\]

Hereby, $\xi \in \mathbb{R}_{+}$is the level of market friction, Ticksize $\times\left(\left|\delta \Delta_{t}\right|\right)$ is a cost relative to the midpoint of crossing a two ticks wide bid-ask spread (for $\xi=1$ ) and $\xi \times$ Ticksize $\left.\times 0.01\left(\delta \Delta_{t}\right)^{2}\right)$ is a simplified version of market impact.

## Features

- Simulates Black-Scholes environment for call option hedging, supports both discrete and continuous action spaces.
- Simulates Heston environment for option hedging, for both discrete and continuous action spaces.
- Simulates Exchange option environment (Margrabe Environment) for option hedging, for only in continuous action spaces.

## Installation

To install Hedging Gym, run the following command:

```bash
pip install hedging-gym
```

## Usage

Here is an example of how to create a Hedging Gym environment:

```python
import gymnasium

# Initialize the environment
env = gymnasium.make(
        "CallHedgingBSCont-v0",
        s0=s0,
        strike=strike,
        expiry=expiry,
        r=r,
        mu=mu,
        sigma=sigma,
        n_steps=n_steps,
    )

# Reset the environment
obs, info = env.reset(seed=SEED)

# Take a step
action = np.array([bs_delta_0], dtype=np.float32)
obs, reward, done, info = env.step(action)
```