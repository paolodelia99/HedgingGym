from gymnasium.envs.registration import register

register(
    id="CallHedgingBSCont-v0", entry_point="hedging_gym.envs.bs_env:BlackScholesEnvCont"
)

register(
    id="CallHedgingBSDiscrete-v0",
    entry_point="hedging_gym.envs.bs_env:BlackScholesEnvDis",
)

register(
    id="CallHedgingHestonCont-v0",
    entry_point="hedging_gym.envs.heston_env:HestonEnvCont",
)

register(
    id="CallHedgingHestonDiscrete-v0",
    entry_point="hedging_gym.envs.heston_env:HestonEnvDis",
)

register(
    id="MargrabeHedgingCont-v0",
    entry_point="hedging_gym.envs.margrabe_env:MargrabeEnvCont",
)
