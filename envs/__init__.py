from gymnasium.envs.registration import register

register(
    id="KickBoxing-v0",
    entry_point="envs.kickboxing_env:KickBoxingEnv",
    max_episode_steps=1000,
)
