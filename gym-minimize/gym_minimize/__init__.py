
from gym.envs.registration import register

register(
    id='minimize-v0',
    entry_point='gym_minimize.envs:MinimizeEnv',
)