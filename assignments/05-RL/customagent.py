import gymnasium as gym

import numpy as np


class Agent:
    """Agent using hard coded solution."""

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """Initialise the outputs."""
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """Decides how to move the lunar lander, adapted from
        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py.
        """
        s = observation

        angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        hover_targ = 0.55 * np.abs(
            s[0]
        )  # target y should be proportional to horizontal offset

        angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
        hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

        if s[6] or s[7]:  # legs have contact
            angle_todo = 0
            hover_todo = (
                -(s[3]) * 0.5
            )  # override to reduce fall speed, that's all we need after contact

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
        return a

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """RL need not apply."""
        pass
