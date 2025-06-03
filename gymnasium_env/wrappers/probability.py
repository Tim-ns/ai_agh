import gymnasium as gym
import random


class SkipMoveWrapper(gym.Wrapper):
    def __init__(self, env, skip_probability=0.1):
        super().__init__(env)
        self.skip_probability = skip_probability

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        if random.random() < self.skip_probability:
            obs2, reward2, terminated2, truncated2, info2 = self.env.step(action)

            reward += reward2
            terminated = terminated2
            truncated = truncated2
            info.update(info2)

        return obs, reward, terminated, truncated, info
