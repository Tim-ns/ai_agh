from collections import defaultdict
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import warnings
warnings.simplefilter('ignore')

class QLearner:
    def __init__(self, n_actions):
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = 0.8 # learning rate (can be set to a constant instead of a function)
        self.discount_factor = 0.7
        self.epsilon = 1
        self.n_actions = n_actions

    def choose_action(self, state):
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > self.epsilon:
            return np.argmax(self.Q[state])

        else:
            return random.randint(0, self.n_actions - 1)

    def update_Q(self, state, new_state, action, reward, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.Q[state][action] = self.Q[state][action] + alpha * (reward + self.discount_factor * np.max(self.Q[new_state]) - self.Q[state][action])

class DiscretizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, bins=8):
        super().__init__(env)
        self.num_bins = bins
        self.obs_low = self.observation_space.low
        self.obs_high = self.observation_space.high

        self.obs_low[self.obs_low == -np.inf] = -10
        self.obs_high[self.obs_high == np.inf] = 10

        self.obs_bins = [
            np.linspace(self.obs_low[i], self.obs_high[i], self.num_bins+1)[1:-1]
            for i in range(len(self.obs_low))
        ]

        self.observation_space = spaces.MultiDiscrete([self.num_bins] * len(self.obs_low))
    def observation(self, observation):
        return tuple(
            int(np.digitize(obs, self.obs_bins[i]))
            for i, obs in enumerate(observation)
        )


def train_agent_alpha(env, agent, max_steps, train_episodes, min_epsilon, max_epsilon, decay):
    raw_states = []
    training_rewards_alpha = []
    epsilons_alpha = []
    state_visit_count = {}
    for episode in tqdm(range(train_episodes), desc="Training Progress"):
        state, info = env.reset()
        # Convert state to a hashable type
        if isinstance(state, dict):
            state = tuple(sorted(state.items()))
        else:
            state = tuple(state)

        total_training_rewards = 0

        for step in range(max_steps):
            raw_states.append(state)
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Convert new_state to hashable type
            if isinstance(new_state, dict):
                new_state = tuple(sorted(new_state.items()))
            else:
                new_state = tuple(new_state)

            if state not in state_visit_count:
                state_visit_count[state] = 0
            state_visit_count[state] += 1

            alpha = 60 / (59 + state_visit_count[state])

            agent.update_Q_alpha(state, new_state, action, reward, alpha)
            total_training_rewards += reward
            state = new_state

            if terminated or truncated:
                break

        agent.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        training_rewards_alpha.append(total_training_rewards)
        epsilons_alpha.append(agent.epsilon)
    return training_rewards_alpha, epsilons_alpha
