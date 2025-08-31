from agents.base_agent import BaseAgent
import numpy as np

class LinearFunctionApproxAgent(BaseAgent):
    def __init__(self, action_space, feature_engineer, epsilon_schedule, alpha=0.99, gamma=0.2):
        self.action_space = action_space
        self.n_actions = action_space.n
        self.feature_engineer = feature_engineer
        self.epsilon_schedule = epsilon_schedule
        self.alpha = alpha
        self.gamma = gamma
        example_obs = np.zeros(4)
        feature_len = len(feature_engineer.transform(example_obs, self.n_actions)[0])
        self.weights = np.ones(feature_len)
        self.epsilon = 1.0
        self.epsilon_curve = []
        self.step = 0
        self.episode = 0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def select_action(self, observation):
        epsilon = self.epsilon_schedule.get_epsilon(ep = self.episode, step = self.step)
        self.step += 1
        self.epsilon_curve.append(epsilon)
        if np.random.rand() < epsilon:
            return self.action_space.sample()
        q_values = self.get_q_values(observation)
        return np.argmax(q_values)

    def get_q_values(self, observation):
        features = self.feature_engineer.transform(observation, self.n_actions)
        return [np.dot(self.weights, f) for f in features]

    def update(self, obs, action, reward, next_obs, done, episode):
        current_features = self.feature_engineer.transform(obs, self.n_actions)
        next_features = self.feature_engineer.transform(next_obs, self.n_actions)

        q_next = np.max([np.dot(self.weights, f) for f in next_features])
        target = reward + (0 if done else self.gamma * q_next)
        td_error = target - np.dot(self.weights, current_features[action])

        self.weights = (1-self.alpha)*(self.weights) + self.alpha*(current_features[action]*td_error)
        self.episode = episode
