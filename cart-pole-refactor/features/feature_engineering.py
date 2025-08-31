import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

class FeatureEngineer:
    def __init__(self, basis='poly', degree=2):
        self.basis = basis
        self.degree = degree

    def transform(self, observation, n_actions):
        obs = self._basis_transform(observation)
        obs = self._scale(obs)

        features = []
        for j in range(n_actions):
            feature = np.zeros((n_actions, len(obs)))
            feature[j] = obs
            flat = feature.flatten()
            flat = np.append(flat, 1)  # bias
            features.append(flat)
        return features

    def _scale(self, obs):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        obs = np.array(obs).reshape(-1, 1)
        return scaler.fit_transform(obs).flatten()

    def _basis_transform(self, observation):
        if self.basis == 'poly':
            poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            return poly.fit_transform([observation])[0]
        elif self.basis == 'raw':
            return np.array(observation)
        else:
            raise NotImplementedError(f"{self.basis} basis not implemented")