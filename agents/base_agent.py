from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.
    Ensures that all agents implement the same core interface.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def preprocess_state(self, state):
        """Optional: convert to tensor, normalize, etc."""
        return state
    
    @abstractmethod
    def select_action(self, observation):
        """
        Selects an action given the current observation.
        """
        pass

    @abstractmethod
    def update(self, obs, action, reward, next_obs, done):
        """
        Updates the agent’s internal state (e.g. Q-values or weights).
        """
        pass

    def reset(self):
        """
        Optional: Reset any episode-specific variables.
        """
        pass
