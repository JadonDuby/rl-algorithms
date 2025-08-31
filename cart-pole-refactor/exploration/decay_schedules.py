from exploration.base_schedule import BaseSchedule
import math
from evaluation import policy_evaluator

class ExponentialDecaySchedule(BaseSchedule):
    def __init__(self, epsilon_start=1.0, epsilon_end=0.05, decay_rate=0.001):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate

    def get_epsilon(self, ep=None, step=None) -> float:
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-self.decay_rate * step)
        return max(epsilon, 0.05)
    
class SigmoidSchedule(BaseSchedule):
    def __init__(self, n_episodes=100):
        self.n_episodes = n_episodes

    def get_epsilon(self, ep=None, step=None) -> float:
        x = self.n_episodes/2 - ep # want to scale x to between -10 and 10 over the episodes
        epsilon = (1/(1+math.exp(-math.pow(5*x/self.n_episodes, 5))))
        return epsilon

class LinearSchedule(BaseSchedule):
    def __init__(self, n_episodes):
        self.n_episodes = n_episodes

    def get_epsilon(self, ep=None, step=None):
        epsilon = 1-(ep/self.n_episodes)
        return max(epsilon, 0.1)
    
class LinearStepSchedule(BaseSchedule):
    def __init__(self, n_episodes):
        self.n_episodes = n_episodes

    def get_epsilon(self, ep=None, step=None):
        epsilon = 1 - round(ep/self.n_episodes, 1)
        return epsilon
    
class DynamicStepSchedule(BaseSchedule):
    def __init__(self, n_episodes, callback=None):
        self.n_episodes = n_episodes

    def get_epsilon(self, ep=None, step=None):
        if math.mod(ep, 10) == 0:
            evaluator = policy_evaluator()
        epsilon = 1 - round(ep/self.n_episodes, 1)
        return epsilon
    
class SawToothSchedule(BaseSchedule):
    def __init__(self, n_episodes, epsilon_start=0.0, epsilon_end = None ): #| callable):?
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
    
    # def get_epsilon(self, ep=None, step=None) -> float:
    #     epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end)
    #     return max(epsilon, 0.05)

    def get_epsilon(self, ep=None, step=None):
        epsilon = 1 - round(ep/(step+1), 1)
        return epsilon

        # self, epsilon_start=1.0, epsilon_end=0.05