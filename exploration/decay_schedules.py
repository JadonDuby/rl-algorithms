from exploration.base_schedule import BaseSchedule
import math
from evaluation import policy_evaluator

class ExponentialDecaySchedule(BaseSchedule):
    def __init__(self, epsilon_start=1.0, epsilon_end=0.05, decay_rate=0.001):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate

    def get_epsilon(self, ep=None, step=None) -> float:
        print(f"{ep=}, {step=}")
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-self.decay_rate * step)
        return max(epsilon, 0.05)

    def update_epsilon(self):
        return 
    
class SigmoidSchedule(BaseSchedule):
    def __init__(self, n_episodes=100):
        self.n_episodes = n_episodes
        self.step = 1/n_episodes

    def get_epsilon(self, ep=None, step=None) -> float:
        x = self.n_episodes/2 - ep # want to scale x to between -10 and 10 over the episodes
        epsilon = max(1/(1+math.exp(-math.pow(x,1)*10/self.n_episodes)), 0.05)
        return epsilon
    
    def update_epsilon(self):
        return

class LinearSchedule(BaseSchedule):
    def __init__(self, n_episodes):
        self.n_episodes = n_episodes
        self.step = 1/n_episodes
        self.epsilon = 0.95

    def get_epsilon(self, ep=None, step=None):
        epsilon = 1-(ep/self.n_episodes)
        return max(epsilon, 0.1)
    
    def update_epsilon(self):
        self.epsilon -= self.step
    
class LinearStepSchedule(BaseSchedule):
    def __init__(self, n_episodes):
        self.n_episodes = n_episodes
        self.epsilon = 1
        self.step = 1/n_episodes

    def get_epsilon(self, ep=None, step=None):
        # epsilon = 1 - round(ep/self.n_episodes, 1)
        epsilon = round(self.epsilon, 1)
        # epsilon = max(self.epsilon, 0.05)
        return max(epsilon, 0.05)
    
    def update_epsilon(self):
        self.epsilon -= self.step
    
class DynamicStepSchedule(BaseSchedule):
    def __init__(self, n_episodes, callback=None):
        self.n_episodes = n_episodes
        self.callback = callback
        self.step = 1/n_episodes

    def get_epsilon(self, ep=None, step=None):
        if math.mod(ep, 10) == 0:
            avg_reward = self.callback
        epsilon = 1 - round(ep/self.n_episodes, 1)
        return epsilon

    def update_epsilon(self):
        self.epsilon -= self.step
        

class SawToothSchedule(BaseSchedule):
    def __init__(self, n_episodes, epsilon_start=0.0, epsilon_end = None ): #| callable):?
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.step = 1/n_episodes
    
    # def get_epsilon(self, ep=None, step=None) -> float:
    #     epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end)
    #     return max(epsilon, 0.05)

    def get_epsilon(self, ep=None, step=None):
        epsilon = 1 - round(ep/(step+1), 1)
        return epsilon
    
    def update_epsilon(self):
        self.epsilon -= self.step

        # self, epsilon_start=1.0, epsilon_end=0.05