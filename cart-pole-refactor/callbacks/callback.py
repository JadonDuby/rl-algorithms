from callbacks.base_callback import Callback
from evaluation.policy_evaluator import PolicyEvaluator
import numpy as np

class PrintRewardCallback(Callback):
    def on_episode_end(self, episode, logs=None):
        print(f"Episode {episode} finished with reward {logs['total_reward']}")

class EvaluationCallback(Callback):
    def __init__(self, env, agent, interval=10, episodes=5):
        self.env = env
        self.agent = agent
        self.interval = interval
        self.eval_episodes = episodes

    def on_episode_end(self, episode, logs=None):
        if episode % self.interval == 0:
            truncated, frames, env = PolicyEvaluator(env=self.env, agent=self.agent, record_dir=f"videos/inter_train_eval").evaluate(num_eval_episodes=10, file_name=f"ep_{episode}_eval.mp4")
            # return truncated, frames, env
            average_episode_length = np.average(env.episode_lengths)
            print(f"{average_episode_length=}")
            return average_episode_length
        return 
    
class EpsilonDecayCallback(Callback):
    def __init__(self, agent, decay=0.99, min_eps=0.05):
        self.agent = agent
        self.decay = decay
        self.min_eps = min_eps

    def on_episode_end(self, episode, logs=None):
        self.agent.epsilon = max(self.agent.epsilon * self.decay, self.min_eps)
        print(f"Episode {episode}: epsilon={self.agent.epsilon:.3f}")
