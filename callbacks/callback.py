from callbacks.base_callback import Callback
from evaluation.policy_evaluator import PolicyEvaluator
import numpy as np
import matplotlib.pyplot as plt



class PrintRewardCallback(Callback):
    def on_episode_end(self, episode, logs=None):
        print(f"Episode {episode} finished with reward {logs['total_reward']}")

class EvaluationCallback(Callback):
    def __init__(self, env, agent, policy):
        self.env = env
        self.agent = agent
        self.policy = policy

    def set_trainer(self, trainer):
        self.trainer = trainer

    # def on_episode_end(self, ep, episode_duration, interval = 100, eval_episodes = 5, step=None, logs=None):
    def on_episode_end(self, ctx, interval = 50, eval_episodes = 3):
        ep=ctx.episode
        if ep % interval == 0:
            # evaluator = PolicyEvaluator(env=self.env, agent=self.agent, record_dir=f"eval/policy_{self.policy}/inter_train_eval/")
            evaluator = PolicyEvaluator(env=self.env, agent=self.agent, record_dir=f"eval/policy_{self.policy}/inter_train_eval/")
            truncated, frames, env = evaluator.evaluate(num_eval_episodes=eval_episodes, file_name=f"ep_{ep}_eval.mp4")
            # return truncated, frames, env
            average_episode_length = np.average(env.episode_lengths)
            print(f"{average_episode_length=}")
            return {"average_episode_length": average_episode_length}
        return 
    
    def on_train_end(self, eval_episodes = 10, logs=None):
        print("starting final evaluation")
        evaluator = PolicyEvaluator(env=self.env, agent=self.agent, record_dir=f"eval/policy_{self.policy}/")
        truncated, frames, env = evaluator.evaluate(num_eval_episodes=eval_episodes, file_name=f"policy.mp4")
        average_episode_length = np.average(env.episode_lengths)
        return {"average_episode_length": average_episode_length}

class EpsilonDecayCallback(Callback):
    def __init__(self, agent, decay=0.99, min_eps=0.05):
        self.agent = agent
        self.decay = decay
        self.min_eps = min_eps

    def on_episode_end(self, episode, logs=None):
        self.agent.epsilon = max(self.agent.epsilon * self.decay, self.min_eps)
        print(f"Episode {episode}: epsilon={self.agent.epsilon:.3f}")

class UpdateEpsilonCallback(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.step = 1/scheduler.n_episodes
    
    def on_episode_end(self):
        self.scheduler.epsilon -= self.step

class EpsilonLoggerCallback(Callback):
    def __init__(self, path_prefix, scheduler):
        self.history = []
        self.path_prefix = path_prefix 
        self.scheduler = scheduler

    def set_trainer():
        return 

    # def on_episode_end(self, ep, step, episode_duration, episode=None, logs=None):
    def on_episode_end(self, ctx):
        self.history.append(self.scheduler.get_epsilon(step = ctx.step, ep = ctx.episode))
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.clf()
        plt.plot(self.history)
        plt.title("Epsilon Decay")
        plt.xlabel("Step")
        plt.ylabel("Epsilon")
        plt.savefig(f"{self.path_prefix}/epsilon_curve.jpg")

    def on_train_end(self, trainer=None, logs=None):
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.clf()
        plt.plot(self.history)
        plt.title("Epsilon Decay")
        plt.xlabel("Step")
        plt.ylabel("Epsilon")
        plt.savefig(f"{self.path_prefix}/epsilon_curve.jpg")

class PlotDurationsCallback(Callback):
    def __init__(self):
        return 
    
    # def on_episode_end(self, step, ep, episode_durations, logs=None):
    def on_episode_end(self, ctx):
        
        plt.figure(1)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        window_size = 10
        episode_durations = ctx.episode_durations
        if len(episode_durations) > window_size:
            # print(f"{len(durations_t)=}")
            weights = np.ones(window_size) / window_size
            sma = np.convolve(episode_durations, weights, mode='same')
            plt.plot(range(0,len(episode_durations)), sma) # test metric is number of frames the episode lasted. 500 is success
            plt.plot(episode_durations)
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.savefig("eval/duration.png")

class PlotRewardsCallback(Callback):
    def __init__(self):
        return 
    
    # def on_episode_end(self, step, ep, episode_durations, logs=None):
    def on_episode_end(self, ctx):
        plt.figure(1)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('reward')
        window_size = 10
        rewards = ctx.rewards
        print(f"info passed to rewards callback on episode end: {ctx.info}")
        if len(rewards) > window_size:
            # print(f"{len(durations_t)=}")
            weights = np.ones(window_size) / window_size
            sma = np.convolve(rewards, weights, mode='same')
            # plt.plot(range(0,len(rewards)), sma) # test metric is number of frames the episode lasted. 500 is success
            plt.plot(rewards)
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.savefig("eval/reward.png")

class MetricsCallback:
    def __init__(self):
        self.episode_durations = []
        self.episode_rewards = []
        self.losses = []
        
        # running trackers for the current episode:
        self._ep_reward = 0
        self._ep_len = 0
        self._ep_epsilon = []

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, logs ):
        return 

    def on_step_end(self, ctx):
        self._ep_reward += ctx.reward
        self._ep_len += 1
        self._ep_epsilon.append(ctx.epsilon)
        loss = ctx.extra["loss"]
        self.losses.append(loss.cpu().item()) if loss else None

    def on_episode_end(self, ctx):
        self.episode_durations.append(self._ep_len)
        self.episode_rewards.append(self._ep_reward)

        # Attach metrics back into callback context
        # ctx.extra["episode_durations"] = self._ep_len
        # ctx.extra["episode_reward"] = self._ep_reward

        self._plot_durations(self.episode_durations)
        self._plot_rewards(self.episode_rewards)
        self._plot_epsilon(self._ep_epsilon)
        self._plot_losses(self.losses)

        self._ep_len = 0
        self._ep_reward = 0

        return {"episode_reward": self._ep_reward, "episode_length": self._ep_len}
    
    def _plot_rewards(self, rewards):
        
        plt.figure(1)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('reward')
        window_size = 10
        if len(rewards) > window_size:
            # print(f"{len(durations_t)=}")
            weights = np.ones(window_size) / window_size
            sma = np.convolve(rewards, weights, mode='same')
            # plt.plot(range(0,len(rewards)), sma) # test metric is number of frames the episode lasted. 500 is success
            plt.plot(rewards)
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.savefig("eval/reward.png")

    def _plot_epsilon(self, epsilons):
        plt.figure(2)
        plt.clf()
        plt.plot(epsilons)
        plt.title("Epsilon Decay")
        plt.xlabel("Step")
        plt.ylabel("Epsilon")
        plt.savefig(f"eval/epsilon_curve.jpg")

    def _plot_losses(self, losses):
        plt.figure(3)
        plt.clf()
        plt.plot(losses)
        plt.title("Loss")
        plt.xlabel("Step")
        plt.ylabel("Epsilon")
        plt.savefig(f"eval/loss_curve.jpg")

    def _plot_durations(self, episode_durations):        
        plt.figure(1)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        window_size = 10
        if len(episode_durations) > window_size:
            # print(f"{len(durations_t)=}")
            weights = np.ones(window_size) / window_size
            sma = np.convolve(episode_durations, weights, mode='same')
            plt.plot(range(0,len(episode_durations)), sma) # test metric is number of frames the episode lasted. 500 is success
            plt.plot(episode_durations)
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.savefig("eval/duration.png")