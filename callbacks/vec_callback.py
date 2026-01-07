from callbacks.base_callback import Callback, CallbackContext
from evaluation.vec_policy_evaluator import VecPolicyEvaluator
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


class VecEvaluationCallback(Callback):
    def __init__(self, ctx: CallbackContext):
        super().__init__(ctx)
        self.env = self.ctx.env
        self.agent = self.ctx.agent

    def set_trainer(self, trainer):
        self.trainer = trainer

    # def on_episode_end(self, ep, episode_duration, interval = 100, eval_episodes = 5, step=None, logs=None):
    def on_episode_end(self, ctx, interval = 50, eval_episodes = 1):
        ep=ctx.episode
        if ep % interval == 0 and ep!=0:
            # evaluator = PolicyEvaluator(env=self.env, agent=self.agent, record_dir=f"eval/policy_{self.policy}/inter_train_eval/")
            evaluator = VecPolicyEvaluator(env=self.env, agent=self.agent, record_dir=f"eval/vids/")
            truncated, frames, env = evaluator.evaluate(num_eval_episodes=eval_episodes, file_name=f"ep_{ep}_eval.mp4")
            # return truncated, frames, env
            average_episode_length = np.average(env.episode_lengths)
            print(f"{average_episode_length=}")
            return {"average_episode_length": average_episode_length}
        return

class VecMetricsCallback:
    def __init__(self, ctx):
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
        # self._ep_reward += ctx.reward
        # self._ep_len += 1
        loss = ctx.extra["loss"]
        self._ep_epsilon.append(ctx.epsilon)
        self.losses.append(loss.cpu().item()) if loss else None

    def on_episode_end(self, ctx):
        info = ctx.info
        idxs = np.argwhere(info["_episode"]>0).ravel()
        for idx in idxs:
            episode_len = info["episode"]["l"][idx].item()
            episode_reward = info["episode"]["r"][idx].item()
            self.episode_durations.append(episode_len)
            self.episode_rewards.append(episode_reward)
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
        plt.figure(2)
        plt.clf()
        plt.plot(losses)
        plt.title("Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(f"eval/loss_curve.jpg")

    def _plot_durations(self, episode_durations):        
        plt.figure(2)
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