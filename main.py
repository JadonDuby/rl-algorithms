import logging

import gymnasium as gym
from features.feature_engineering import FeatureEngineer

from agents.linear_agent import LinearFunctionApproxAgent
from agents.dqn_agent import DeepQLearningAgent
from train.trainer import Trainer

from callbacks.callback import EvaluationCallback, UpdateEpsilonCallback, EpsilonLoggerCallback, PlotDurationsCallback, PlotRewardsCallback

import matplotlib.pyplot as plt

from exploration.decay_schedules import *
import mlflow

import torch 

from gymnasium.envs.registration import register

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def main():
    register(
    id='SwingUp-v0',
    entry_point='envs.swing_up:SwingUpEnv', # Replace with your module and class name
    # max_episode_steps=100, # Optional: Set a default max episode steps
    )
    truncated=False
    i = 0
    fig, ax = plt.subplots(2,2)
    ax = ax.flat
    while not truncated and i < 1:
        fig, ax = plt.subplots(2,2)
        ax = ax.flat
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        # envs = gym.wrappers.vector.ClipReward(envs, min_reward=0.2, max_reward=0.8)
        # env = gym.make("LunarLander-v3", render_mode='rgb_array')
        # env = gym.make("Acrobot-v1", render_mode='rgb_array')
        # env = gym.make("BipedalWalker-v3", render_mode='rgb_array') # needs continuious actiion space support still
        # env = gym.make("SwingUp-v0", render_mode='rgb_array')
        # env = gym.make("InvertedDoublePendulum-v5")
        fe = FeatureEngineer(basis='poly', degree=2)
        n_episodes = 500
        # schedule = ExponentialDecaySchedule()
        # schedule = SigmoidSchedule(n_episodes=n_episodes)
        # schedule = LinearSchedule(n_episodes=n_episodes)
        schedule = LinearStepSchedule(n_episodes=n_episodes)    
        # scheduler = DynamicStepSchedule(eval_callback=lambda: EvaluationCallback(env=env, agent=agent))
        # schedule = SawToothSchedule(n_episodes=n_episodes):
        
        agent = LinearFunctionApproxAgent(
            env.action_space,
            feature_engineer=fe,
            epsilon_schedule=schedule,
            alpha=0.999,
            gamma=0.2
        )

        # agent = DeepQLearningAgent(
        #     env,
        #     schedule
        # )

        callbacks = [
            EvaluationCallback(env=env, agent=agent, policy=i),
            EpsilonLoggerCallback(f"eval/policy_{i}", scheduler = schedule),
            PlotDurationsCallback(),
            PlotRewardsCallback(),
            ]
        trainer = Trainer(env, agent, schedule, callbacks=callbacks)
        # rewards = trainer.train_2()
        durations = trainer.train_dqn(n_episodes)

        # window_size = 3
        # weights = np.ones(window_size) / window_size
        # sma = np.convolve(rewards, weights, mode='valid')
        # ax[0].plot(range(0,len(rewards)-2), sma) # test metric is number of frames the episode lasted. 500 is success
        # # ax[1].plot(eval_env.length_queue)
        # # ax[2].plot(agent.epsilon_curve)
        # # ax[3].plot(max(rewards))
        # print(f"{rewards=}")
        # fig.tight_layout()
        # fig.savefig(f"eval/policy_{i}/graphs.jpg")
        i += 1
if __name__ == "__main__":
    # print("ran")
    main()


