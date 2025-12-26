import logging

import gymnasium as gym
from features.feature_engineering import FeatureEngineer

from agents.linear_agent import LinearFunctionApproxAgent
from agents.dqn_agent import DeepQLearningAgent
from agents.VectorDqn_agent import VectorDeepQLearningAgent
from train.trainer import Trainer
from train.vec_train import VecTrainer

from callbacks.callback import EvaluationCallback, UpdateEpsilonCallback, EpsilonLoggerCallback, PlotDurationsCallback, PlotRewardsCallback, MetricsCallback
from callbacks.vec_callback import VecMetricsCallback, VecEvaluationCallback


# from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.wrappers.vector import RecordEpisodeStatistics



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

    register(
    id='DoubleCartPole-v0',
    entry_point='envs.double_cart_pole:DoubleCartPole', # Replace with your module and class name
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
        # envs = gym.wrappers.vector.ClipReward(env, min_reward=0.2, max_reward=0.8)
        # envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")
        # SyncVectorEnv(..., autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
        num_envs=6
        # envs = gym.make_vec("CartPole-v1", num_envs=num_envs, vectorization_mode="sync", render_mode="rgb_array")
        envs = gym.make_vec("SwingUp-v0", num_envs=num_envs, vectorization_mode="sync", render_mode="rgb_array")
        envs = gym.make_vec("DoubleCartPole-v0", num_envs=num_envs, vectorization_mode="sync", render_mode="rgb_array")
        envs = RecordEpisodeStatistics(envs)

        # envs = gym.wrappers.vector.ClipReward(envs, mgitqin_reward=0.2, max_reward=0.8)

        # env = gym.make("LunarLander-v3", render_mode='rgb_array')
        # env = gym.make("Acrobot-v1", render_mode='rgb_array')
        # env = gym.make("BipedalWalker-v3", render_mode='rgb_array') # needs continuious actiion space support still
        # env = gym.make("SwingUp-v0", render_mode='rgb_array')
        # env = gym.make("InvertedDoublePendulum-v5")
        fe = FeatureEngineer(basis='poly', degree=2)
        n_episodes = 500
        schedule = ExponentialDecaySchedule(decay_rate=0.001)
        # schedule = SigmoidSchedule(n_episodes=n_episodes)
        # schedule = LinearSchedule(n_episodes=n_episodes)
        # schedule = LinearStepSchedule(n_episodes=n_episodes)    
        # scheduler = DynamicStepSchedule(eval_callback=lambda: EvaluationCallback(env=env, agent=agent))
        # schedule = SawToothSchedule(n_episodes=n_episodes):
        
        # agent = LinearFunctionApproxAgent(
        #     env.action_space,
        #     feature_engineer=fe,
        #     epsilon_schedule=schedule,
        #     alpha=0.999,
        #     gamma=0.2
        # )

        # agent = DeepQLearningAgent(
        #     env,
        #     schedule
        # )
        # callbacks = [
        #     EvaluationCallback(env=env, agent=agent, policy=i),
        #     MetricsCallback(),
        #     ]
        # trainer = Trainer(env, agent, schedule, callbacks=callbacks,)
        # durations = trainer.train(500)

        agent = VectorDeepQLearningAgent(
            envs,
            schedule
        )
        vec_callbacks = [
            VecEvaluationCallback(envs=envs, agent=agent, policy=i),
            VecMetricsCallback(),
            ]
        trainer = VecTrainer(envs, agent, schedule, callbacks=vec_callbacks,)
        durations = trainer.train(100_000_00)


        i += 1
if __name__ == "__main__":
    # print("ran")
    main()





