import logging

import gymnasium as gym
from features.feature_engineering import FeatureEngineer

from agents.linear_agent import LinearFunctionApproxAgent
from train.trainer import train

from callbacks.callback import PrintRewardCallback, EvaluationCallback

from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from evaluation.policy_evaluator import PolicyEvaluator
import imageio

from exploration.decay_schedules import *
import mlflow

def main():
    # with mlflow.start_run:
    truncated=False
    i = 0
    fig, ax = plt.subplots(2,2)
    ax = ax.flat
    while not truncated and i < 5:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        fe = FeatureEngineer(basis='poly', degree=2)
        n_episodes = 120
        schedule = ExponentialDecaySchedule()
        schedule = SigmoidSchedule(n_episodes=n_episodes)
        schedule = LinearSchedule(n_episodes=n_episodes)
        schedule = LinearStepSchedule(n_episodes=n_episodes)
        scheduler = DynamicStepSchedule(eval_callback=lambda: EvaluationCallback(env=env, agent=agent))
        # schedule = SawToothSchedule(n_episodes=n_episodes):
        
        agent = LinearFunctionApproxAgent(
            env.action_space,
            feature_engineer=fe,
            epsilon_schedule=schedule,
            alpha=0.99,
            gamma=0.6
        )
        rewards = train(
            env,
            agent,
            n_episodes=n_episodes,
            callbacks=[PrintRewardCallback(), EvaluationCallback(env=env, agent=agent), ]
        )
        
        # print(max(rewards), rewards.index(max(rewards)))
        evaluator = PolicyEvaluator(
            env,
            agent,
            record_dir="./videos",
        )
        
        truncated, frames, eval_env = evaluator.evaluate(num_eval_episodes=10, record=True, file_name=f"videos/policy_{i}.mp4")
        # imageio.mimsave(f"videos/policy_{i}.mp4", frames, fps=60)
        # ax[0].scatter(rewards, range(0,len(rewards))) # test metric is number of frames the episode lasted. 500 is success
        ax[1].plot(eval_env.length_queue)
        ax[2].plot(agent.epsilon_curve)
        # ax[3].plot(max(rewards))
        i += 1
    fig.savefig("videos/graphs.jpg")
if __name__ == "__main__":
    main()


