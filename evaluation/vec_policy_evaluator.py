import os
import numpy as np
# from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.wrappers.vector import RecordEpisodeStatistics

import imageio 

class VecPolicyEvaluator:
    def __init__(self, env, agent, record_dir=None, file_name = None, render_mode="rgb_array"):
        self.env = env
        self.agent = agent
        self.record_dir = record_dir
        self.render_mode = render_mode

    def evaluate(self, num_eval_episodes=5, record=True, file_name = None):
        env = self.env
        print(f"Starting evaluation for {num_eval_episodes} episodes...")
        print(f"Videos will be saved to: {self.record_dir}{file_name}")

        frames = []
        for episode_num in range(num_eval_episodes):
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0

            done = np.array([False])
            while step_count < 500:
                if record:
                    frame = env.render()
                    frames.append(frame[0])
                action = self.agent.select_action(obs, epsilon=0)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

            print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}")

        # env.close()

        # Print summary statistics
        print(f'\nEvaluation Summary:')
        print(f'Episode durations: {list(env.time_queue)}')
        print(f'Episode rewards: {list(env.return_queue)}')
        print(f'Episode lengths: {list(env.length_queue)}')

        # Calculate some useful metrics
        avg_reward = np.mean(env.return_queue)
        avg_length = np.mean(env.length_queue)
        std_reward = np.std(env.return_queue)

        print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
        print(f'Average episode length: {avg_length:.1f} steps')
        print(file_name)
        frames = [f for f in frames if f is not None]
        imageio.mimsave(f"{self.record_dir}/{file_name}", frames, fps=60)
        return truncated, frames, env

