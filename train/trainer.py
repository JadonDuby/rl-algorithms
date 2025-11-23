import numpy as np
import torch 
from itertools import count
from typing import Any, Optional
from dataclasses import dataclass, field

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

@dataclass
class CallbackContext:
    episode: Optional[int] = None
    step: Optional[int] = None
    epsilon: Optional[float] = None
    reward: Optional[float] = None
    trainer: Optional[Any] = None
    episode_durations: Optional[list] = None
    rewards: Optional[list] = None
    extra: dict = field(default_factory=dict)


class Trainer:
    def __init__(self, env, agent, scheduler, callbacks=[]):
        self.env = env
        self.agent = agent
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.logs = {}

        for cb in self.callbacks:
            try:
                cb.set_trainer(self)
            except:
                print(f'callback {cb} does not have a set_trainer method')
                pass

    def train(self):
        for cb in self.callbacks:
            cb.on_train_begin(logs={})

        rewards = []
        ep = 0
        epsilon = self.scheduler.get_epsilon
        while self.scheduler.epsilon > 0:
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                epsilon = self.scheduler.get_epsilon()
                action = self.agent.select_action(obs, epsilon)
                next_obs, reward, done, _, _ = self.env.step(action)
                self.agent.update(obs, action, reward, next_obs, done, ep)
                obs = next_obs
                total_reward += reward
            rewards.append(total_reward)
            # print(f"Episode {ep}: {total_reward}")

            for cb in self.callbacks:
                response = cb.on_episode_end(ep, logs=self.logs)
                try:
                    self.logs.update(response)
                except:
                    continue

            if "new_scheduler" in self.logs:
                self.scheduler = self.logs["new_scheduler"]
            ep += 1
            self.scheduler.update_epsilon()
        for cb in self.callbacks:
            response = cb.on_training_end(logs=self.logs)
            print(response)
        return rewards
    
    
    def train_dqn(self, num_episodes):
        for cb in self.callbacks:
            cb.on_train_begin()
        global device
        # if torch.cuda.is_available() or torch.backends.mps.is_available():
        #     num_episodes = 100
        # else:
        #     num_episodes = 50
        episode_durations = []
        rewards = []
        step = 0
        for ep in range(num_episodes):
            
            # Initialize the environment and get its state
            state, info = self.env.reset()
            # state = self.agent.preprocess_state(state)
            for t in count():
                step += 1
                epsilon = self.scheduler.get_epsilon(ep=ep, step=step)
                action = self.agent.select_action(state, epsilon)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                print(f"{reward=}")
                # reward = torch.tensor([reward], device =device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    # next_state = self.agent.preprocess_state(observation)
                    next_state = observation

                # # Store the transition in memory
                # self.agent.memory.push(state, action, next_state, reward, done)

                # Perform one step of the optimization (on the policy network)
                self.agent.update(state, action, next_state, reward, done)

                # # Soft update of the target network's weights
                # # θ′ ← τ θ + (1 −τ )θ′
                # target_net_state_dict = self.agent.target_net.state_dict()
                # policy_net_state_dict = self.agent.policy_net.state_dict()
                # for key in policy_net_state_dict:
                #     target_net_state_dict[key] = policy_net_state_dict[key]*self.agent.TAU + target_net_state_dict[key]*(1-self.agent.TAU)
                # self.agent.target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    rewards.append(reward)
                    ctx = CallbackContext(
                        episode=ep,
                        step=step,
                        epsilon=epsilon,
                        episode_durations=episode_durations,
                        reward=reward, 
                        rewards=rewards
                    )
                    break

                # Move to the next state
                state = next_state

            self.scheduler.update_epsilon()
            for cb in self.callbacks:
                # response = cb.on_episode_end(ep=ep, step=step, episode_durations=episode_durations, logs=self.logs)
                # reuse the last context. or else we lost the most recent step
                response = cb.on_episode_end(ctx)
                try:
                    self.logs.update(response)
                except:
                    continue
        ctx = CallbackContext(
            episode=ep,
            step=step,
            epsilon=epsilon,
            episode_durations=episode_durations,
            reward=reward,
            rewards=rewards,
            trainer=self
        )
        for cb in self.callbacks:
            cb.on_train_end()
        return episode_durations

        # print('Complete')
        # plot_durations(show_result=True)
        # plt.ioff()
        # plt.show()