from agents.base_agent import BaseAgent
import numpy as np
from collections import namedtuple, deque 
import math 
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from components.replay_buffer import ReplayBuffer
from gymnasium.vector import SyncVectorEnv
import gymnasium as gym

# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
# device = torch.device("cpu")

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
   
class VectorDeepQLearningAgent(BaseAgent):
    def __init__(self, env, epsilon_schedule):  
        print("envs:")
        print(env)
        self.total_steps = 0
        self.envs = env
        self.num_envs = self.envs.num_envs
        self.schedule = epsilon_schedule
        self.action_space = self.envs.action_space    
        self.n_actions = self.action_space.nvec[0]
        # self.n_actions = self.envs.single_action_space
        self.BATCH_SIZE = 256
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.01
        self.EPS_DECAY = 2500
        self.TAU = 0.005
        self.LR = 6e-4
        state, info = self.envs.reset()
        n_observations = state.shape[-1]
        self.policy_net = DQN(n_observations, self.n_actions).to(device)
        self.target_net = DQN(n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayBuffer(100_000)
        self.step = 0

    def preprocess_state(self, state):
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    @property
    def is_vectorized(self):
        return isinstance(self.envs, gym.vector.VectorEnv)


    def select_action(self, state, epsilon):
        # global steps_done
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        sample = random.random()
        # epsilon = self.schedule.get_epsilon()
        step = self.total_steps
        self.total_steps += 1
        # epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        #     math.exp(-1. * step / self.EPS_DECAY)
        # steps_done += 1
        if sample > epsilon:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.policy_net(state)
                if self.is_vectorized:
                    return np.array(self.policy_net(state).max(2).indices.cpu()).flatten()
                else:
                    return self.policy_net(state).max(1).indices.view(1, 1).item()
        else:
            if self.is_vectorized:
                return np.array(self.envs.action_space.sample())
            else:
                return torch.tensor(self.envs.action_space.sample(), device=device, dtype=torch.long)
    
    def update(self, state, action, next_state, reward, done):

        for i in range(self.num_envs):
            self.memory.push(
                state[i],
                action[i],
                next_state[i],
                reward[i],
                done[i]
            )
        print(f"{len(self.memory)=}")
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = self.memory.sample(self.BATCH_SIZE)
        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.reward
        done_batch = batch.done
        next_state_batch = batch.next_state

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(batch.state).gather(1, batch.action)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_q_values = self.target_net(batch.next_state)
            next_state_values = next_q_values.max(1, keepdim=True).values
            next_state_values = next_state_values * (1 - batch.done)
            target_q_values = batch.reward + (self.GAMMA * next_state_values)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, target_q_values)

        # debugging
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 100 == 0:
            done_sum = batch.done.sum().item()
            with torch.no_grad():
                sample_q = self.policy_net(torch.randn(1, state_batch.shape[1], device=device))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        if self.step % 10_000:
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.step += 1
        self.target_net.load_state_dict(target_net_state_dict)
        return loss
