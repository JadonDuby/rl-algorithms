import random
from collections import namedtuple
import numpy as np
import torch

# Define transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """A simple replay buffer that stores transitions and converts them to tensors at sampling time."""

    def __init__(self, capacity, device='mps'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device

    def push(self, state, action, next_state, reward, done):
        """Saves a transition (keeps Python/numpy types — converts to tensor later)."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        # print(f"{batch=}")
        # print(f"{batch.state=}")
        # Find the shape of a valid state
        state_shape = np.array([s for s in batch.next_state if s is not None][0]).shape
        
        # Replace None with a zero array of the same shape
        next_state_batch = torch.tensor(
            np.array([
                s if s is not None else np.zeros(state_shape, dtype=np.float32)
                for s in batch.next_state
            ]),
            dtype=torch.float32,
            device=self.device
        )

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        return Transition(state_batch, action_batch, next_state_batch, reward_batch, done_batch)


    def __len__(self):
        return len(self.memory)