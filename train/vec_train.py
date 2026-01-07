import numpy as np
from core.env_interface import BaseEnv, BaseVecEnv

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
    ep_len: Optional[list] = None
    rewards: Optional[list] = None
    info: Optional[dict] = None
    extra: dict = field(default_factory=dict)


class VecTrainer:
    def __init__(self, envs, agent, scheduler, callbacks=[]):
        self.envs = envs
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

    
    def train(self, num_steps):
        # for cb in self.callbacks:
        #     cb.on_train_begin()
        global device
        step = 0
        ep = 0
        # ctx = CallbackContext()
        # for cb in self.callbacks:
        #     cb.on_episode_begin(ctx)
        state, info = self.envs.reset()
        while num_steps > 0:
            for t in count():
                num_steps -= 1
                step += 1
                epsilon = self.scheduler.get_epsilon(step=step)
                action = self.agent.select_action(state, epsilon)
                observation, reward, terminated, truncated, info = self.envs.step(action)
                dones = terminated | truncated
                next_state = observation
                # Perform one step of the optimization (on the policy network)
                loss = self.agent.update(state, action, next_state, reward, dones)
                step_ctx = CallbackContext(
                    step=step,
                    reward=reward,
                    epsilon=epsilon,
                    ep_len=t,
                    extra={'loss':loss}
                )
                for cb in self.callbacks:
                    cb.on_step_end(step_ctx)

                # Move to the next state
                state = next_state

                self.scheduler.update_epsilon()
                ctx = CallbackContext(
                        step=step,
                        reward=reward,
                        epsilon=epsilon,
                        ep_len=t,
                        episode=ep,
                        info=info
                    )
                if dones.any():
                    for cb in self.callbacks:
                        print(f"{self.callbacks}=")
                        ep += dones.sum()
                        # reuse the last context. or else we lost the most recent step
                        # print(f"{info=}")
                        response = cb.on_episode_end(ctx)
                        try:
                            self.logs.update(response)
                        except:
                            continue
        ctx = CallbackContext(
            step=step,
            epsilon=epsilon,
            # ep_len=episode_durations,
            reward=reward,
            # rewards=rewards,
            trainer=self
        )
        for cb in self.callbacks:
            cb.on_train_end()
        # return episode_durations
