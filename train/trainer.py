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
    ep_len: Optional[list] = None
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

    
    def train(self, num_episodes):
        # for cb in self.callbacks:
        #     cb.on_train_begin()
        global device
        step = 0
        loss = None
        for ep in range(num_episodes):
            ctx = CallbackContext(episode=ep)
            # for cb in self.callbacks:
            #     cb.on_episode_begin(ctx)
            state, info = self.env.reset()
            for t in count():
                step += 1
                epsilon = self.scheduler.get_epsilon(ep=ep, step=step)
                action = self.agent.select_action(state, epsilon)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = observation
                # Perform one step of the optimization (on the policy network)
                loss = self.agent.update(state, action, next_state, reward, done)
                print(loss)
                step_ctx = CallbackContext(
                    episode=ep,
                    step=step,
                    reward=reward,
                    epsilon=epsilon,
                    ep_len=t,
                    extra={"loss":loss}
                )
                for cb in self.callbacks:
                    cb.on_step_end(step_ctx)
                if done:
                    break

                # Move to the next state
                state = next_state

            self.scheduler.update_epsilon()
            ctx = CallbackContext(
                    episode=ep,
                    step=step,
                    reward=reward,
                    epsilon=epsilon,
                    ep_len=t
                )
            for cb in self.callbacks:
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
            # ep_len=episode_durations,
            reward=reward,
            # rewards=rewards,
            trainer=self
        )
        for cb in self.callbacks:
            cb.on_train_end()
        # return episode_durations
