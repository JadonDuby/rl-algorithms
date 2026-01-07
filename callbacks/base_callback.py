from dataclasses import dataclass
from typing import Any

@dataclass
class CallbackContext:
    env: Any
    agent: Any | None = None


class Callback:
    """Base class for RL training callbacks."""
    def __init__(self, ctx: CallbackContext):
        self.ctx = ctx
    def on_train_begin(self, logs=None):
        """Called at the start of training."""
        pass

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        pass

    def on_episode_begin(self, episode, logs=None):
        """Called at the beginning of each episode."""
        pass

    def on_episode_end(self, episode, logs=None):
        """Called at the end of each episode."""
        pass

    def on_step_end(self, step, logs=None):
        """Called at the end of each environment step."""
        pass
