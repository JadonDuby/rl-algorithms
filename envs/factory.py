import gymnasium as gym
from gymnasium.envs.registration import register
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
import logging
log = logging.getLogger(__name__)


def make_env(
    id: str,
    num_envs: int,
    seed: int | None = None,
    wrappers: list[str] | None = None,
    render_mode: str | None = None,
    entry_point: str | None = None,
):
    log.info(f"making environment with id: {id}")
    if entry_point:
        log.info("registering environment")
        register(id=id,entry_point=entry_point) 
    
    if num_envs > 1:
        env = gym.make_vec(id, num_envs=num_envs, vectorization_mode="async", render_mode=render_mode)
    else:
        env = gym.make(id, render_mode=render_mode)
    log.info("adding wrappers to env")
    if wrappers:
        for w in wrappers:
            log.info(f"env before wrapper: {w}")
            env = w(env)
            log.info(f"env after wrapper: {w}")
    log.info(f"returning env: {env}")
    return env


