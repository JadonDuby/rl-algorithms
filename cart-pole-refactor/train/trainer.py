import numpy as np

def train(env, agent, n_episodes=100, callbacks=[]):
    for cb in callbacks:
        cb.on_train_begin(logs={})

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, done, ep)
            obs = next_obs
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {ep}: {total_reward}")

        for cb in callbacks:
            cb.on_episode_end(
                ep,
                logs={"total_reward": total_reward}
            )
    return 

# def train(env, agent, n_episodes, callbacks=[]):
#     # Call training begin
#     for cb in callbacks:
#         cb.on_train_begin(logs={})

#     for episode in range(n_episodes):
#         obs, info = env.reset()
#         done = False
#         total_reward = 0
#         step = 0

#         for cb in callbacks:
#             cb.on_episode_begin(episode, logs={})

#         while not done:
#             action = agent.select_action(obs)
#             obs, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             agent.update(obs, action, reward, done)

#             total_reward += reward
#             step += 1

#             for cb in callbacks:
#                 cb.on_step_end(step, logs={"reward": reward})

#         for cb in callbacks:
#             cb.on_episode_end(
#                 episode,
#                 logs={"total_reward": total_reward, "steps": step}
#             )

#     for cb in callbacks:
#         cb.on_train_end(logs={})
# def train(env, agent, episodes, callbacks=[]):
    # Call training begin
    print(f"{callbacks=}")
    for cb in callbacks:
        cb.on_train_begin(logs={})

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0

        for cb in callbacks:
            cb.on_episode_begin(episode, logs={})

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, done)

            total_reward += reward
            step += 1

            for cb in callbacks:
                cb.on_step_end(step, logs={"reward": reward})

        for cb in callbacks:
            cb.on_episode_end(
                episode,
                logs={"total_reward": total_reward, "steps": step}
            )

    for cb in callbacks:
        cb.on_train_end(logs={})


