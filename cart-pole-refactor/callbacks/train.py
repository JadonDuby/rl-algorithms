def train(env, agent, episodes, callbacks=[]):
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
