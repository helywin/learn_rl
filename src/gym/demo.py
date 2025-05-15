import gymnasium
env = gymnasium.make("CartPole-v1",  render_mode="human")
env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward, terminated, truncated, info)
    # if terminated or truncated:  # 检查是否需要重置环境
    #     observation, info = env.reset()

env.close()