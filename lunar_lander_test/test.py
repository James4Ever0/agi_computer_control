import gymnasium as gym
# import pygame

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    # let's check.
    # pygame.image.save(env.screen, "image.png")  # type:ignore
    # exit()

    # observation: (8,)
    # reward: float value

    # no image?

    if terminated or truncated:
        observation, info = env.reset()

env.close()
