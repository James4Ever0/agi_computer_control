# import gym
# import universe  # register the universe environments
# import universe.wrappers.experimental

# env = gym.make("gym-core.PongDeterministic-v0")
# env = universe.wrappers.experimental.SafeActionSpace(env)
# env.configure(remotes=1)

# observation_n = env.reset()

# while True:
#     action_n = [env.action_space.sample() for ob in observation_n]
#     observation_n, reward_n, done_n, info = env.step(action_n)
#     env.render()

# this sucks.
import asyncio, asyncvnc
from credentials import username, password
# # you must use your computer password here.
ipaddr = "192.168.10.6"
import rich
rich.print(asyncvnc.key_codes)
# async def run_client():
#     async with asyncvnc.connect(ipaddr, 5900, username, password) as client:
#         client.keyboard.press
#         client.keyboard.write("hello world!")


# asyncio.run(run_client())
