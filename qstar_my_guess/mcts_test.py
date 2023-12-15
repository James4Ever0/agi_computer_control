# any example to run?
# import dill
# import pickle
# pickle.dump = dill.dump
# pickle.load = dill.load

# if the viewer-producer model is probablistic or state-machine like, we can use linear programming.
# train multiple producers and viewers, selecting the most appropriate topics suitable for some part of the platform content.

import imageio
import base64

# import IPython
import gymnasium as gym
from mcts_simple import *
import sys

sys.setrecursionlimit(int(1e9))  # workaroud to pickle error.


class CartPole(Game):
    """
    The episode ends if any one of the following occurs:
        * Termination: Pole Angle is greater than ±12°
        * Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
        * Truncation: Episode length is greater than 500 (200 for v0)
    """

    def __init__(self):
        # self.env = gym.make("CartPole-v1", render_mode="human")
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.current_state, _ = self.env.reset()
        self.frames = []
        self.terminated, self.truncated = False, False

    def render(self):
        self.frames.append(self.env.render())
        if self.has_outcome():
            # IPython.display.display(
            #     IPython.display.HTML(
            #         data=f"""
            # <video controls src = "data:video/mp4;base64,{base64.b64encode(imageio.mimsave(
            # "<bytes>", self.frames, "MP4", fps = 20, **{"macro_block_size": None})).decode()}"></video>
            # """
            #     )
            # )
            self.frames.clear()

    def get_state(self):
        return self.current_state

    def number_of_players(self):
        return 1

    def current_player(self):
        return 0

    def possible_actions(self):
        return [i for i in range(self.env.action_space.n)]

    def take_action(self, action):
        if not self.has_outcome():
            self.current_state, _, self.terminated, self.truncated, _ = self.env.step(
                action
            )

    def has_outcome(self):
        return self.terminated or self.truncated

    def winner(self):
        # Noting that backprop code is: node.w += (prev_node.player in winners) / number_of_winners
        # It is possible to manipulate list size = self.env._max_episode_steps - self.env._elapsed_steps, since there will always be only 1 player
        # winner() will return a reward instead, where 0 <= reward <= 1, where it will increase exponentially as elapsed steps increase
        return [
            self.current_player()
            for _ in range(
                max(1, self.env._max_episode_steps - self.env._elapsed_steps + 1)
            )
        ]


game = CartPole()


tree = OpenLoopMCTS(game, training=True)
tree.self_play(iterations=1000)


tree.training = False
tree.self_play()


tree.save("cartpole.mcts")  # cannot save.
