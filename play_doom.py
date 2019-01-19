from vizdoom import *

import random
import time

def setup_scenario_basic():
    game = DoomGame()
    game.load_config("vizdoom/basic.cfg")
    game.set_doom_scenario_path("vizdoom/basic.wad")
    game.init()
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    actions = [left, right, shoot]

    return game, actions

environment, actions = setup_scenario_basic()

def test_scenario(environment):
    for i in range(10):
        environment.new_episode()
        while not environment.is_episode_finished():
            action = random.choice(actions)
            reward = environment.make_action(action)
            time.sleep(0.02)

test_scenario(environment)