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


episodes = 10
for i in range(episodes):
    environment.new_episode()
    while not environment.is_episode_finished():
        state = environment.get_state()
        img = state.screen_buffer
        misc = state.game_variables
        action = random.choice(actions)
        print(action)
        reward = environment.make_action(action)
        print("\treward:", reward)
        time.sleep(0.02)
    print("Result:", environment.get_total_reward())
