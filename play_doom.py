from vizdoom import *
import numpy as np
from PIL import Image
import random
import time
import frame
from frame import FrameStack
from skimage import img_as_ubyte


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

def save_test_image(image_data):
    image = Image.fromarray(image_data)
    image.save('debug/frame.png')
    image.show()

def test_scenario(environment):
    first = True
    for i in range(10):
        environment.new_episode()
        while not environment.is_episode_finished():
            state = environment.get_state()
            img = state.screen_buffer
            action = random.choice(actions)
            reward = environment.make_action(action)
            time.sleep(1/60)
            if first:
                img = img_as_ubyte(frame.preprocess(img))
                save_test_image(img)
                first = False

stack = FrameStack(size=4)

test_scenario(environment)