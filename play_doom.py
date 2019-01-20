from vizdoom import *
import numpy as np
from PIL import Image
import random
import time
from skimage import img_as_ubyte
import tensorflow as tf

import frame
from frame import FrameStack
from networks import DeepQNetworkBatch
from learning import Agent

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
            action = random.choice(actions_available)
            reward = environment.make_action(action)
            time.sleep(1/60)
            if first:
                img = img_as_ubyte(frame.preprocess(img))
                save_test_image(img)
                first = False



stack = FrameStack(size=4)

environment, actions_available = setup_scenario_basic()
#test_scenario(environment)
#exit(1)

num_actions = len(actions_available)
dqn = DeepQNetworkBatch([84, 84, 4], num_actions, 0.0002)


def learn_online(environment, num_episodes):
    with tf.Session() as session:
        agent = Agent(dqn, session, actions_available)
        session.run(tf.global_variables_initializer())
        environment.init()

        for e in range(num_episodes):
            t = 0
            rewards = []
            environment.new_episode()
            stack.init_new(environment.get_state().screen_buffer)
            state = stack.as_state()

            while t < 100:
                t += 1
                action = agent.get_policy_action(state)
                reward = environment.make_action(action)
                rewards.append(reward)
                done = environment.is_episode_finished()

                if done:
                    stack.add(np.zeros((84, 84), dtype=np.int), process=False)
                    next_state = stack.as_state()
                    agent.train(state, action, reward, next_state, terminal=True)
                    t = 100
                    reward_total = np.sum(rewards)
                    print('episode {}: total reward: {}, loss: {:.4f}'.format(e, reward_total, agent.loss))

                else:
                    stack.add(environment.get_state().screen_buffer)
                    next_state = stack.as_state()
                    agent.train(state, action, reward, next_state)
                    state = next_state



learn_online(environment, 1000000)