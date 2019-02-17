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
from networks import DeepQNetworkSimple
from learning import Agent
from learning import ReplayMemory


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


def test_scenario(environment, actions):
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


def learn_online(environment, actions, stack, num_episodes):
    num_actions = len(actions)
    dqn = DeepQNetworkSimple([84, 84, 4], num_actions, 0.0002)
    with tf.Session() as session:
        agent = Agent(
            dqn,
            session,
            actions,
            board_directory='simple',
            model_path='debug/models/model_simple.ckpt',
            restore_model=False)
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
                    agent.train(state, action, reward, next_state, t, terminal=True)
                    t = 100
                    reward_total = np.sum(rewards)
                    print('episode {}: total reward: {}, loss: {:.4f}'.format(e, reward_total, agent.loss))
                else:
                    stack.add(environment.get_state().screen_buffer)
                    next_state = stack.as_state()
                    agent.train(state, action, reward, next_state, t)
                    state = next_state


def init_replay_memory(environment, actions, stack, replay_memory_capacity, num_samples):
    memory = ReplayMemory(replay_memory_capacity)
    environment.new_episode()
    stack.init_new(environment.get_state().screen_buffer)
    state = stack.as_state()
    i = 0

    while i < num_samples:
        i += 1
        action = random.choice(actions)
        reward = environment.make_action(action)
        done = environment.is_episode_finished()

        if done:
            stack.add(np.zeros((84, 84), dtype=np.int), process=False)
            next_state = stack.as_state()
            memory.add((state, action, reward, next_state, done))
            environment.new_episode()
            stack.init_new(environment.get_state().screen_buffer)
            state = stack.as_state()
        else:
            stack.add(environment.get_state().screen_buffer)
            next_state = stack.as_state()
            memory.add((state, action, reward, next_state, done))
            state = next_state

    return memory

# replay_memory = init_replay_memory(environment, 1000000, 64)
# print(len(replay_memory.data))


def learn_batch(environment, actions, stack, num_episodes):
    num_actions = len(actions)
    dqn = DeepQNetworkBatch([84, 84, 4], num_actions, 0.0002)
    replay_memory = init_replay_memory(environment, 1000000, 64)

    with tf.Session() as session:
        agent = Agent(
            dqn,
            session,
            actions,
            board_directory='batch',
            model_path='debug/models/model_batch.ckpt',
            restore_model=False)
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
                    replay_memory.add((state, action, reward, next_state, done))
                    t = 100
                    reward_total = np.sum(rewards)
                    print('episode {}: total reward: {}, loss: {:.4f}'.format(e, reward_total, agent.loss))
                else:
                    stack.add(environment.get_state().screen_buffer)
                    next_state = stack.as_state()
                    replay_memory.add((state, action, reward, next_state, done))
                    state = next_state

                agent.train_batch(replay_memory.sample(64), e)


def play_as_human():
    game = DoomGame()
    game.add_game_args("+freelook 1")
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_available_buttons([
        Button.ATTACK, Button.SPEED, Button.STRAFE, Button.USE,
        Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.MOVE_BACKWARD, Button.MOVE_FORWARD,
        Button.TURN_LEFT, Button.TURN_RIGHT,
        Button.SELECT_WEAPON1, Button.SELECT_WEAPON2, Button.SELECT_WEAPON3,
        Button.SELECT_WEAPON4, Button.SELECT_WEAPON5, Button.SELECT_WEAPON6,
        Button.SELECT_NEXT_WEAPON, Button.SELECT_PREV_WEAPON,
        Button.LOOK_UP_DOWN_DELTA, Button.TURN_LEFT_RIGHT_DELTA , Button.MOVE_LEFT_RIGHT_DELTA])
    game.add_available_game_variable(GameVariable.AMMO2)
    game.set_mode(Mode.SPECTATOR)
    game.init()
    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            s = game.get_state()
            game.advance_action()
            a = game.get_last_action()
            r = game.get_last_reward()

    game.close()

'''
stack = FrameStack(size=4)
environment, actions_available = setup_scenario_basic()
# learn_batch(environment, 5000)
learn_online(environment, actions_available, stack, 1000)
'''


play_as_human()

'''
def play(environment, num_episodes):
    with tf.Session() as session:
        agent = Agent(
            dqn,
            session,
            actions_available,
            model_path='debug/models/model_batch.ckpt',
            restore_model=True)
        # session.run(tf.global_variables_initializer())
        environment.init()

        for e in range(num_episodes):
            t = 0
            environment.new_episode()
            stack.init_new(environment.get_state().screen_buffer)
            state = stack.as_state()

            while True:
                t += 1
                action = agent.get_policy_action(state)
                environment.make_action(action)
                done = environment.is_episode_finished()

                if done:
                    stack.add(np.zeros((84, 84), dtype=np.int), process=False)
                    time.sleep(1 / 60)
                    break
                else:
                    stack.add(environment.get_state().screen_buffer)
                    next_state = stack.as_state()
                    state = next_state
                    time.sleep(1 / 60)

play(environment, 1000)
'''