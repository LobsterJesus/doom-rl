import numpy as np
import random
import tensorflow as tf
from collections import deque


class Agent:

    def get_policy_action(self, state):
        if np.random.rand() > self.epsilon:
            q_values = self.session.run(self.dqn.output, feed_dict={self.dqn.inputs: state.reshape((1, *state.shape))})
            choice = np.argmax(q_values)
            return self.actions[int(choice)]
        else:
            return random.choice(self.actions)

    def get_q_values_batch(self, state_batch):
        return self.session.run(self.dqn.output, feed_dict={self.dqn.inputs: state_batch})

    def train(self, state, action, reward, next_state, time_step, terminal=False):
        q_values = self.session.run(self.dqn.output, feed_dict={self.dqn.inputs: state.reshape((1, *state.shape))})
        q_values_next = self.session.run(self.dqn.output, feed_dict={self.dqn.inputs: next_state.reshape((1, *next_state.shape))})

        target = q_values[0]
        for i, v in enumerate(action):
            if v == 1:
                target[i] = (reward + self.gamma * np.max(q_values_next)) if not terminal else reward

        if terminal:
            self.record_episode_statistics(state, target, action, reward, time_step)

        self.loss, _ = self.session.run(
            [self.dqn.loss, self.dqn.optimizer],
            feed_dict={
                self.dqn.inputs: state.reshape((1, *state.shape)),
                self.dqn.q_target: target,
                self.dqn.actions: [action]})

    def train_batch(self, batch, episode_index):
        states = np.array([sample[0] for sample in batch], ndmin=3)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_states = np.array([sample[3] for sample in batch], ndmin=3)
        terminals = np.array([sample[4] for sample in batch])
        q_values_next_state = self.get_q_values_batch(next_states)
        q_targets = []

        for i in range(0, len(batch)):
            done = terminals[i]
            if done:
                q_targets.append(rewards[i])
            else:
                q_targets.append(rewards[i] + self.gamma * np.max(q_values_next_state[i]))

        targets_dqn = np.array([each for each in q_targets])

        self.loss, _ = self.session.run(
            [self.dqn.loss, self.dqn.optimizer],
            feed_dict={
                self.dqn.inputs: states,
                self.dqn.q_target: targets_dqn,
                self.dqn.actions: actions})

        summary = self.session.run(
            self.merged_summary,
            feed_dict={
                self.dqn.reward: 0,
                self.dqn.inputs: states,
                self.dqn.q_target: targets_dqn,
                self.dqn.actions: actions})

        self.tf_writer.add_summary(summary, episode_index)
        self.tf_writer.flush()

        if episode_index % 5 == 0:
            self.tf_saver.save(self.session, self.model_path)

    def record_episode_statistics(self, state, target, action, reward, time_step):
        summary = self.session.run(
            self.merged_summary,
            feed_dict={
                self.dqn.inputs: state.reshape((1, *state.shape)),
                self.dqn.q_target: target,
                self.dqn.actions: [action],
                self.dqn.reward: reward})
        self.tf_writer.add_summary(summary, time_step)
        self.tf_writer.flush()

    def __init__(
        self, dqn, session, actions, epsilon=0.1, gamma=0.95, restore_model=True,
        board_path='debug/tensorboard/online/1', model_path='debug/models/model.ckpt'):

        self.actions = actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.dqn = dqn
        self.session = session
        self.loss = 0
        self.model_path = model_path

        self.tf_saver = tf.train.Saver()
        tf.summary.scalar('loss', self.dqn.loss)
        tf.summary.scalar('reward', self.dqn.reward)
        self.merged_summary = tf.summary.merge_all()
        self.tf_writer = tf.summary.FileWriter(board_path)

        if restore_model:
            self.tf_saver.restore(self.session, self.model_path)


class ReplayMemory:

    def add(self, tuple):
        """
        :param tuple: (state, action, reward, next state, terminal)
        """
        self.data.append(tuple)

    def sample(self, sample_size):
        index = np.random.choice(
            np.arange(len(self.data)),
            size=sample_size,
            replace=False)
        return [self.data[i] for i in index]

    def __init__(self, size):
        self.size = size
        self.data = deque(maxlen=size)
