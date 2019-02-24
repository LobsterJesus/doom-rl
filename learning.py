import numpy as np
import random
import tensorflow as tf
from collections import deque
from datetime import datetime
from networks import copy_network_variables


class Agent:

    def get_policy_action(self, state, use_target_network=False):
        # Use exponential epsilon decay for exploration/exploitation strategy
        self.exploration_probability = \
            self.epsilon_stop + (self.epsilon_start - self.epsilon_stop) * np.exp(-self.epsilon_decay_rate * self.internal_step)

        if self.exploration_probability < np.random.rand():
            if not use_target_network:
                q_values = self.session.run(
                    self.dqn.output,
                    feed_dict={self.dqn.inputs: state.reshape((1, *state.shape))})
            else:
                q_values = self.session.run(
                    self.dqn_target.output,
                    feed_dict={self.dqn_target.inputs: state.reshape((1, *state.shape))})
            choice = np.argmax(q_values)
            return self.actions[int(choice)]
        else:
            return random.choice(self.actions)

    def get_q_values_batch(self, state_batch):
        return self.session.run(self.dqn.output, feed_dict={self.dqn.inputs: state_batch})

    # todo: use internal timestep
    def train(self, state, action, reward, next_state, time_step, terminal=False):
        self.timestep += 1
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
        self.internal_step += 1
        if self.internal_step % self.skip_frames != 0:
            return False

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

        if self.log and self.timestep % 10 == 0:
            summary = self.session.run(
                self.merged_summary,
                feed_dict={
                    self.dqn.reward: 0,
                    self.dqn.inputs: states,
                    self.dqn.q_target: targets_dqn,
                    self.dqn.actions: actions})

            # self.tf_writer.add_summary(summary, episode_index)
            self.tf_writer.add_summary(summary)
            self.tf_writer.flush()

        if self.timestep == 0 and episode_index > 0 and episode_index % 5 == 0:
            print("saving model to '" + self.model_path + "'")
            self.tf_saver.save(self.session, self.model_path)

        self.timestep += 1

        return True

    def train_batch_dqn(self, batch, episode_index):
        self.internal_step += 1
        if self.internal_step % self.skip_frames != 0:
            return False

        states = np.array([sample[0] for sample in batch], ndmin=3)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_states = np.array([sample[3] for sample in batch], ndmin=3)
        terminals = np.array([sample[4] for sample in batch])
        q_values_next_state = self.get_q_values_batch(next_states)
        q_targets = []
        q_dqn_targets = self.session.run(self.dqn_target.output, feed_dict={self.dqn_target.inputs: next_states})

        for i in range(0, len(batch)):
            done = terminals[i]
            action = np.argmax(q_values_next_state[i])

            if done:
                q_targets.append(rewards[i])
            else:
                target = rewards[i] + self.gamma * q_dqn_targets[i][action]
                q_targets.append(target)
                # q_targets.append(rewards[i] + self.gamma * np.max(q_values_next_state[i]))

        targets_dqn = np.array([each for each in q_targets])

        self.loss, _ = self.session.run(
            [self.dqn.loss, self.dqn.optimizer],
            feed_dict={
                self.dqn.inputs: states,
                self.dqn.q_target: targets_dqn,
                self.dqn.actions: actions})

        if self.log and self.timestep % 10 == 0:
            summary = self.session.run(
                self.merged_summary,
                feed_dict={
                    self.dqn.reward: 0,
                    self.dqn.inputs: states,
                    self.dqn.q_target: targets_dqn,
                    self.dqn.actions: actions})

            # self.tf_writer.add_summary(summary, episode_index)
            self.tf_writer.add_summary(summary)
            self.tf_writer.flush()

        if self.timestep == 0 and episode_index > 0 and episode_index % 5 == 0:
            print("saving model to '" + self.model_path + "'")
            self.tf_saver.save(self.session, self.model_path)

        self.timestep += 1

        return True

    def finish_episode(self):
        self.timestep = 0

    def train_batch_target_network(self, batch, episode_index):
        if self.train_batch_dqn(batch, episode_index):
            self.tau += 1
            if self.tau >= self.max_tau:
                update_target = copy_network_variables(self.dqn.name, self.dqn_target.name)
                self.session.run(update_target)
                self.tau = 0
                print("Model updated")

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

    def setup_logger(self, root_logdir):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        logdir = "{}/run-{}/".format(root_logdir, now)
        return tf.summary.FileWriter(logdir)

    def __init__(
        self, dqn, session, actions,
        epsilon_start=1.0, epsilon_stop=0.01, epsilon_decay_rate=0.0001,
        gamma=0.95, restore_model=True, dqn_target=None,
        logger_path='debug/tf_logs/default',
        model_path='debug/models/model.ckpt',
        log=False):

        self.actions = actions
        self.gamma = gamma
        self.dqn = dqn
        self.dqn_target = dqn_target
        self.session = session
        self.loss = 0
        self.model_path = model_path
        self.tau = 0
        self.max_tau = 30
        self.epsilon_start = epsilon_start
        self.epsilon_stop = epsilon_stop
        self.epsilon_decay_rate = epsilon_decay_rate
        self.exploration_probability = 0
        self.timestep = 0
        self.internal_step = 0
        self.skip_frames = 4
        self.log = log

        self.tf_saver = tf.train.Saver()

        # start tensorboard using
        # tensorboard --logdir debug/tf_logs/[agent]

        if self.log:
            tf.summary.scalar('loss', self.dqn.loss)
            tf.summary.scalar('reward', self.dqn.reward)
            self.merged_summary = tf.summary.merge_all()
            self.tf_writer = self.setup_logger(logger_path)

        if restore_model:
            print("restoring model '" + self.model_path + "'...")
            self.tf_saver.restore(self.session, self.model_path)
        else:
            print("initializing model variables for '" + self.model_path + "'")
            self.session.run(tf.global_variables_initializer())


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
