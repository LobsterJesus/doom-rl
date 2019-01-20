import numpy as np
import random
import tensorflow as tf


class Agent:

    def get_policy_action(self, state):
        if np.random.rand() > self.epsilon:
            q_values = self.session.run(self.dqn.output, feed_dict={self.dqn.inputs: state.reshape((1, *state.shape))})
            choice = np.argmax(q_values)
            return self.actions[int(choice)]
        else:
            return random.choice(self.actions)

    def train(self, state, action, reward, next_state, time_step, terminal=False):
        q_values = self.session.run(self.dqn.output, feed_dict={self.dqn.inputs: state.reshape((1, *state.shape))})
        q_values_next = self.session.run(self.dqn.output, feed_dict={self.dqn.inputs: next_state.reshape((1, *next_state.shape))})

        target = q_values[0]
        for i, v in enumerate(action):
            if v == 1:
                target[i] = (reward + self.gamma * np.max(q_values_next)) if not terminal else reward

        if terminal:
            summary = self.session.run(self.merged_summary, feed_dict={self.dqn.inputs: state.reshape((1, *state.shape)),
                                                             self.dqn.q_target: target,
                                                             self.dqn.actions: [action]})
            self.tf_writer.add_summary(summary, time_step)
            self.tf_writer.flush()

        self.loss, _ = self.session.run([self.dqn.loss, self.dqn.optimizer],
            feed_dict={self.dqn.inputs: state.reshape((1, *state.shape)),
                      self.dqn.q_target: target,
                      self.dqn.actions: [action]})


    def __init__(self, dqn, session, actions, epsilon=0.1, gamma=0.95):
        self.actions = actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.dqn = dqn
        self.session = session
        self.loss = 0
        tf.summary.scalar("loss", self.dqn.loss)
        self.merged_summary = tf.summary.merge_all()
        self.tf_writer = tf.summary.FileWriter("debug/tensorboard/online/1")
