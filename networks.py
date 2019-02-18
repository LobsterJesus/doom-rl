import tensorflow as tf


class DeepQNetworkSimple:

    def __init__(self, state_size, action_size, learning_rate, name='DeepQNetworkSimple'):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions = tf.placeholder(tf.float32, [None, 3], name="actions")
            self.q_target = tf.placeholder(tf.float32, [None], name="target")
            self.reward = tf.placeholder(tf.float32, (), name="reward")

            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1")

            self.conv1_out = tf.nn.elu(
                self.conv1,
                name="conv1_out")

            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2")

            self.conv2_out = tf.nn.elu(
                self.conv2,
                name="conv2_out")

            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=128,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3")

            self.conv3_out = tf.nn.elu(
                self.conv3,
                name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1")

            self.output = tf.layers.dense(
                inputs=self.fc,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=3,
                activation=None)

            self.q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.q_target - self.q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class DeepQNetworkBatch:

    def __init__(self, state_size, action_size, learning_rate, name='DeepQNetworkBatch'):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions = tf.placeholder(tf.float32, [None, 3], name="actions")
            self.q_target = tf.placeholder(tf.float32, [None], name="target")
            self.reward = tf.placeholder(tf.float32, (), name="reward")

            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(
                self.conv1,
                training=True,
                epsilon=1e-5,
                name='batch_norm1')

            self.conv1_out = tf.nn.elu(
                self.conv1_batchnorm,
                name="conv1_out")

            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(
                self.conv2,
                training=True,
                epsilon=1e-5,
                name='batch_norm2')

            self.conv2_out = tf.nn.elu(
                self.conv2_batchnorm,
                name="conv2_out")

            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=128,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(
                self.conv3,
                training=True,
                epsilon=1e-5,
                name='batch_norm3')

            self.conv3_out = tf.nn.elu(
                self.conv3_batchnorm,
                name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1")

            self.output = tf.layers.dense(
                inputs=self.fc,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=3,
                activation=None)

            self.q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.q_target - self.q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class DeepQNetworkDueling:

    def __init__(self, state_size, action_size, learning_rate, name='DeepQNetworkDueling'):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions = tf.placeholder(tf.float32, [None, 3], name="actions")
            self.q_target = tf.placeholder(tf.float32, [None], name="target")
            self.reward = tf.placeholder(tf.float32, (), name="reward")

            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(
                self.conv1,
                training=True,
                epsilon=1e-5,
                name='batch_norm1')

            self.conv1_out = tf.nn.elu(
                self.conv1_batchnorm,
                name="conv1_out")

            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(
                self.conv2,
                training=True,
                epsilon=1e-5,
                name='batch_norm2')

            self.conv2_out = tf.nn.elu(
                self.conv2_batchnorm,
                name="conv2_out")

            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=128,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(
                self.conv3,
                training=True,
                epsilon=1e-5,
                name='batch_norm3')

            self.conv3_out = tf.nn.elu(
                self.conv3_batchnorm,
                name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            # SPLIT streams
            # Value function stream
            self.value_fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="value_fc")

            self.value = tf.layers.dense(
                inputs=self.value_fc,
                units=1,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="value")

            # Advantage function stream
            self.advantage_fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="advantage_fc")

            self.advantage = tf.layers.dense(
                inputs=self.advantage_fc,
                units=self.action_size,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="advantages")

            # MERGE streams
            self.output = self.value + tf.subtract(
                self.advantage,
                tf.reduce_mean(self.advantage, axis=1, keep_dims=True))

            self.q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.q_target - self.q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


def copy_network_variables(from_name, to_name):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_name)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_name)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

