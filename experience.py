import numpy as np
from collections import deque


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


class SumTree(object):
    """
    Code by Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    Modifications by Thomas Simonini:
    https://gist.github.com/simoninithomas
    """

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class ReplayMemoryPER:  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def add(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        # priority_segment =self.tree.total_priority / n
        priority_segment = np.nan_to_num(self.tree.total_priority / n)
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = ((p_min * n) ** (-self.PER_b)) if p_min > 0 else 1

        if priority_segment <= 0:
            print("priority_segment zero, total: {}, n: {}".format(self.tree.total_priority, n))

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            # print("{}, {}".format(a, b))
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            sampling_probabilities = priority / self.tree.total_priority
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            b_idx[i] = index
            experience = [data]
            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

