from skimage import transform
from collections import deque
import numpy as np


def preprocess(frame):
    frame = frame / 255.0  # normalize
    frame = transform.resize(frame, [84, 84])  # resize
    frame = frame[26:70]  # crop
    return frame


class FrameStack:

    def clear(self):
        self.stack = deque([np.zeros((84, 84), dtype=np.int) for i in range(self.size)], maxlen=self.size)

    def add(self, frame, process=True):
        if process:
            frame = preprocess(frame)
        self.stack.append(frame)

    def as_state(self):
        return np.stack(self.stack, axis=2)

    def init_new(self, frame):
        for i in range(self.size):
            self.add(frame)

    def __init__(self, size):
        self.size = size
        self.stack = None
        self.clear()
