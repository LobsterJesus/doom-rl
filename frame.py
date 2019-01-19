from skimage import transform

import numpy as np


def preprocess(frame):
    frame = frame / 255.0  # normalize
    frame = transform.resize(frame, [84, 84])  # resize
    frame = frame[26:70]  # crop
    return frame
