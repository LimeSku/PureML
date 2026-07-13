import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, random_state=42):
        rng = np.random.default_rng(random_state)
        kh, kw = kernel_size
