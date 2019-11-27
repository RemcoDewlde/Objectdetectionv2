import numpy as np
from main import net


class Color:
    def __init__(self):
        self.color = self.gen_colors()

    def gen_colors(self):
        return np.random.uniform(0, 255, size=(len(net.classes), 3))

# TODO: change this maybe?
