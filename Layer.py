import numpy as np


class MulLayer:

    def __init__(self):
        self.x = None
        self.y = None

    def Forword(self, x, y):
        self.x = x
        self.y = y

        return x * y

    def Backword(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def Forword(self, x, y):
        return x + y

    def Backword(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


class ReLU:
    def __init__(self):
        self.mask = None

    def Forword(self, x):
        self.mask = (x <= 0)
        out = x.copy
        out[self.mask] = 0

        return out

    def Backword(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigoid:
    def __init__(self):
        self.out = None

    def Forword(self, x):
        return 1 / (1 + np.exp(-x))

    def Backword(self, dout):
        return dout * (1.0 - self.out) * self.out
