import numpy as np

class Environment:
    def __init__(self, shape):
        self.shape = shape
        self.stage = None

    def step(self):
        raise NotImplementedError

    def midpoint(self):
        if isinstance(self.shape, tuple):
            return tuple(s//2 for s in self.shape)
        else:
            return self.shape//2

    def left_eye(self, heading):
        raise NotImplementedError
 
    def right_eye(self, heading):
        raise NotImplementedError

class SinusoidalLine(Environment):
    def __init__(self, shape,
                       start = -np.pi,
                       stop = np.pi):
        super(SinusoidalLine, self).__init__(shape)
        self.start = start
        self.stop = stop
        self.stage = np.sin(np.linspace(self.start,
                                        self.stop,
                                        steps=self.shape))
        self.phase = 0

    def step(self):
        self.phase += 1
        return np.roll(self.stage, self.phase-1)

