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

class SinusoidalCircle(Environment):
    def __init__(self,
                 shape,
                 dt = 1e-2,
                 static=True):
        super(SinusoidalCircle, self).__init__(shape)
        self.start = 0
        self.stop = 2 * np.pi
        self.phase = 3 * np.pi / 2
        self.dt = dt
        self.stage = 0.5 * (1 + np.sin(np.linspace(self.start,
                                            self.stop,
                                            num=self.shape)
                                       + self.phase))
        self.static = static

    def step(self):
        if not self.static:
            self.phase = (self.phase + self.dt) % (2 * np.pi)
            line = 0.5 * (1 + np.sin(np.linspace(self.start,
                                          self.stop,
                                          num=self.shape)
                                     + self.phase))
            self.stage = line
    
    def left_eye(self, heading):
        heading = heading % (2 * np.pi)
        step_size = (2 * np.pi) / self.shape
        index = int(np.clip(heading / step_size, 0, self.shape))
        return self.stage.take(list(range(index, index+(self.shape // 4))),
                               mode='wrap')

    def right_eye(self, heading):
        heading = heading % (2 * np.pi)
        step_size = (2 * np.pi) / self.shape
        index = int(np.clip(heading / step_size, 0, self.shape))
        return self.stage.take(list(range(index-(self.shape // 4), index)),
                               mode='wrap')
