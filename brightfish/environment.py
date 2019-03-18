import numpy as np

class Environment:
    def __init__(self, shape):
        self.shape = shape
        self.stage = None

    def step(self):
        raise NotImplementedError

    def run(self, timesteps):
        raise NotImplementedError

    def midpoint(self):
        if isinstance(self.shape, tuple):
            return tuple(s//2 for s in self.shape)
        else:
            return self.shape//2

    def left_eye(self, position):
        if isinstance(self.shape, tuple):
            start = np.unravel_index(0, self.shape)
            return self.stage[[slice(start[i], position[i])
                               for i in range(len(self.shape))]]
        else:
            return self.stage[0:position]
 
    def right_eye(self, position):
        if isinstance(self.shape, tuple):
            stop = np.unravel_index(-1, self.shape)
            return self.stage[[slice(position[i], stop[i])
                               for i in range(len(self.shape))]]
        else:
            return self.stage[position:-1]

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

    def run(self, timesteps):
        lines = []
        for i in range(timesteps):
            line = self.step()
            lines.append(line)
        lines = np.stack(lines)
        return lines
