import numpy as np

class Environment:
    def __init__(self, shape):
        self.shape = shape
        self.stage = None

    def step(self):
        raise NotImplementedError

    def run(self, timesteps):
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

    def run(self, timesteps):
        lines = []
        for i in range(timesteps):
            line = self.step()
            lines.append(line)
        lines = np.stack(lines)
        return lines
