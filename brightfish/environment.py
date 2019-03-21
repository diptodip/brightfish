import numpy as np

class Environment:
    """
    Base class for defining environments. A base environment takes at least a
    shape. Subclasses of Environment should implement methods for changes in the
    environment, visibility from a left eye, and visibility from a right eye.

    Args:
	shape (tuple or int): Integer or tuple of integers defining the shape of
	the environment.

    Attributes:
	shape (tuple or int): Integer or tuple of integers defining the shape of
	the environment.

	stage (``None`` or ``np.ndarray``): Contains actual data about the
	environment, initialized to ``None``. The ``__init__()`` method of
	subclasses of ``Environment`` should define the stage as an ndarray.
    """
    def __init__(self, shape):
        self.shape = shape
        self.stage = None

    def step(self):
        """Defines the behavior of the environment in one time step."""
        raise NotImplementedError

    def midpoint(self):
        """Gives the middle coordinate/value of ``self.shape``."""
        if isinstance(self.shape, tuple):
            return tuple(s//2 for s in self.shape)
        else:
            return self.shape//2

    def left_eye(self, heading):
        """
	Returns the information observed by the left eye.

	Args:
	    heading (float): Gives the heading in radians from which to gather
	    left eye information.
        """
        raise NotImplementedError
 
    def right_eye(self, heading):
        """
	Returns the information observed by the right eye.

	Args:
	    heading (float): Gives the heading in radians from which to gather
	    right eye information.
        """
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
