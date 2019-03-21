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
    """
    Simple environment consisting of a circular sine wave. The sine wave peaks
    at a heading of $\pi$ radians and is squished to the range [0, 1].

    Args:
	shape (int): Integer defining the number of values of sine included in
	``self.stage``.

	dt (float, optional): Float that designates how much the peak moves
	counterclockwise in one time step if ``self.static`` is ``False``.

	static (bool, optional): Boolean variable that determines whether peak
	is stationary at each time step or moves autonomously at each time step.

    Attributes:
	shape (int): Integer defining the number of values of sine included in
	``self.stage``.

	start (float): Contains the start heading for a circle, $0$.

	stop (float): Contains the end heading for a circle, $2\pi$.

	phase (float): Defines the phase of the sine function in radians.
	Initialized to $\frac{3\pi}{2}$.

	dt (float, optional): Float that designates how much the peak moves
	counterclockwise in one time step if ``self.static`` is ``False``.

	stage (None or ndarray): Flat ``np.ndarray`` that contains
	``self.shape`` values of sine from $0$ to $2\pi$.

	static (bool, optional): Boolean variable that determines whether peak
	is stationary at each time step or moves autonomously at each time step.
    """
    def __init__(self,
                 shape,
                 dt = 1e-2,
                 static=True):
        super(SinusoidalCircle, self).__init__(shape)
        self.start = 0.0
        self.stop = 2 * np.pi
        self.phase = 3 * np.pi / 2
        self.dt = dt
        self.stage = 0.5 * (1 + np.sin(np.linspace(self.start,
                                            self.stop,
                                            num=self.shape)
                                       + self.phase))
        self.static = static

    def step(self):
        """
	If ``self.static`` is ``False``, move peak counterclockwise by
	``self.dt``.
        """
        if not self.static:
            self.phase = (self.phase + self.dt) % (2 * np.pi)
            line = 0.5 * (1 + np.sin(np.linspace(self.start,
                                          self.stop,
                                          num=self.shape)
                                     + self.phase))
            self.stage = line
    
    def left_eye(self, heading):
        """
	Returns the information observed by the left eye.

	Args:
	    heading (float): Gives the heading in radians from which to gather
	    left eye information.

        Returns:
	    An ``np.ndarray`` containing values from ``self.stage`` that are $\frac{\pi}{2}$ radians
	    counterclockwise to ``heading``.
        """
        heading = heading % (2 * np.pi)
        step_size = (2 * np.pi) / self.shape
        index = int(np.clip(heading / step_size, 0, self.shape))
        return self.stage.take(list(range(index, index+(self.shape // 4))),
                               mode='wrap')

    def right_eye(self, heading):
	"""
	Returns the information observed by the right eye.

	Args:
	    heading (float): Gives the heading in radians from which to gather
	    right eye information.

	Returns:
	    An ``np.ndarray`` containing values from ``self.stage`` that are
	    $\frac{\pi}{2}$ radians clockwise to ``heading``.
	"""
        heading = heading % (2 * np.pi)
        step_size = (2 * np.pi) / self.shape
        index = int(np.clip(heading / step_size, 0, self.shape))
        return self.stage.take(list(range(index-(self.shape // 4), index)),
                               mode='wrap')
