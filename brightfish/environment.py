import numpy as np

class Environment:
    """
    Base class for defining environments. A base environment takes at least a
    shape. Subclasses of Environment should implement methods for changes in the
    environment, visibility from a left eye, and visibility from a right eye.

    Args:
	shape (tuple of ints): Tuple of integers defining the shape of the
	environment.

    Attributes:
	shape (tuple of ints): Tuple of integers defining the shape of the
	environment.

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
        return tuple(s//2 for s in self.shape)

class SinusoidalGradient(Environment):
    """
    Simple 2D environment consisting of a sine wave gradient of intensity. The
    sine wave peaks in the middle and is squished to the range [0, 1].

    Args:
	shape (tuple of ints): Tuple of integers defining the shape of the
	environment; the sine gradient will be changing across each row.

	dt (float, optional): Float that designates how much the peak moves
	counterclockwise in one time step if ``self.static`` is ``False``.

	static (bool, optional): Boolean variable that determines whether peak
	is stationary at each time step or moves autonomously at each time step.

    Attributes:
	shape (tuple of ints): Tuple of integers defining the shape of the
	environment; the sine gradient will be changing across each row.

	start (float): Contains the start heading for a circle, $0$.

	stop (float): Contains the end heading for a circle, $2\pi$.

	phase (float): Defines the phase of the sine function in radians.
	Initialized to $\frac{3\pi}{2}$.

	dt (float, optional): Float that designates how much the peak moves
	counterclockwise in one time step if ``self.static`` is ``False``.

	stage (``np.ndarray``): 2D ``np.ndarray`` where each row contains
	``self.shape[1]`` values of sine from $0$ to $2\pi$.

	static (bool, optional): Boolean variable that determines whether peak
	is stationary at each time step or moves autonomously at each time step.
    """
    def __init__(self,
                 shape,
                 dt = 1e-2,
                 static=True):
        super(SinusoidalGradient, self).__init__(shape)
        self.start = 0.0
        self.stop = 2 * np.pi
        self.phase = 3 * np.pi / 2
        self.dt = dt
        self.stage = 0.5 * (1 + np.sin(np.linspace(self.start,
                                            self.stop,
                                            num=self.shape[1])
                                       + self.phase))
        self.stage = np.repeat(self.stage[None, :], self.shape[0], axis=0)
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
                                          num=self.shape[1])
                                     + self.phase))
            line = np.repeat(line[:, None], self.shape[0], axis=0)
            self.stage = line
