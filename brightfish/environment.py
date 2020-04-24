import numpy as np
from skimage.draw import circle
from skimage.filters import gaussian

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
    
    @property
    def midpoint(self):
        """Gives the middle coordinate/value of ``self.shape``."""
        return tuple(s//2 for s in self.shape)

class Spotlight(Environment):
    """
    An Environment that simulates the experiment from Burgess 2010. The
    experiment consists of a 2D arena in which a Fish is adapted to a uniform
    brightness for some time. After this time period, a target Gaussian spot of
    a specified brightness is shown at the desired location (this should fall
    in the field of view of the Fish).
    """
    def __init__(self,
                 shape,
                 burnin_time,
                 spot_coordinate,
                 initial_value=0.25,
                 spot_value=0.25,
                 spot_radius=1):
        super(Spotlight, self).__init__(shape)
        assert self.shape[0] == self.shape[1], "Arena must be square"
        self.burnin_time = burnin_time
        self.spot_coordinate = spot_coordinate
        self.initial_value = initial_value
        self.spot_value = spot_value
        self.spot_radius = spot_radius
        # TODO: need to fix fish FOV to allow making use of aperture
        aperture_rows, aperture_cols = circle(int(self.shape[0] / 2),
                                              int(self.shape[1] / 2),
                                              int(self.shape[0] / 2),
                                              shape=self.shape)
        self.aperture = np.zeros(self.shape)
        self.aperture[aperture_rows, aperture_cols] = 1.0
        self.reset()

    def step(self):
        self.time_step += 1
        if self.time_step == self.burnin_time:
            self.place_spot()

    def place_spot(self):
        self.stage.fill(0.0)
        spot_rows, spot_cols = circle(*self.spot_coordinate,
                                      self.spot_radius,
                                      shape=self.shape)
        self.stage[spot_rows, spot_cols] = self.spot_value
        self.stage = gaussian(self.stage)

    def reset(self):
        self.time_step = 0
        self.stage = np.full(self.shape, self.initial_value)

class PartitionedHalves(Environment):
    """
    Simple 2D Environment where two halves of an arena can have different
    brightness values. The two sides can switch their brightness values every
    few time steps. The arena can also start at some uniform value for both
    halves as a burn in period.

    Args:
	shape (tuple of ints): Tuple of integers defining the sahpe of the
	environment.

        burnin_time (int): Integer that defines how many time steps are taken
        before changing the brightness of the two halves of the arena.

	switch_time (int): Integer that defines how many time steps are taken
	before switching the brightness of the two halves.

        initial_value (float, optional): Float that defines the initial value
        for both halves of the arena for burnin before the two halves take
        different values.

	initial_half (str, optional): String that is either 'left' or 'right',
	defining which half of the environment starts out bright. The other half
	will start dark.

	static (bool, optional): Boolean variable that determines whether the
	environment switches the brightnesses of its halves or remains static.

    Attributes:
	shape (tuple of ints): Tuple of integers defining the sahpe of the
	environment.

        burnin_time (int): Integer that defines how many time steps are taken
        before changing the brightness of the two halves of the arena.

	switch_time (int): Integer that defines how many time steps are taken
	before switching the brightness of the two halves.

        initial_value (float): Float that defines the initial value
        for both halves of the arena for burnin before the two halves take
        different values.

	initial_half (str): String that is either 'left' or 'right',
	defining which half of the environment starts out bright. The other half
	will start dark.

	static (bool, optional): Boolean variable that determines whether the
	environment switches the brightnesses of its halves or remains static.
    """
    def __init__(self,
                 shape,
                 burnin_time,
                 switch_time,
                 initial_value=0.25,
                 initial_half='left',
                 static=False):
        super(PartitionedHalves, self).__init__(shape)
        self.burnin_time = burnin_time
        self.switch_time = switch_time
        self.initial_value = initial_value
        self.initial_half = initial_half
        self.static = static
        self.reset()

    def step(self):
        self.time_step += 1
        if (self.time_step == self.burnin_time) and not self.static:
            self.partition()
        if (self.time_step % self.switch_time == 0) and not self.static:
            self.switch_halves()

    def switch_halves(self):
        self.stage = np.flip(self.stage, axis=1).copy()

    def partition(self):
        self.stage.fill(0.0)
        if self.initial_half == 'left':
            self.stage[:, :self.midpoint[1]+1] = 1.0
        else:
            self.stage[:, self.midpoint[1]:] = 1.0

    def reset(self):
        self.time_step = 0
        self.stage = np.full(self.shape, self.initial_value)

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
    def __init__(self, shape, dt = 1e-2, static=True):
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
            line = np.repeat(line[None, :], self.shape[0], axis=0)
            self.stage = line
