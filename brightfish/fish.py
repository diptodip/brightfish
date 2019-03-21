import numpy as np

class Fish:
    """
    Base class for defining a simulated zebrafish that respond to brightness. A
    base fish takes at least a heading to define its starting orientation.
    Subclasses of fish should implement methods for what happens in a given time
    step and running for multiple time steps.

    Args:
	heading (float): Defines the heading in radians of the fish.

	set_point (float or list of floats, optional): Defines the set point of the fish,
	i.e. the intensity/ies that the fish should seek to turn towards.

	learning_rate (float, optional): Defines how fast the fish updates its set point
	and turning probabilities.

	turning_rate (float, optional): Defines how fast the fish turns in a given time
	step.

    Attributes:
	heading (float): Defines the heading in radians of the fish.

	set_point (float): Defines the set point of the fish, i.e. the intensity
	that the fish should seek to turn towards.

	learning_rate(float): Defines how fast the fish updates its set point
	and turning probabilities.

	turning_rate(float): Defines how fast the fish turns in a given time
	step.

	p_right (float): Defines the probability of turning clockwise. Should be
	clamped to $[0, 1]$.

	p_left (float): Defines the probability of turning counterclockwise.
	Should be clamped to $[0, 1]$.
    """
    def __init__(self,
                 heading,
                 set_point=0.5,
                 learning_rate=5e-2,
                 turning_rate=1e-2):
        self.heading = heading
        self.set_point = set_point
        self.learning_rate = learning_rate
        self.turning_rate = turning_rate
        self.p_right = 0.5
        self.p_left = 0.5

    def __str__(self):
        message = ("{0}: heading: {1:.2f} set_point: {2:.2f} "
                   + "p_left: {3:.2f} p_right: {4:.2f}")
        return message.format(self.__class__.__name__,
                              self.heading,
                              self.set_point,
                              self.p_left,
                              self.p_right)
    
    def __repr__(self):
        message = ("{0}: heading: {1:.2f} set_point: {2:.2f} "
                   + "p_left: {3:.2f} p_right: {4:.2f}")
        return message.format(self.__class__.__name__,
                              self.heading,
                              self.set_point,
                              self.p_left,
                              self.p_right)

    def turn(self):
        """
	Updates ``self.heading`` by ``self.turning_rate`` radians in a random
	direction determined by the turning probabilities.
        """
        # 1 if turning counterclockwise
        turn_direction = np.random.binomial(1, self.p_left)
        # -1 if turning clockwise
        if turn_direction == 0:
            turn_direction = -1
        self.heading += turn_direction * self.turning_rate
        self.heading = self.heading % (2 * np.pi)

    def step(self, environment):
        """
	Defines the behavior of the fish in one time step in the given
	environment.

	Args:
	    environment (``Environment``): Defines the environment in which the
	    fish takes a step.
        """
        raise NotImplementedError

    def run(self, environment, timesteps):
        """
        Defines the behavior of the fish over multiple time steps.
	
        Args:
	    environment (``Environment``): Defines the environment in which the
	    fish takes a step.
        """
        raise NotImplementedError

class BinocularFish(Fish):
    """
    Model zebrafish that integrates binocular information to update a set point
    of preferred brightness.

    Args:
	heading (float): Defines the heading in radians of the fish.

	set_point (float): Defines the set point of the fish, i.e. the intensity
	that the fish should seek to turn towards.

	learning_rate (float): Defines how fast the fish updates its set point
	and turning probabilities.

	turning_rate (float): Defines how fast the fish turns in a given time
	step.

    Attributes:
	heading (float): Defines the heading in radians of the fish.

	set_point (float): Defines the set point of the fish, i.e. the intensity
	that the fish should seek to turn towards.

	learning_rate(float): Defines how fast the fish updates its set point
	and turning probabilities.

	turning_rate(float): Defines how fast the fish turns in a given time
	step.

	p_right (float): Defines the probability of turning clockwise. Should be
	clamped to $[0, 1]$.

	p_left (float): Defines the probability of turning counterclockwise.
	Should be clamped to $[0, 1]$.
    """
    def __init__(self, heading, set_point=0.5, learning_rate=5e-2):
        super(BinocularFish, self).__init__(heading, set_point, learning_rate)

    def step(self, environment):
        """
	Defines the behavior of the fish in one time step in the given
	environment. This fish takes the average of brightness information from
	both eyes and updates its set point to be closer to this average. This
	fish also updates its turning probabilities to increase probability to
	turn towards the eye with brightness closer to the updated set point.

	Args:
	    environment (``Environment``): Defines the environment in which the
	    fish takes a step.

	Returns:
	    A list of the parameters defining the status of the fish.
        """
        # calculate differences from both eyes
        brightness_left = environment.left_eye(self.heading).mean()
        brightness_right = environment.right_eye(self.heading).mean()

        # update set point to be closer to mean of two eyes
        update = self.set_point - np.mean([brightness_left, brightness_right])
        self.set_point -= self.learning_rate * update

        # update turn probabilities to turn towards area closer to set point
        # first calculate differences from set point on both sides
        diff_left = np.abs(brightness_left - self.set_point)
        diff_right = np.abs(brightness_right - self.set_point)
        # then calculate difference of differences
        diff_left_right = diff_left - diff_right
        # update turn probabilities appropriately
        self.p_right += self.learning_rate * diff_left_right
        self.p_left -= self.learning_rate * diff_left_right
        # clip updated values to maintain valid probabilities
        self.p_right = np.clip(self.p_right, 0.0, 1.0)
        self.p_left = np.clip(self.p_left, 0.0, 1.0)

        # turn fish
        self.turn()

        # step environment
        environment.step()

        # return updated parameters
        return [self.heading, self.set_point, self.p_left, self.p_right]

    def run(self, environment, timesteps):
        """
	Defines the behavior of the fish over multiple time steps.

	Args:
	    environment (``Environment``): Defines the environment in which the
	    fish takes a step.

	    timesteps (int): Defines the number of time steps to perform.

	Returns:
	    An ``np.ndarray`` of the parameters defining the status of the fish
	    at each time point.
        """
        params = [[self.heading, self.set_point, self.p_left, self.p_right]]
        for i in range(timesteps):
            params.append(self.step(environment))
        params = np.stack(params)
        return params

