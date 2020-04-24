import numpy as np
from skimage.draw import polygon

from .utils import *

class Fish:
    """
    Base class for defining a simulated zebrafish that respond to brightness. A
    base fish takes at least a heading to define its starting orientation.
    Subclasses of fish should implement methods for what happens in a given time
    step and running for multiple time steps.

    Args:
	heading (float): Defines the heading in radians of the fish. The fish
	exists in environments defined by 2D arrays where a heading of $0$
	points directly to the right of the array from whatever position in the
	array the fish is in.

	position (list of ints): Defines the position of the fish as an index
	into a 2D array.

	set_point (float or list of floats, optional): Defines the set point of
	the fish, i.e. the intensity/ies that the fish should seek to turn
	towards.

	learning_rate (float, optional): Defines how fast the fish updates its
	set point and turning probabilities.
	
	turning_cap (float, optional): Defines maximum probability for a given
	direction.
	
	turning_scale (float, optional): Defines scaling of ``turning_rate`` due
	to angular distance from set point.

	turning_rate (float, optional): Defines how fast the fish turns in a
	given time step.

	p_move (float, optional): Defines probability of moving on a given time
	step.

	move_distance (float, optional): Defines number of units moved in a
	direction given by ``self.heading`` if fish moves in a given time step.

    Attributes:
	heading (float): Defines the heading in radians of the fish. The fish
	exists in environments defined by 2D arrays where a heading of $0$
	points directly to the right of the array from whatever position in the
	array the fish is in.

	position (list of ints): Defines the position of the fish as an index
	into a 2D array.

	set_point (float): Defines the set point of the fish, i.e. the intensity
	that the fish should seek to turn towards.

	learning_rate(float): Defines how fast the fish updates its set point
	and turning probabilities.
	
	turning_cap (float): Defines maximum probability for a given direction.
	
	turning_scale (float): Defines scaling of ``turning_rate`` due to
	angular distance from set point.

	turning_rate (float): Defines how fast the fish turns in a given time
	step.

	p_right (float): Defines the probability of turning clockwise. Should be
	clamped to $[0, 1]$.

	p_left (float): Defines the probability of turning counterclockwise.
	Should be clamped to $[0, 1]$.
	
        p_move (float): Defines probability of moving on a given time step.

	move_distance (float): Defines number of units moved in a direction
	given by ``self.heading`` if fish moves in a given time step.
    """
    def __init__(self,
                 heading,
                 position,
                 set_point=0.5,
                 learning_rate=5e-2,
                 turning_cap=1.0,
                 turning_scale=2.0,
                 turning_rate=5e-2,
                 p_move=0.2,
                 move_distance=5.0):
        self.heading = heading
        self.position = position
        self.set_point = set_point
        self.learning_rate = learning_rate
        self.turning_cap = turning_cap
        self.turning_scale = turning_scale
        self.turning_rate = turning_rate
        self.p_right = 1.0/3.0
        self.p_left = 1.0/3.0
        self.p_noturn = 1.0/3.0
        self.p_move = p_move
        self.move_distance = move_distance

    def __str__(self):
        message = ("{0}: heading: {1:.2f} position: {2} set_point: {3:.2f} "
                   + "p_left: {4:.2f} p_right: {5:.2f}")
        return message.format(self.__class__.__name__,
                              self.heading,
                              self.position,
                              self.set_point,
                              self.p_left,
                              self.p_right)
    
    def __repr__(self):
        message = ("{0}: heading: {1:.2f} position: {2} set_point: {3:.2f} "
                   + "p_left: {4:.2f} p_right: {5:.2f}")
        return message.format(self.__class__.__name__,
                              self.heading,
                              self.position,
                              self.set_point,
                              self.p_left,
                              self.p_right)

    def turn(self):
        """
	Updates ``self.heading`` by ``self.turning_rate`` radians in a random
	direction determined by the turning probabilities.
        """
        #TODO: need to change how turning works to choose angles from a
        #      distribution based on the error from set point
        f
        # determine direction of turn from multinomial distribution
        # 0 if turning counterclockwise
        # 1 if turning clockwise
        # 2 if not turning
        turn_direction = np.random.multinomial(1,
                                               [self.p_left,
                                                self.p_right,
                                                self.p_noturn])
        turn_direction = turn_direction.argmax()
        # determine whether to add/subtract/do nothing to heading (radians)
        # 1 if turning counterclockwise
        # -1 if turning clockwise
        # 0 if not turning
        if turn_direction == 0:
            turn_direction = 1
        elif turn_direction == 1:
            turn_direction = -1
        elif turn_direction == 2:
            turn_direction = 0
        self.heading += turn_direction * self.turning_rate
        self.heading = self.heading % (2 * np.pi)
    
    def move(self, shape):
        """
	Updates ``self.position`` by moving ``self.move_distance`` units in a
	direction given by ``self.heading``.
        """
        # decide if moving
        moving = np.random.binomial(1, self.p_move)
        # if moving, update position to move
        # by ``self.move_distance`` in ``self.heading`` direction
        if moving:
            self.turn()
            r, c = pol2cart(self.move_distance, self.heading, origin=self.position)
            if r >= 0 and r < shape[0] and c >= 0 and c < shape[1]:
                self.position = [r, c]

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

	    timesteps (int): Defines the number of time steps to perform.
        """
        raise NotImplementedError

    def left_eye(self, shape):
        """
	Returns the coordinates in a 2D array of dimensions ``shape`` observed
	by the left eye.

	Args:
	    shape (tuple of ints): Gives the shape of the 2D array in which the
	    fish can observe information.

	Returns:
	    A tuple of ``np.ndarray``s ``rr`` and ``cc`` containing row
	    coordinates and column coordinates of a $\frac{\pi}{2}$ degree field
	    of view observed by the fish in a direction $\frac{\pi}{2}$ from its
	    heading. These values may be used to index dirrectly into a 2D
	    array, e.g. ``arr[rr, cc]``.
        """
        radius = max(shape) * 1000
        r1, c1 = pol2cart(radius,
                          (self.heading + np.pi/4) % (2 * np.pi),
                          origin=self.position)
        r2, c2 = pol2cart(radius,
                          (self.heading + (3 * np.pi/4)) % (2 * np.pi),
                          origin=self.position)
        r = [self.position[0], r1, r2]
        c = [self.position[1], c1, c2]
        return polygon(r, c, shape=shape)
    
    def right_eye(self, shape):
        """
	Returns the coordinates in a 2D array of dimensions ``shape`` observed
	by the right eye.

	Args:
	    shape (tuple of ints): Gives the shape of the 2D array in which the
	    fish can observe information.

	Returns:
	    A tuple of ``np.ndarray``s ``rr`` and ``cc`` containing row
	    coordinates and column coordinates of a $\frac{\pi}{2}$ degree field
	    of view observed by the fish in a direction $-\frac{\pi}{2}$ from
	    its heading. These values may be used to index dirrectly into a 2D
	    array, e.g. ``arr[rr, cc]``.
        """
        radius = max(shape) * 1000
        r1, c1 = pol2cart(radius,
                          (self.heading - np.pi/4) % (2 * np.pi),
                          origin=self.position)
        r2, c2 = pol2cart(radius,
                          (self.heading - (3 * np.pi/4)) % (2 * np.pi),
                          origin=self.position)
        r = [self.position[0], r1, r2]
        c = [self.position[1], c1, c2]
        return polygon(r, c, shape=shape)

class BinocularFish(Fish):
    """
    Model zebrafish that integrates binocular information to update a set point
    of preferred brightness.

    Args:
	heading (float): Defines the heading in radians of the fish.
	
        position (list of ints): Defines the position of the fish as an index
	into a 2D array.

	set_point (float, optional): Defines the set point of the fish, i.e. the intensity
	that the fish should seek to turn towards.

	learning_rate (float, optional): Defines how fast the fish updates its set point
	and turning probabilities.
	
        turning_cap (float, optional): Defines maximum probability for a given
	direction.

	turning_rate (float, optional): Defines how fast the fish turns in a given time
	step.
	
        p_move (float, optional): Defines probability of moving on a given time
	step.

	move_distance (float, optional): Defines number of units moved in a
	direction given by ``self.heading`` if fish moves in a given time step.

    Attributes:
	heading (float): Defines the heading in radians of the fish.

	position (list of ints): Defines the position of the fish as an index
	into a 2D array.

	set_point (float): Defines the set point of the fish, i.e. the intensity
	that the fish should seek to turn towards.

	learning_rate(float): Defines how fast the fish updates its set point
	and turning probabilities.
	
        turning_cap (float): Defines maximum probability for a given direction.

	turning_rate(float): Defines how fast the fish turns in a given time
	step.

	p_right (float): Defines the probability of turning clockwise. Should be
	clamped to $[0, 1]$.

	p_left (float): Defines the probability of turning counterclockwise.
	Should be clamped to $[0, 1]$.
	
        p_move (float): Defines probability of moving on a given time step.

	move_distance (float): Defines number of units moved in a direction
	given by ``self.heading`` if fish moves in a given time step.
    """
    def __init__(self,
                 heading,
                 position,
                 set_point=0.5,
                 learning_rate=5e-2,
                 turning_cap=1.0,
                 turning_scale=2.0,
                 turning_rate=5e-2,
                 p_move=0.2,
                 move_distance=5.0):
        super(BinocularFish, self).__init__(heading,
                                            position,
                                            set_point=set_point,
                                            learning_rate=learning_rate,
                                            turning_cap=turning_cap,
                                            turning_scale=turning_scale,
                                            turning_rate=turning_rate,
                                            p_move=p_move,
                                            move_distance=move_distance)

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
        # collect brightness information from both eyes
        left_fov = self.left_eye(environment.shape)
        right_fov = self.right_eye(environment.shape)
        # check for empty fovs (due to being at edge of environment)
        if left_fov[0].size > 0 and left_fov[1].size > 0:
            brightness_left = environment.stage[left_fov[0], left_fov[1]].mean()
        else:
            brightness_left = 0.0
        if right_fov[0].size > 0 and right_fov[1].size > 0:
            brightness_right = environment.stage[right_fov[0], right_fov[1]].mean()
        else:
            brightness_right = 0.0

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
        self.p_left -= self.learning_rate * diff_left_right
        # clip updated values to maintain valid probabilities
        self.p_left = np.clip(self.p_left, 0.0, self.turning_cap)
        self.p_right = self.turning_cap - self.p_left
        self.p_noturn = 1.0 - (self.p_left + self.p_right)
        self.p_noturn = np.clip(self.p_noturn, 0.0, 1.0)

        # move fish
        self.move(environment.shape)

        # step environment
        environment.step()

        # return updated parameters
        return [self.heading,
                self.position[0],
                self.position[1],
                self.set_point,
                self.p_left,
                self.p_right,
                self.p_noturn]

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
        params = [[self.heading,
                   self.position[0],
                   self.position[1],
                   self.set_point,
                   self.p_left,
                   self.p_right,
                   self.p_noturn]]
        for i in range(timesteps):
            params.append(self.step(environment))
        params = np.stack(params)
        return params

class MonocularFish(Fish):
    """
    Model zebrafish that integrates monocular information to update two set
    points of preferred brightness (one for each eye).

    Args:
	heading (float): Defines the heading in radians of the fish.
	
        position (list of ints): Defines the position of the fish as an index
	into a 2D array.

	set_point (list of floats, optional): Defines the set points of the
	fish, i.e. the intensities that the fish should seek to turn towards
	given information from each eye, respectively.

	learning_rate (float, optional): Defines how fast the fish updates its
	set points and turning probabilities.
	
        turning_cap (float, optional): Defines maximum probability for a given
	direction.

	turning_rate (float, optional): Defines how fast the fish turns in a
	given time step.
	
        p_move (float, optional): Defines probability of moving on a given time
	step.

	move_distance (float, optional): Defines number of units moved in a
	direction given by ``self.heading`` if fish moves in a given time step.

    Attributes:
	heading (float): Defines the heading in radians of the fish.
	
        position (list of ints): Defines the position of the fish as an index
	into a 2D array.

	set_point (list of floats): Defines the set points of the fish, i.e. the
	intensities that the fish should seek to turn towards given information
	from each eye, respectively.

	learning_rate (float): Defines how fast the fish updates its set points
	and turning probabilities.
	
        turning_cap (float): Defines maximum probability for a given direction.

	turning_rate (float): Defines how fast the fish turns in a given time
	step.

	p_right (float): Defines the probability of turning clockwise. Should be
	clamped to $[0, 1]$.

	p_left (float): Defines the probability of turning counterclockwise.
	Should be clamped to $[0, 1]$.
	
	p_move (float): Defines probability of moving on a given time step.

	move_distance (float): Defines number of units moved in a direction
	given by ``self.heading`` if fish moves in a given time step.
    """
    def __init__(self,
                 heading,
                 position,
                 set_point=[0.5, 0.5],
                 learning_rate=5e-2,
                 turning_cap=1.0,
                 turning_rate=5e-2,
                 p_move=0.2,
                 move_distance=5.0):
        super(MonocularFish, self).__init__(heading,
                                            position,
                                            set_point=set_point,
                                            learning_rate=learning_rate,
                                            turning_cap=turning_cap,
                                            turning_rate=turning_rate,
                                            p_move=p_move,
                                            move_distance=move_distance)
    
    def __str__(self):
        message = ("{0}: heading: {1:.2f} position: {2} "
                   + "set_point_left: {3:.2f} set_point_right: {4:.2f} "
                   + "p_left: {5:.2f} p_right: {6:.2f}")
        return message.format(self.__class__.__name__,
                              self.heading,
                              self.position,
                              self.set_point[0],
                              self.set_point[1],
                              self.p_left,
                              self.p_right)
    
    def __repr__(self):
        message = ("{0}: heading: {1:.2f} position: {2} "
                   + "set_point_left: {3:.2f} set_point_right: {4:.2f} "
                   + "p_left: {5:.2f} p_right: {6:.2f}")
        return message.format(self.__class__.__name__,
                              self.heading,
                              self.position,
                              self.set_point[0],
                              self.set_point[1],
                              self.p_left,
                              self.p_right)

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
        # collect brightness information from both eyes
        left_fov = self.left_eye(environment.shape)
        right_fov = self.right_eye(environment.shape)
        # check for empty fovs (due to being at edge of environment)
        if left_fov[0].size > 0 and left_fov[1].size > 0:
            brightness_left = environment.stage[left_fov[0], left_fov[1]].mean()
        else:
            brightness_left = 0.0
        if right_fov[0].size > 0 and right_fov[1].size > 0:
            brightness_right = environment.stage[right_fov[0], right_fov[1]].mean()
        else:
            brightness_right = 0.0

        # update set points to be closer to observed brightness from each eye
        update1 = self.set_point[0] - brightness_left
        update2 = self.set_point[1] - brightness_right
        self.set_point[0] -= self.learning_rate * update1
        self.set_point[1] -= self.learning_rate * update2

        # update turn probabilities to turn towards eye closer to its set point
        # first calculate differences from set points on both sides
        diff_left = np.abs(brightness_left - self.set_point[0])
        diff_right = np.abs(brightness_right - self.set_point[1])
        # then calculate difference of differences
        diff_left_right = diff_left - diff_right
        # update turn probabilities appropriately
        self.p_left -= self.learning_rate * diff_left_right
        # clip updated values to maintain valid probabilities
        self.p_left = np.clip(self.p_left, 0.0, self.turning_cap)
        self.p_right = self.turning_cap - self.p_left
        self.p_noturn = 1.0 - (self.p_left + self.p_right)
        self.p_noturn = np.clip(self.p_noturn, 0.0, 1.0)

        # move fish
        self.move(environment.shape)

        # step environment
        environment.step()

        # return updated parameters
        return [self.heading,
                self.position[0],
                self.position[1],
                self.set_point[0],
                self.set_point[1],
                self.p_left,
                self.p_right,
                self.p_noturn]

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
        params = [[self.heading,
                   self.position[0],
                   self.position[1],
                   self.set_point[0],
                   self.set_point[1],
                   self.p_left,
                   self.p_right,
                   self.p_noturn]]
        for i in range(timesteps):
            params.append(self.step(environment))
        params = np.stack(params)
        return params

