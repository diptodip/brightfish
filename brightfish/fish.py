import numpy as np
from skimage.draw import polygon

from .utils import pol2cart, nonlinearity


class Fish:
    """
    Base class for defining a simulated zebrafish that respond to brightness. A
    base fish takes at least a heading and position to define its starting
    orientation. Subclasses of fish should implement methods for turning
    logic, what happens in a given time step, and running for multiple time
    steps.

    Args:
        heading (float): Defines the heading in radians of the fish. The fish
        exists in environments defined by 2D arrays where a heading of $0$
        points directly to the right of the array from whatever position in the
        array the fish is in.

        position (list of ints): Defines the position of the fish as an index
        into a 2D array.

        set_point (float, optional): Defines the set point of
        the fish, i.e. the intensity/ies that the fish should seek to turn
        towards.

        max_diff (float, optional): Defines the maximum difference between the
        brightness observed by an eye and the set point that should be
        considered. This is used to normalize the brightness difference between
        an observation from an eye and the set point when deciding how to
        interpolate between the no error and high error turn distribution
        samples.

        learning_rate (float, optional): Defines how fast the fish updates its
        set point and turning probabilities.

        p_move (float, optional): Defines probability of moving on a given time
        step.

        move_dist (tuple, optional): A tuple  of floats defining the mean and
        standard deviation of move distance on a given time step.

        no_turn_dist (tuple, optional): A tuple of floats defining the mean and
        standard deviation of heading change in radians when there is no
        difference between the observed brightnesses in each eye and the set
        point.

        left_turn_dist (tuple, optional): A tuple of floats defining the mean
        and standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the right eye and the set
        point, meaning the fish is turning left. In our coordinate system, a
        change of positive radians is a left (counterclockwise) turn.

        right_turn_dist (tuple, optional): A tuple of floats defining the mean
        and standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the left eye and the set point,
        meaning the fish is turning right. In our coordinate system, a change
        of negative radians is a right (clockwise) turn.

    Attributes:
        heading (float): Defines the heading in radians of the fish. The fish
        exists in environments defined by 2D arrays where a heading of $0$
        points directly to the right of the array from whatever position in the
        array the fish is in.

        position (list of ints): Defines the position of the fish as an index
        into a 2D array.

        set_point (float, optional): Defines the set point of the fish, i.e.
        the intensity/ies that the fish should seek to turn towards.

        max_diff (float): Defines the maximum difference between the brightness
        observed by an eye and the set point that should be considered. This is
        used to normalize the brightness difference between an observation from
        an eye and the set point when deciding how to interpolate between the
        no error and high error turn distribution samples.

        learning_rate (float): Defines how fast the fish updates its set point
        and turning probabilities.

        p_move (float): Defines probability of moving on a given time step.

        move_dist (tuple): A tuple  of floats defining the mean and standard
        deviation of move distance on a given time step.

        no_turn_dist (tuple): A tuple of floats defining the mean and standard
        deviation of heading change in radians when there is no difference
        between the observed brightnesses in each eye and the set point.

        left_turn_dist (tuple): A tuple of floats defining the mean and
        standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the right eye and the set
        point, meaning the fish is turning left. In our coordinate system, a
        change of positive radians is a left (counterclockwise) turn.

        right_turn_dist (tuple): A tuple of floats defining the mean and
        standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the left eye and the set point,
        meaning the fish is turning right. In our coordinate system, a change
        of negative radians is a right (clockwise) turn.
    """

    def __init__(
        self,
        heading,
        position,
        static=False,
        set_point=0.5,
        max_diff=0.75,
        learning_rate=5e-2,
        p_move=0.2,
        move_dist={"mu": 1.0, "sigma": 1.0},
        no_turn_dist={"mu": 0.01, "sigma": 0.50},
        left_turn_dist={"mu": 0.52, "sigma": 0.59},
        right_turn_dist={"mu": -0.52, "sigma": 0.59},
    ):
        self.heading = heading
        self.position = position
        self.static = static
        self.set_point = set_point
        self.max_diff = max_diff
        self.learning_rate = learning_rate
        self.p_move = p_move
        self.move_dist = move_dist
        self.no_turn_dist = no_turn_dist
        self.left_turn_dist = left_turn_dist
        self.right_turn_dist = right_turn_dist

    def __str__(self):
        message = (
            "{0}: heading: {1:.2f} position: {2} set_point: {3:.2f} "
            + "mu_left: {4:.2f} mu_right: {5:.2f}"
        )
        return message.format(
            self.__class__.__name__,
            self.heading,
            self.position,
            self.set_point,
            self.left_turn_dist[0],
            self.right_turn_dist[0],
        )

    def __repr__(self):
        message = (
            "{0}: heading: {1:.2f} position: {2} set_point: {3:.2f} "
            + "mu_left: {4:.2f} mu_right: {5:.2f}"
        )
        return message.format(
            self.__class__.__name__,
            self.heading,
            self.position,
            self.set_point,
            self.left_turn_dist[0],
            self.right_turn_dist[0],
        )

    def turn(self, environment):
        """
        Updates ``self.heading`` by an angle chosen as a weighted mixture of
        samples from the current Fish's high error turning distribution and no
        error turning distribution.
        """
        raise NotImplementedError

    def move(self, environment):
        """
        Updates ``self.position`` by moving in a direction given by
        ``self.heading`` by a distance sampled from a Normal distribution with
        this Fish's movement distribution.
        """
        # update fish heading
        theta = self.turn(environment)
        # decide if moving
        moving = np.random.binomial(1, self.p_move)
        move_distance = 0.0
        # if moving, update position by moving
        # ``move_distance`` in ``self.heading`` direction
        if moving:
            move_distance = np.random.normal(
                self.move_dist["mu"], self.move_dist["sigma"]
            )
            shape = environment.shape
            r, c = pol2cart(move_distance, self.heading, origin=self.position)
            update = (
                (not self.static)
                and (r >= 0)
                and (r < shape[0])
                and (c >= 0)
                and (c < shape[1])
            )
            if update:
                self.position = [r, c]
        return (move_distance, theta)

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
            heading. These values may be used to index directly into a 2D
            array, e.g. ``arr[rr, cc]``.
        """
        radius = max(shape) * 1000
        r1, c1 = pol2cart(
            radius,
            (self.heading + 0.1 * np.pi) % (2 * np.pi),
            origin=self.position,
        )
        r2, c2 = pol2cart(
            radius,
            (self.heading + 0.9 * np.pi) % (2 * np.pi),
            origin=self.position,
        )
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
            its heading. These values may be used to index directly into a 2D
            array, e.g. ``arr[rr, cc]``.
        """
        radius = max(shape) * 1000
        r1, c1 = pol2cart(
            radius,
            (self.heading - 0.1 * np.pi) % (2 * np.pi),
            origin=self.position,
        )
        r2, c2 = pol2cart(
            radius,
            (self.heading - 0.9 * np.pi) % (2 * np.pi),
            origin=self.position,
        )
        r = [self.position[0], r1, r2]
        c = [self.position[1], c1, c2]
        return polygon(r, c, shape=shape)

    def brightness_left(self, environment):
        """Returns the brightness from the left eye's FOV."""
        # collect brightness information from left eye
        left_fov = self.left_eye(environment.shape)
        # check for empty fov (due to being at edge of environment)
        if left_fov[0].size > 0 and left_fov[1].size > 0:
            brightness_left = environment.stage[left_fov[0], left_fov[1]].mean()
        else:
            brightness_left = 0.0
        return brightness_left

    def brightness_right(self, environment):
        """Returns the brightness from the right eye's FOV."""
        # collect brightness information from right eye
        right_fov = self.right_eye(environment.shape)
        # check for empty fov (due to being at edge of environment)
        if right_fov[0].size > 0 and right_fov[1].size > 0:
            brightness_right = environment.stage[right_fov[0], right_fov[1]].mean()
        else:
            brightness_right = 0.0
        return brightness_right


class BinocularFish(Fish):
    """
    Model zebrafish that integrates binocular information to update a set point
    of preferred brightness.

    Args:
        heading (float): Defines the heading in radians of the fish. The fish
        exists in environments defined by 2D arrays where a heading of $0$
        points directly to the right of the array from whatever position in the
        array the fish is in.

        position (list of ints): Defines the position of the fish as an index
        into a 2D array.

        set_point (float, optional): Defines the set point of
        the fish, i.e. the intensity/ies that the fish should seek to turn
        towards.

        max_diff (float, optional): Defines the maximum difference between the
        brightness observed by an eye and the set point that should be
        considered. This is used to normalize the brightness difference between
        an observation from an eye and the set point when deciding how to
        interpolate between the no error and high error turn distribution
        samples.

        learning_rate (float, optional): Defines how fast the fish updates its
        set point and turning probabilities.

        p_move (float, optional): Defines probability of moving on a given time
        step.

        move_dist (tuple, optional): A tuple  of floats defining the mean and
        standard deviation of move distance on a given time step.

        no_turn_dist (tuple, optional): A tuple of floats defining the mean and
        standard deviation of heading change in radians when there is no
        difference between the observed brightnesses in each eye and the set
        point.

        left_turn_dist (tuple, optional): A tuple of floats defining the mean
        and standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the right eye and the set
        point, meaning the fish is turning left. In our coordinate system, a
        change of positive radians is a left (counterclockwise) turn.

        right_turn_dist (tuple, optional): A tuple of floats defining the mean
        and standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the left eye and the set point,
        meaning the fish is turning right. In our coordinate system, a change
        of negative radians is a right (clockwise) turn.

    Attributes:
        heading (float): Defines the heading in radians of the fish. The fish
        exists in environments defined by 2D arrays where a heading of $0$
        points directly to the right of the array from whatever position in the
        array the fish is in.

        position (list of ints): Defines the position of the fish as an index
        into a 2D array.

        set_point (float, optional): Defines the set point of the fish, i.e.
        the intensity/ies that the fish should seek to turn towards.

        max_diff (float): Defines the maximum difference between the brightness
        observed by an eye and the set point that should be considered. This is
        used to normalize the brightness difference between an observation from
        an eye and the set point when deciding how to interpolate between the
        no error and high error turn distribution samples.

        learning_rate (float): Defines how fast the fish updates its set point
        and turning probabilities.

        p_move (float): Defines probability of moving on a given time step.

        move_dist (tuple): A tuple  of floats defining the mean and standard
        deviation of move distance on a given time step.

        no_turn_dist (tuple): A tuple of floats defining the mean and standard
        deviation of heading change in radians when there is no difference
        between the observed brightnesses in each eye and the set point.

        left_turn_dist (tuple): A tuple of floats defining the mean and
        standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the right eye and the set
        point, meaning the fish is turning left. In our coordinate system, a
        change of positive radians is a left (counterclockwise) turn.

        right_turn_dist (tuple): A tuple of floats defining the mean and
        standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the left eye and the set point,
        meaning the fish is turning right. In our coordinate system, a change
        of negative radians is a right (clockwise) turn.
    """

    def __init__(
        self,
        heading,
        position,
        static=False,
        set_point=0.5,
        max_diff=0.75,
        learning_rate=5e-2,
        p_move=0.2,
        move_dist={"mu": 1.0, "sigma": 1.0},
        no_turn_dist={"mu": 0.01, "sigma": 0.50},
        left_turn_dist={"mu": 0.52, "sigma": 0.59},
        right_turn_dist={"mu": -0.52, "sigma": 0.59},
    ):
        super(BinocularFish, self).__init__(
            heading,
            position,
            static=static,
            set_point=set_point,
            max_diff=max_diff,
            learning_rate=learning_rate,
            p_move=p_move,
            move_dist=move_dist,
            no_turn_dist=no_turn_dist,
            left_turn_dist=left_turn_dist,
            right_turn_dist=right_turn_dist,
        )

    def turn(self, environment):
        """
        Updates ``self.heading`` by an angle chosen as a weighted mixture of
        samples from the current Fish's high error turning distribution and no
        error turning distribution.
        """
        # calculate left and right eye differences
        brightness_left = self.brightness_left(environment)
        brightness_right = self.brightness_right(environment)
        diff_left = abs(brightness_left - self.set_point)
        diff_right = abs(brightness_right - self.set_point)
        diff_diff = diff_left - diff_right
        # use sign of diff_left - diff_right to choose turn distribution
        if diff_diff > 0:
            # left is worse, turn right
            turn_dist = self.right_turn_dist
        else:
            # right is worse, turn left
            turn_dist = self.left_turn_dist
        diff_diff = self.max_diff * nonlinearity(diff_diff)
        # calculate turn angle in radians
        no_turn_rad = np.random.normal(
            self.no_turn_dist["mu"], self.no_turn_dist["sigma"]
        )
        turn_rad = np.random.normal(turn_dist["mu"], turn_dist["sigma"])
        theta = turn_rad if np.random.binomial(1, diff_diff) else no_turn_rad
        # update heading by theta radians
        if not self.static:
            self.heading += theta
            self.heading = self.heading % (2 * np.pi)
        # return calculated update
        return theta

    def step(self, environment):
        """
        Defines the behavior of the fish in one time step in the given
        environment. This fish takes the average of brightness information from
        both eyes and updates its set point to be closer to this average.

        Args:
            environment (``Environment``): Defines the environment in which the
            fish takes a step.

        Returns:
            A list of the parameters defining the status of the fish.
        """
        # collect brightness information from both eyes
        brightness_left = self.brightness_left(environment)
        brightness_right = self.brightness_right(environment)

        # calculate differences
        diff_left = abs(brightness_left - self.set_point)
        diff_right = abs(brightness_right - self.set_point)

        # update set point to be closer to mean of two eyes
        update = self.set_point - np.mean([brightness_left, brightness_right])
        self.set_point -= self.learning_rate * update

        # move fish
        (move_distance, theta) = self.move(environment)

        # step environment
        environment.step()

        # return updated parameters
        return {
            "heading": self.heading,
            "r": self.position[0],
            "c": self.position[1],
            "set_point": self.set_point,
            "diff_left": diff_left,
            "diff_right": diff_right,
            "move_distance": move_distance,
            "theta": theta,
        }

    def run(self, environment, timesteps):
        """
        Defines the behavior of the fish over multiple time steps.

        Args:
            environment (``Environment``): Defines the environment in which the
            fish takes a step.

            timesteps (int): Defines the number of time steps to perform.

        Returns:
            A dict mapping parameter names to lists containing the value of
            that parameter for each timestep.
        """
        params = {
            "heading": [self.heading],
            "r": [self.position[0]],
            "c": [self.position[1]],
            "set_point": [self.set_point],
            "diff_left": [0],
            "diff_right": [0],
            "move_distance": [0],
            "theta": [0],
        }
        for i in range(timesteps):
            step_params = self.step(environment)
            for key in step_params.keys():
                if key in params:
                    params[key].append(step_params[key])
                else:
                    params[key] = [step_params[key]]
        return params


class MonocularFish(Fish):
    """
    Model zebrafish that separately uses monocular information to update one
    set point of brightness for each eye.

    Args:
        heading (float): Defines the heading in radians of the fish. The fish
        exists in environments defined by 2D arrays where a heading of $0$
        points directly to the right of the array from whatever position in the
        array the fish is in.

        position (list of ints): Defines the position of the fish as an index
        into a 2D array.

        set_point (tuple, optional): A tuple of floats that defines the set
        points of each eye (left, right) of the fish, i.e. the intensity/ies
        that the fish should seek to turn towards.

        max_diff (float, optional): Defines the maximum difference between the
        brightness observed by an eye and the set point that should be
        considered. This is used to normalize the brightness difference between
        an observation from an eye and the set point when deciding how to
        interpolate between the no error and high error turn distribution
        samples.

        learning_rate (float, optional): Defines how fast the fish updates its
        set point and turning probabilities.

        p_move (float, optional): Defines probability of moving on a given time
        step.

        move_dist (tuple, optional): A tuple  of floats defining the mean and
        standard deviation of move distance on a given time step.

        no_turn_dist (tuple, optional): A tuple of floats defining the mean and
        standard deviation of heading change in radians when there is no
        difference between the observed brightnesses in each eye and the set
        point.

        left_turn_dist (tuple, optional): A tuple of floats defining the mean
        and standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the right eye and the set
        point, meaning the fish is turning left. In our coordinate system, a
        change of positive radians is a left (counterclockwise) turn.

        right_turn_dist (tuple, optional): A tuple of floats defining the mean
        and standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the left eye and the set point,
        meaning the fish is turning right. In our coordinate system, a change
        of negative radians is a right (clockwise) turn.

    Attributes:
        heading (float): Defines the heading in radians of the fish. The fish
        exists in environments defined by 2D arrays where a heading of $0$
        points directly to the right of the array from whatever position in the
        array the fish is in.

        position (list of ints): Defines the position of the fish as an index
        into a 2D array.

        set_point (tuple): A tuple of floats that defines the set points of
        each eye (left, right) of the fish, i.e. the intensity/ies that the
        fish should seek to turn towards.

        max_diff (float): Defines the maximum difference between the brightness
        observed by an eye and the set point that should be considered. This is
        used to normalize the brightness difference between an observation from
        an eye and the set point when deciding how to interpolate between the
        no error and high error turn distribution samples.

        learning_rate (float): Defines how fast the fish updates its set point
        and turning probabilities.

        p_move (float): Defines probability of moving on a given time step.

        move_dist (tuple): A tuple  of floats defining the mean and standard
        deviation of move distance on a given time step.

        no_turn_dist (tuple): A tuple of floats defining the mean and standard
        deviation of heading change in radians when there is no difference
        between the observed brightnesses in each eye and the set point.

        left_turn_dist (tuple): A tuple of floats defining the mean and
        standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the right eye and the set
        point, meaning the fish is turning left. In our coordinate system, a
        change of positive radians is a left (counterclockwise) turn.

        right_turn_dist (tuple): A tuple of floats defining the mean and
        standard deviation of heading change in radians when there is a
        difference of ``self.max_diff`` between the left eye and the set point,
        meaning the fish is turning right. In our coordinate system, a change
        of negative radians is a right (clockwise) turn.
    """

    def __init__(
        self,
        heading,
        position,
        static=False,
        set_point=(0.5, 0.5),
        max_diff=0.75,
        learning_rate=5e-2,
        p_move=0.2,
        move_dist={"mu": 1.0, "sigma": 1.0},
        no_turn_dist={"mu": 0.01, "sigma": 0.50},
        left_turn_dist={"mu": 0.52, "sigma": 0.59},
        right_turn_dist={"mu": -0.52, "sigma": 0.59},
    ):
        super(MonocularFish, self).__init__(
            heading,
            position,
            static=static,
            set_point=set_point,
            max_diff=max_diff,
            learning_rate=learning_rate,
            p_move=p_move,
            move_dist=move_dist,
            no_turn_dist=no_turn_dist,
            left_turn_dist=left_turn_dist,
            right_turn_dist=right_turn_dist,
        )

    def turn(self, environment):
        """
        Updates ``self.heading`` by an angle chosen as a weighted mixture of
        samples from the current Fish's high error turning distribution and no
        error turning distribution.
        """
        # calculate left and right eye differences
        brightness_left = self.brightness_left(environment)
        brightness_right = self.brightness_right(environment)
        # left and right eye use different set points
        diff_left = abs(brightness_left - self.set_point[0])
        diff_right = abs(brightness_right - self.set_point[1])
        diff_diff = diff_left - diff_right
        # use sign of diff_left - diff_right to choose turn distribution
        if diff_diff > 0:
            # left is worse, turn right
            turn_dist = self.right_turn_dist
        else:
            # right is worse, turn left
            turn_dist = self.left_turn_dist
        diff_diff = self.max_diff * nonlinearity(diff_diff)
        # calculate turn angle in radians
        no_turn_rad = np.random.normal(
            self.no_turn_dist["mu"], self.no_turn_dist["sigma"]
        )
        turn_rad = np.random.normal(turn_dist["mu"], turn_dist["sigma"])
        theta = turn_rad if np.random.binomial(1, diff_diff) else no_turn_rad
        # update heading by theta radians
        if not self.static:
            self.heading += theta
            self.heading = self.heading % (2 * np.pi)
        # return calculated update
        return theta

    def step(self, environment):
        """
        Defines the behavior of the fish in one time step in the given
        environment. This fish takes separate brightness information from each
        eye and updates a set point for each eye to be closer to the
        corresponding brightness information.

        Args:
            environment (``Environment``): Defines the environment in
            which the fish takes a step.

        Returns:
            A list of the parameters defining the status of the fish.
        """
        # collect brightness information from both eyes
        brightness_left = self.brightness_left(environment)
        brightness_right = self.brightness_right(environment)

        # calculate differences
        diff_left = abs(brightness_left - self.set_point[0])
        diff_right = abs(brightness_right - self.set_point[1])

        # update set point to be closer to mean of two eyes
        update_left = self.set_point[0] - brightness_left
        update_right = self.set_point[1] - brightness_right
        learning_rate = self.learning_rate
        self.set_point = (
            self.set_point[0] - learning_rate * update_left,
            self.set_point[1] - learning_rate * update_right,
        )

        # move fish
        (move_distance, theta) = self.move(environment)

        # step environment
        environment.step()

        # return updated parameters
        return {
            "heading": self.heading,
            "r": self.position[0],
            "c": self.position[1],
            "set_point": self.set_point,
            "diff_left": diff_left,
            "diff_right": diff_right,
            "move_distance": move_distance,
            "theta": theta,
        }

    def run(self, environment, timesteps):
        """
        Defines the behavior of the fish over multiple time steps.

        Args:
            environment (``Environment``): Defines the environment in which the
            fish takes a step.

            timesteps (int): Defines the number of time steps to perform.

        Returns:
            A dict mapping parameter names to lists containing the value of
            that parameter for each timestep.
        """
        params = {
            "heading": [self.heading],
            "r": [self.position[0]],
            "c": [self.position[1]],
            "set_point": [self.set_point],
            "diff_left": [0],
            "diff_right": [0],
            "move_distance": [0],
            "theta": [0],
        }
        for i in range(timesteps):
            step_params = self.step(environment)
            for key in step_params.keys():
                if key in params:
                    params[key].append(step_params[key])
                else:
                    params[key] = [step_params[key]]
        return params
