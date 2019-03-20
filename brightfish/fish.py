import numpy as np

class Fish:
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
        # 1 if turning left
        turn_direction = np.random.binomial(1, self.p_left)
        # -1 if turning right
        if turn_direction == 0:
            turn_direction = -1
        self.heading += turn_direction * self.turning_rate
        self.heading = self.heading % (2 * np.pi)

    def step(self, environment):
        raise NotImplementedError

    def run(self, environment, timesteps):
        raise NotImplementedError

class BinocularFish(Fish):
    def __init__(self, heading, set_point=0.5, learning_rate=5e-2):
        super(BinocularFish, self).__init__(heading, set_point, learning_rate)

    def step(self, environment):
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
        params = [[self.heading, self.set_point, self.p_left, self.p_right]]
        for i in range(timesteps):
            params.append(self.step(environment))
        params = np.stack(params)
        return params

