import torch

class Fish:
    def __init__(set_point=0.5, learning_rate=1e-2):
        self.set_point = set_point
        self.learning_rate = learning_rate
        self.p_right = 0.5
        self.p_left = 0.5

    def step(self, environment):
        raise NotImplementedError

    def run(self, environment, timesteps):
        raise NotImplementedError

class BinocularFish(Fish):
    def __init__(set_point=0.5, learning_rate=1e-2):
        super(BinocularFish, self).__init__(set_point)

    def step(self, environment):
        # calculate differences from both eyes
        brightness_left = environment.left_eye().mean()
        brightness_right = environment.right_eye().mean()

        # update set point to be closer to mean of two eyes
        update = self.set_point - np.mean([brightness_left, brightness_right])
        self.set_point -= learning_rate * update

        # update turn probabilities to turn towards area closer to set point
        diff_left = np.abs(brightness_left - self.set_point)
        diff_right = np.abs(brightness_right - self.set_point)
        diff_right_left = diff_right - diff_left
        self.p_right += self.learning_rate * diff_right_left
        self.p_left -= self.learning_rate * diff_right_left
        self.p_right = np.clip(self.p_right, 0.0, 1.0)
        self.p_left = np.clip(self.p_left, 0.0, 1.0)

        # return updated parameters
        return [self.set_point, self.p_right, self.p_left]

    def run(self, environment, timesteps):
        env_steps = environment.run(timesteps)
        params = [[self.set_point, self.p_right, self.p_left]]
        for i in range(timesteps):
            params.append(self.step(env_steps[i]))
        params = np.stack(params)
        return params

