from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import math

class BasketEnv(Env):
    def __init__(self):
        pass
    def step(self):
        pass
    def render(self):
        pass
    def reset(self):
        pass



class BasketEnv(Env):
    def __init__(self, s, h):
        # Calculate velocity
        self.v = (s * math.sqrt(9.8 / (s - 3.05 + h)))
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Velocity array
        self.observation_space = Box(low=np.array([self.v - 3.0]), high=np.array([self.v + 3.0]))
        # Set start velocity
        self.state = self.v + random.randint(-1, 1)
        # Set play length
        self.play_length = 60

    def step(self, action):
        # Apply action
        # 0/20 - 0.05 = -0.05 m/s
        # 1/20 - 0.05 = 0
        # 2/20 - 0.05 = 0.05 m/s
        self.state += (action / 20) - 0.05
        # Reduce play length by 1 second
        self.play_length -= 1

        # Calculate reward
        if self.state >= math.floor(self.v) and self.state <= math.ceil(self.v):
            reward = 1
        else:
            reward = -1

            # Check if play is done
        if self.play_length <= 0:
            done = True
        else:
            done = False

        # Apply velocity noise
        self.state += random.uniform(-0.05, 0.05)
        # Set info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        # Reset velocity
        self.state = self.v + random.randint(-1, 1)
        # Reset play time
        self.play_length = 60
        return self.state