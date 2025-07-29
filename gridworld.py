import math
import gym

from gym import spaces
from gym.utils import seeding
import numpy as np

class Grid(object):
    def __init__(self, x: int = None, y: int = None, type: int = 0, reward: int = 0.0, value: float = 0.0):
        self.x = x
        self.y = y
        self.type = type
        self.reward = reward
        self.value = value
        self.name = None
        self._update_name()
    
    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x, self.y)
    
    def __str__(self):
        return "name:{4}, x:{0}, y:{1}, type:{2}, value{3}".format(self.x,
                                                                   self.y,
                                                                   self.type,
                                                                   self.reward,
                                                                   self.value,
                                                                   self.name
                                                                    )

class GridMatrix(object):
    
    def __init__(self, n_width:int,                     # defines the number of cells horizontally
                       n_height:int,                    # vertically
                       default_type: int = 0,           # default cell type
                       default_reward: float = 0.0,     # default instant reward
                       default_value: float = 0.0       # default value
                       ):
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type
        self.reset()
    
    def reset(self):
        pass