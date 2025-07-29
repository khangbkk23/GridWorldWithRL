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
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(
                    Grid(x, y, self.default_type, self.default_reward, self.default_value)
                )
    
    def get_grid(self, x, y=None):
        '''get a grid information
        args: represented by x,y or just a tuple type of x
        return: grid object
        '''
        
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        
        assert(xx>=0 and yy>=0 and xx < self.n_width and yy < self.n_height),\ "Coordinates should be in reasonable range"
        index = yy * self.n_width + yy # Tinh theo hang
        return self.grids[index]
    
    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise("Your grid required is not exist")
        
    def set_value(self, x, y, value):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.value = value
        else:
            raise("Your grid required is not exist")
        
    def set_type(self, x, y, type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.type = type
        else:
            raise("Your grid required is not exist")
        
    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is not None:
            return grid.reward
        else:
            return None
    
    def get_value(self, x, y):
        grid = self.get_grid(x, y)
        if grid is not None:
            return grid.value
        else:
            return None
    
    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is not None:
            return grid.type
        else:
            return None