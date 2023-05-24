import numpy as np
from numpy import pi, cos, sin

# constants
N = 1000
L = 64.0
v = 100.0
r = 1.0

class State:
    def __init__(self):
        self.dir = np.random.uniform(low=0.0, high=2*pi, size=N)
        self.pos = np.random.uniform(low=0.0, high=L, size=(N,2))
        self.vel = np.array(v*cos(dir),v*sin(dir))

# def find_neighbors(state: State):
