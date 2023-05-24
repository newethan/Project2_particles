import numpy as np
from numpy import pi, cos, sin, arctan
from scipy.spatial.distance import pdist, squareform
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# constants
N = 1000
L = 64.0
v = 100.0
r = 1.0
k = 100
h = 1e-6


class State:
    def __init__(self):
        self.dir = np.random.uniform(low=0.0, high=2*pi, size=N)
        self.pos = np.random.uniform(low=0.0, high=L, size=(2,N))
        self.update_vel()
    
    def update_vel(self):
        self.vel = np.array([v*cos(self.dir),v*sin(self.dir)]) 

    def adjacency_matrix(self):
        distances = squareform(pdist(self.pos.T))
        return distances < r
    
    def update_dir(self):
        adj = self.adjacency_matrix()
        dir_tiled = np.tile(self.dir, (N,1))
        avg_sin = np.sum(np.multiply(adj, sin(dir_tiled)),axis=1) / N
        avg_cos = np.sum(np.multiply(adj, cos(dir_tiled)),axis=1) / N
        self.dir = arctan(avg_sin / avg_cos)
    
    #Plot the state 
    def PlotState(self):
        plt.quiver(self.pos[0], self.pos[1], cos(self.dir), sin(self.dir), scale=70)
        plt.title('Particles Positions and velocity')



st = State()
#PlotState(st)


def update (frame):
    plt.cla()
    print(st.vel)
    st.update_Pos(0.001)
    PlotState(st)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.title(f'Time = {frame * h:.6f} sec')
    plt.grid(True)


fig = plt.figure()
animation = FuncAnimation(fig, update, frames=k, interval=10)


plt.show()