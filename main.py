import numpy as np
from numpy import pi, cos, sin, arctan2
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# constants
N = 500
L = 7.0
v = 0.03
r = 0.2
k = 100
h = 1e-6
eta = 1.5
dt = 2


class Particles:
    title = 'Particles Positions and velocity'

    def __init__(self):
        self.dir = np.random.uniform(low=-pi, high=pi, size=N)
        self.pos = np.random.uniform(low=0.0, high=L, size=(2,N))
        self.update_vel()
    
    def update_vel(self):
        self.vel = np.array([v*cos(self.dir),v*sin(self.dir)]) 
        
    def update_Pos(self):
        self.pos += self.vel * dt
        self.pos %= L

    def adjacency_matrix(self):
        distances = squareform(pdist(self.pos.T))
        return distances < r
    
    def update_dir(self):
        adj = self.adjacency_matrix()
        dir_tiled = np.tile(self.dir, (N,1))

        num_neigh = np.sum(adj, axis=0)

        # average sin of neighbors
        avg_sin = np.divide(np.sum(np.multiply(adj, sin(dir_tiled)),axis=1), num_neigh)

        # average cos of neighbors
        avg_cos = np.divide(np.sum(np.multiply(adj, cos(dir_tiled)),axis=1), num_neigh)

        self.dir = arctan2(avg_sin, avg_cos) + np.random.uniform(low=-eta/2,high=eta/2,size=N)

    #Plot the state 
    def plot_state(self):
        plt.quiver(self.pos[0], self.pos[1], cos(self.dir), sin(self.dir), scale=70)
        plt.title(Particles.title)
        plt.xlim(0,L)
        plt.ylim(0,L)
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')

st = Particles()
#PlotState(st)


def update(frame):
    plt.cla()
    st.update_Pos()
    st.update_dir()
    st.update_vel()
    st.plot_state()
    plt.title(Particles.title + f' at time={frame * dt:.6f}[sec]')


fig = plt.figure()
animation = FuncAnimation(fig, update, interval=10)


plt.show()
print('hi')