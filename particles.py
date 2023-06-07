import numpy as np
from numpy import pi, cos, sin, arctan2
import scipy.spatial as sp
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import itertools

class Particles:
    title = 'Particles Positions and velocity'

    def __init__(self, N:int=150, L:float=7, v:float=0.03,
                 r:float=1.0, eta:float=1.5, dt:float=2.0,
                 is_extrinstic_noise:bool = False):
        self.N = N
        self.L = L
        self.v = v
        self.r = r
        self.eta = eta
        self.dt = dt
        self.is_extrinstic_noise = is_extrinstic_noise

        self.pos = np.random.uniform(low=0.0, high=L, size=(2,N))
        self.dir = np.random.uniform(low=-pi, high=pi, size=N)
        self.update_vel()
    
    def update_vel(self):
        self.vel = self.v*np.array([cos(self.dir),sin(self.dir)]) 
        
    def update_Pos(self):
        self.pos += self.vel * self.dt
        self.pos %= self.L

    def neighbors(self):
        tree = sp.KDTree(self.pos.T, boxsize=(self.L,self.L))
        neigh = tree.query_ball_tree(tree, r=self.r)
        neigh = np.array(list(itertools.zip_longest(*neigh, fillvalue=-1))).T
        return neigh

    
    def update_dir(self):
        neigh = self.neighbors()
        num_neigh = np.zeros(self.N)

        avg_sin = sin(self.dir[neigh])
        avg_sin[neigh == -1] = 0
        avg_sin = avg_sin.sum(axis=1)

        avg_cos = cos(self.dir[neigh])
        avg_cos[neigh == -1] = 0
        avg_cos = avg_cos.sum(axis=1)

        num_neigh = (neigh != -1).sum(axis=1)
        avg_sin = np.divide(avg_sin, num_neigh)
        avg_cos = np.divide(avg_cos, num_neigh)

        instrinsic_noise = 0.0
        if not self.is_extrinstic_noise:
            instrinsic_noise = np.random.uniform(low=-self.eta/2,high=self.eta/2,size=self.N)
        else:
            noise_dir = np.random.uniform(low=-pi, high=pi, size=self.N)
            avg_sin += self.eta * sin(noise_dir)
            avg_cos += self.eta * cos(noise_dir)

        self.dir = arctan2(avg_sin, avg_cos)
        self.dir += instrinsic_noise
    
    def update(self):
        self.update_dir()
        self.update_vel()
        self.update_Pos()

    #Plot the state 
    def plot_state(self):
        # color = np.full(self.N, 'k')
        # neigh = self.neighbors()
        # color[neigh[0]] = 'c'
        # color[0] = 'r'
        plt.quiver(self.pos[0], self.pos[1], cos(self.dir), sin(self.dir), scale=70)
        plt.title(Particles.title)
        plt.xlim(0,self.L)
        plt.ylim(0,self.L)
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
    
    def avg_vel(self):
        tot_vel = np.sum(self.vel, axis=1)
        return np.linalg.norm(tot_vel) / (self.N * self.v)
    
    def avg_vel_run(self, iter_before: int, iter_to_avg: int):
        for i in range(iter_before):
            self.update()
        
        avg_va = 0
        for i in range(iter_to_avg):
            self.update()
            avg_va += self.avg_vel()
        avg_va /= iter_to_avg

        return avg_va