import numpy as np
from numpy import pi, cos, sin, arctan2
import scipy.spatial as sp
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt

class Particles:
    title = 'Particles Positions and velocity'

    def __init__(self, N:int=10, L:float=7.0, v:float=0.03,
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

    def adjacency_matrix(self):
        tree = sp.cKDTree(self.pos.T, boxsize=(self.L,self.L))
        adj = np.full((self.N, self.N), False)
        pair_coords = tree.query_pairs(self.r,output_type='ndarray').T
        adj[pair_coords[0], pair_coords[1]] = True
        adj[pair_coords[1], pair_coords[0]] = True
        np.fill_diagonal(adj, True)
        return adj

    
    def update_dir(self):
        adj = self.adjacency_matrix()
        dir_tiled = np.tile(self.dir, (self.N,1))

        num_neigh = np.sum(adj, axis=1)

        # average sin of neighbors
        avg_sin = np.divide(np.sum(np.multiply(adj, sin(dir_tiled)),axis=1), num_neigh)

        # average cos of neighbors
        avg_cos = np.divide(np.sum(np.multiply(adj, cos(dir_tiled)),axis=1), num_neigh)

        instrinsic_noise = 0
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