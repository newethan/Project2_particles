import numpy as np
from numpy import pi, cos, sin, arctan2
import scipy.spatial as sp
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import itertools

# class to represent sample of particles
class Particles:
    # title for graphs
    title = 'Particles Positions and velocity'

    # initialize sample
    def __init__(self, N:int=150, L:float=7, v:float=0.03,
                 r:float=1.0, eta:float=1.5, dt:float=2.0,
                 is_extrinstic_noise:bool = False):
        self.N = N # number of particles
        self.L = L # side length of sample
        self.v = v # particle velocity
        self.r = r # neighbor detection radius
        self.eta = eta # noise strength
        self.dt = dt # timestep
        self.is_extrinstic_noise = is_extrinstic_noise # whether noise is extrinsic or intrinsic nosie

        self.pos = np.random.uniform(low=0.0, high=L, size=(2,N)) # random positions
        self.dir = np.random.uniform(low=-pi, high=pi, size=N) # random directions
        self.update_vel() # update velocity vectors accordingly
    
    # update particle velocity vectors
    def update_vel(self):
        # update velocity vectors according to direction
        self.vel = self.v*np.array([cos(self.dir),sin(self.dir)]) 
        
    # update particle positions
    def update_Pos(self):
        # assume constant velocity and update position
        self.pos += self.vel * self.dt
        self.pos %= self.L # periodic boundary condition

    # find neighbors using kdtree method
    def neighbors(self):
        tree = sp.KDTree(self.pos.T, boxsize=(self.L,self.L)) # init kdtree
        neigh = tree.query_ball_tree(tree, r=self.r) # query all neighbor pairs within radius r of each other
        neigh = np.array(list(itertools.zip_longest(*neigh, fillvalue=-1))).T # make neighbor indices into matrix
        return neigh

    # update particle directions
    def update_dir(self):
        neigh = self.neighbors()
        num_neigh = np.zeros(self.N)

        # average direction of neighbors
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
        if not self.is_extrinstic_noise: # noise is intrinsic
            # choose random vector to add to calculated direction
            instrinsic_noise = np.random.uniform(low=-self.eta/2,high=self.eta/2,size=self.N)
        else: # noise is extrinsic
            # add random vector to neighbor direction vector sum BEFORE calculating the average direction from it
            noise_dir = np.random.uniform(low=-pi, high=pi, size=self.N)
            avg_sin += self.eta * sin(noise_dir)
            avg_cos += self.eta * cos(noise_dir)

        self.dir = arctan2(avg_sin, avg_cos)
        self.dir += instrinsic_noise # if noise is extrinsic, this is 0
    
    # update all particle parameters
    def update(self):
        self.update_dir()
        self.update_vel()
        self.update_Pos()

    #Plot the state 
    def plot_state(self):
        # arrows for particles
        plt.quiver(self.pos[0], self.pos[1], cos(self.dir), sin(self.dir), scale=70)

        # title and graph details
        plt.title(Particles.title)
        plt.xlim(0,self.L)
        plt.ylim(0,self.L)
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
    
    # calculate v_a, kinetic order parameter
    def avg_vel(self):
        tot_vel = np.sum(self.vel, axis=1) # sum of all particle velocities
        return np.linalg.norm(tot_vel) / (self.N * self.v) # get magnitude of sum, normalize it
    
    # calculate average velocity by averaging over many timesteps
    def avg_vel_run(self, iter_before: int, iter_to_avg: int):
        # iters before starting to calculate v_a average
        for i in range(iter_before):
            self.update()
        
        # iters to calculate v_a average
        avg_va = 0
        for i in range(iter_to_avg):
            self.update()
            avg_va += self.avg_vel()
        avg_va /= iter_to_avg

        return avg_va