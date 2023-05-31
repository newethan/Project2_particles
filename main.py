import numpy as np
from numpy import pi, cos, sin, arctan2
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# constants
k = 100
dt = 2.0
num_pts = 50

class Particles:
    title = 'Particles Positions and velocity'

    def __init__(self, N:int=150, L:float=7.0, v:float=0.03,
                 r:float=1, eta:float=1.5):
        self.N = N
        self.L = L
        self.v = v
        self.r = r
        self.eta = eta

        self.pos = np.random.uniform(low=0.0, high=L, size=(2,N))
        self.dir = np.random.uniform(low=-pi, high=pi, size=N)
        self.update_vel()
    
    def update_vel(self):
        self.vel = self.v*np.array([cos(self.dir),sin(self.dir)]) 
        
    def update_Pos(self):
        self.pos += self.vel * dt
        self.pos %= self.L

    def adjacency_matrix(self):
        distances = squareform(pdist(self.pos.T))
        return distances < self.r
    
    def update_dir(self):
        adj = self.adjacency_matrix()
        dir_tiled = np.tile(self.dir, (self.N,1))

        num_neigh = np.sum(adj, axis=0)

        # average sin of neighbors
        avg_sin = np.divide(np.sum(np.multiply(adj, sin(dir_tiled)),axis=1), num_neigh)

        # average cos of neighbors
        avg_cos = np.divide(np.sum(np.multiply(adj, cos(dir_tiled)),axis=1), num_neigh)

        self.dir = arctan2(avg_sin, avg_cos)
        self.dir += np.random.uniform(low=-self.eta/2,high=self.eta/2,size=self.N)
    
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


def animate():
    st = Particles()

    def update(frame):
        plt.cla()
        st.update()
        st.plot_state()
        print(st.avg_vel())
        plt.title(Particles.title + f' at time={frame * dt:.6f}[sec]')

    fig = plt.figure()
    animation = FuncAnimation(fig, update, interval=10)

    plt.show()

def plot_v_eta():
    eta_pts = np.linspace(0.0, 5.0, num_pts)
    avg_vel_pts = []
    st = Particles()
    for eta in eta_pts:
        st.eta = eta
        for i in range(1500):
            st.update()
        va = st.avg_vel()
        # st.plot_state()
        # plt.title(f'eta = {eta}, va = {va}')
        # plt.show()
        avg_vel_pts.append(va)
    
    plt.plot(eta_pts, avg_vel_pts)
    plt.show()

animate()
plot_v_eta()