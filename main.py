import numpy as np
from numpy import pi, cos, sin, arctan2
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import math

# constants
k = 100
dt = 2.0
num_pts = 50
num_runs_per_pt = 20
Ns = {
    40: 's',
    200: '+',
    # 400: 'x'
}
# 4000: '^',
# 10000: 'D'

class Particles:
    title = 'Particles Positions and velocity'

    def __init__(self, N:int=100, L:float=7.0, v:float=0.03,
                 r:float=1.0, eta:float=1.5):
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
        # needs to be 2-dimensional for pdist
        # so these are 2d with only 1 row
        xs = np.reshape(self.pos[0], (-1,1))
        ys = np.reshape(self.pos[1], (-1,1))

        x_dif = squareform(pdist(xs)) % (self.L/2)
        y_dif = squareform(pdist(ys)) % (self.L/2)
        distances = np.sqrt(x_dif**2 + y_dif**2)
        # distances = squareform(pdist(self.pos.T))
        return distances < self.r
    
    def update_dir(self):
        adj = self.adjacency_matrix()
        dir_tiled = np.tile(self.dir, (self.N,1))

        num_neigh = np.sum(adj, axis=1)

        # average sin of neighbors
        avg_sin = np.divide(np.sum(np.multiply(adj, sin(dir_tiled)),axis=1), num_neigh)

        # average cos of neighbors
        avg_cos = np.divide(np.sum(np.multiply(adj, cos(dir_tiled)),axis=1), num_neigh)

        noise_dir = np.random.uniform(low=-pi, high=pi, size=self.N)
        avg_sin += self.eta * sin(noise_dir)
        avg_cos += self.eta * cos(noise_dir)

        self.dir = arctan2(avg_sin, avg_cos)
        # self.dir += np.random.uniform(low=-self.eta/2,high=self.eta/2,size=self.N)
    
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
        plt.title(Particles.title + f' at time={frame * dt:.6f}[sec]')

    fig = plt.figure()
    animation = FuncAnimation(fig, update, interval=10)

    plt.show()

def va_particles_run(N, eta, num_iter:int=100, rho:float=4.0):
    # changes L to have constant rho
    st = Particles(N=N, eta=eta, L=np.sqrt(N/rho))
    for i in range(num_iter):
        st.update()
    # st.plot_state()
    # plt.title(f'N={N}, eta={st.eta}, va={st.avg_vel()}')
    # plt.show()
    return st.avg_vel()


def plot_v_eta():
    # x axis
    eta_pts = np.linspace(0.0, 5.0, num_pts)
    va_pts = {}

    # run for multiple numbers of particles
    for N in Ns:
        print(f'\nrunning for {N} particles...')

        avg_vel_pts = []
        for i, eta in enumerate(eta_pts):
            # print progress bar
            sys.stdout.write('\r')
            p = int(math.ceil(20 * i/num_pts))
            sys.stdout.write("[%-20s] %d%%" % ('='*p, 5*p))
            sys.stdout.flush()

            avg = 0.0
            for j in range(num_runs_per_pt):
                avg += va_particles_run(N, eta, num_iter=100)
            avg /= num_runs_per_pt
            avg_vel_pts.append(avg)
        va_pts[N] = avg_vel_pts
    
    for N in Ns:
        plt.scatter(eta_pts, va_pts[N], marker=Ns[N], label=f'N={N}')
    plt.title('$v_a$ as a function of $\eta$')
    plt.xlabel('$\eta$')
    plt.ylabel('$v_a$')
    plt.legend()
    plt.show()

def plot_v_rho():
    # x axis
    rho_pts = np.linspace(0.0, 10.0, num_pts)
    va_pts = {}
    eta = 0.2
    L = 20
    
    avg_vel_pts = []
    for i, rho in enumerate(rho_pts):
        N = int(np.round(rho * L**2))
        # print progress bar
        sys.stdout.write('\r')
        p = int(math.ceil(20 * i/num_pts))
        sys.stdout.write("[%-20s] %d%%" % ('='*p, 5*p))
        sys.stdout.flush()

        avg = 0.0
        for j in range(num_runs_per_pt):
            st = Particles(N=N, eta=eta, L=L)
            for i in range(400):
                st.update()
            avg += st.avg_vel()
        avg /= num_runs_per_pt
        avg_vel_pts.append(avg)
    
    
    plt.scatter(rho_pts, avg_vel_pts, marker='s')
    plt.title('$v_a$ as a function of $\rho$')
    plt.xlabel('$\rho$')
    plt.ylabel('$v_a$')
    plt.legend()
    plt.show()

animate()
plot_v_eta()