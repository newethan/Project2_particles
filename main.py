import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from progress import Bar
from particles import Particles

# constants
num_pts = 15
num_runs_per_pt = 5
Ns = {
    40: 's',
    100: '+',
    400: 'x',
    4000: '^',
    # 10000: 'D',
}


def animate():
    st = Particles()

    def update(frame):
        plt.cla()
        st.update()
        st.plot_state()
        plt.title(Particles.title + f' at time={frame * st.dt:.6f}[sec]')

    fig = plt.figure()
    animation = FuncAnimation(fig, update, interval=10)

    plt.show()

def va_particles_run(N, eta, iter_before:int, iter_to_avg:int, rho:float=4.0):
    # changes L to have constant rho
    st = Particles(N=N, eta=eta, L=np.sqrt(N/rho))
    return st.avg_vel_run(iter_before, iter_to_avg)


def plot_v_eta():
    # x axis
    eta_pts = np.linspace(0.0, 5.0, num_pts)
    va_pts = {}

    # run for multiple numbers of particles
    for N in Ns:
        print(f'\nrunning for {N} particles...')

        avg_vel_pts = []
        
        # init progress bar
        bar = Bar()

        for i, eta in enumerate(eta_pts):
            avg = 0.0
            for j in range(num_runs_per_pt):
                avg += va_particles_run(N, eta, iter_before=400, iter_to_avg=80)
            avg /= num_runs_per_pt
            avg_vel_pts.append(avg)

            # update progress bar
            bar.update((i+1)/num_pts)
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
    rho_pts = np.linspace(0.1, 2.0, num_pts)
    va_pts = {}
    eta = 0.2
    L = 20.0
    
    avg_vel_pts = []
    bar = Bar()
    for i, rho in enumerate(rho_pts):
        N = int(np.ceil(rho * L**2))

        bar.update((i+1)/num_pts)

        avg = 0.0
        for j in range(num_runs_per_pt):
            st = Particles(N=N, eta=eta, L=L)
            for i in range(400):
                st.update()
            avg += st.avg_vel()
        avg /= num_runs_per_pt
        avg_vel_pts.append(avg)
    
    
    plt.scatter(rho_pts, avg_vel_pts, marker='s')
    plt.title('$v_a$ as a function of $\\rho$')
    plt.xlabel('$\\rho$')
    plt.ylabel('$v_a$')
    # plt.legend()
    plt.show()

# animate()
plot_v_rho()