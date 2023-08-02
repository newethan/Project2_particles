import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
from matplotlib.animation import FuncAnimation
from progress import Bar
from particles import Particles

# constants
num_pts = 30
num_runs_per_pt = 30
Ns = { # dict - numner of particles and corresponding marker on graph
    40: 's',
    100: '+',
    400: 'x',
    # 4000: '^',
    # 10000: 'D',
}

#################################
# Experiments 
#################################

# animate sample
def animate():
    # create sample
    st = Particles()
    
    # new frame
    def update(frame):
        plt.cla()
        st.update()
        st.plot_state()
        plt.title(Particles.title + f' at time={frame * st.dt:.6f}[sec]')

    # matplotlib animation
    fig = plt.figure()
    animation = FuncAnimation(fig, update, interval=10)

    plt.show()

# calculate kinetic order parameter v_a by running many iterations and veraging over them
def va_particles_run(N, eta, iter_before:int, iter_to_avg:int, rho:float=4.0):
    # changes L to have constant rho
    st = Particles(N=N, eta=eta, L=np.sqrt(N/rho), is_extrinstic_noise=False)
    return st.avg_vel_run(iter_before, iter_to_avg)

# plot kinetic order parameter v_a as a function of noise strength eta
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

        # calculate for each eta point, doing several runs
        for i, eta in enumerate(eta_pts):
            avg = 0.0
            for j in range(num_runs_per_pt):
                avg += va_particles_run(N, eta, iter_before=400, iter_to_avg=2000)
            avg /= num_runs_per_pt
            avg_vel_pts.append(avg)

            # update progress bar
            bar.update((i+1)/num_pts)
        va_pts[N] = avg_vel_pts
    
    # plot for each N, graph title and details
    for N in Ns:
        plt.scatter(eta_pts, va_pts[N], marker=Ns[N], label=f'N={N}')
    plt.title('$v_a$ as a function of $\eta$')
    plt.xlabel('$\eta$')
    plt.ylabel('$v_a$')
    plt.legend()
    plt.show()

    # plot derivative of previous function for each N, graph title and details
    for N in Ns:
        va_dif = (va_pts[N] - np.roll(va_pts[N], 1))[1:]
        eta_shifted = (eta_pts - np.roll(eta_pts, 1))[1:]
        deriv = va_dif / eta_shifted
        plt.scatter(eta_pts[1:], deriv, marker=Ns[N], label=f'N={N}')
    plt.title('$(\\frac {\partial v_a} {\partial \eta}$) as a function of $\eta$')
    plt.xlabel('$\eta$')
    plt.ylabel('$(\\frac {\partial v_a} {\partial \eta})$')
    plt.legend()
    plt.show()

# plot kinetic order parameter v_a as a function of noise strength eta near critical noise
def plot_v_eta_crit():
    # x axis
    eta_crit = 2.9

    nu_pts = np.linspace(0.01, 1.0, num_pts)
    eta_pts = eta_crit - nu_pts * eta_crit
    va_pts = {}

    # run for multiple numbers of particles
    for N in Ns:
        print(f'\nrunning for {N} particles...')

        avg_vel_pts = []
        
        # init progress bar
        bar = Bar()

        # calculate for each eta point, doing several runs
        for i, eta in enumerate(eta_pts):
            avg = 0.0
            for j in range(num_runs_per_pt):
                avg += va_particles_run(N, eta, iter_before=100, iter_to_avg=80)
            avg /= num_runs_per_pt
            avg_vel_pts.append(avg)

            # update progress bar
            bar.update((i+1)/num_pts)
        va_pts[N] = avg_vel_pts
    
    log_nu_pts = np.log10(nu_pts)
    log_va_pts = {}
    for N in Ns:
        log_va_pts[N] = np.log10(va_pts[N])
    
    # plot, graph title and details
    for N in Ns:
        plt.scatter(nu_pts, va_pts[N], marker=Ns[N], label=f'N={N}')
    plt.title('$v_a$ as a function of $\\nu$')
    plt.xlabel('$\\nu$')
    plt.ylabel('$v_a$')

    # get linear fit
    z = np.polyfit(log_nu_pts, log_va_pts[400], 1)
    p = np.poly1d(z)
    plt.plot(nu_pts, 10**p(log_nu_pts), 'r--')

    # set axes to log and show linear fit details on graph
    ax = plt.gca()
    text = f'$y={z[0]:0.3f}\;x{z[1]:+0.3f}$'
    ax.text(0.05, 0.95, text,transform=plt.gca().transAxes,
    fontsize=14, verticalalignment='bottom')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend()
    plt.show()

# plot kinetic order parameter v_a as a function of density rho
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

        # calculate for each rho point, doing several runs
        avg = 0.0
        for j in range(num_runs_per_pt):
            st = Particles(N=N, eta=eta, L=L)
            for i in range(400):
                st.update()
            avg += st.avg_vel()
        avg /= num_runs_per_pt
        avg_vel_pts.append(avg)
    
    # plot, graph title and details
    plt.scatter(rho_pts, avg_vel_pts, marker='s')
    plt.title('$v_a$ as a function of $\\rho$')
    plt.xlabel('$\\rho$')
    plt.ylabel('$v_a$')
    plt.show()

# plot kinetic order parameter v_a as a function of density rho near critical density
def plot_v_rho_crit():
    eta = 2.0
    L = 7.0

    rho_crit = 0.25

    nu_pts = np.logspace(np.log10(0.1), np.log10(10.0), num_pts)

    rho_pts = nu_pts * rho_crit + rho_crit

    va_pts = []
    
    # init progress bar
    bar = Bar()

    # calculate for each rho point, doing several runs
    for i, rho in enumerate(rho_pts):
        avg = 0.0
        for j in range(num_runs_per_pt):
            st = Particles(N=int(np.ceil(L**2 * rho)), eta=eta, L=L)
            avg += st.avg_vel_run(iter_before=100, iter_to_avg=80)
        avg /= num_runs_per_pt
        va_pts.append(avg)

        # update progress bar
        bar.update((i+1)/num_pts)

    log_nu_pts = np.log10(nu_pts)
    log_va_pts = np.log10(va_pts)

    # plot, graph title and details
    plt.scatter(nu_pts, va_pts, marker='s')
    plt.title('$v_a$ as a function of $\\nu$')
    plt.xlabel('$\\nu$')
    plt.ylabel('$v_a$')

    # get linear fit
    z = np.polyfit(log_nu_pts, log_va_pts, 1)
    p = np.poly1d(z)
    plt.plot(nu_pts, 10**p(log_nu_pts), 'r--')

    # set axes to log and show linear fit details on graph
    ax = plt.gca()
    text = f'$y={z[0]:0.3f}\;x{z[1]:+0.3f}$'
    ax.text(0.05, 0.95, text,transform=plt.gca().transAxes,
    fontsize=14, verticalalignment='bottom')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend()
    plt.show()


# run all experiments
animate()
plot_v_eta()
plot_v_eta_crit()
plot_v_rho()
plot_v_rho_crit()