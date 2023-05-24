import numpy as np
from numpy import pi, cos, sin
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
        self.vel = np.array([v*cos(self.dir),v*sin(self.dir)])

    def update_Pos(self, dt: float = 1e-6):
        self.pos += self.vel * dt
        self.pos %= L
        
    

# def find_neighbors(state: State):

#Plot the state 
def PlotState(St: State):

    color_set = ['lightcoral', 'moccasin', 'palegreen', 'aquamarine', 'lightskyblue', 'plum', 'lightcoral']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color_set)

    plt.quiver(St.pos[0], St.pos[1], cos(St.dir), sin(St.dir), scale=70)

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