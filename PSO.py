
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# generating the working space and the fitness function---Gaussion Distribution
def gaussion_func(x, y):
    z = (1/2*np.pi) * np.exp(-1/10 * ((x+5)**2 + (y+5)**2)) + (1/2*np.pi) * np.exp(-1/20 * ((x-5)**2 + (y-5)**2))*0.6
    return z


x = np.arange(-15, 15, 0.5)
y = np.arange(-15, 15, 0.5)
xx, yy = np.meshgrid(x, y)
z = gaussion_func(xx, yy)

figure = plt.figure()
ax = Axes3D(figure)
ax.plot_surface(xx, yy, z, cmap='jet')
plt.show()


# the Particle Swarm Optimization (PSO) process
# hyper-parameters
class Options:
    def __init__(self):
        self.num_particles = 10   # number of used particles
        self.c_1 = 2              # learning factor c1 for partial bests
        self.c_2 = 2              # learning factor c2 for global best
        self.x_range = 15         # limitation of particle position
        self.v_range = 0.5        # limitation of particle velocity
        self.w_init = 1.5         # initial value of momentum for velocity
        self.e_end = 0.4          # end value of momentum for velocity
        self.iterations = 100     # number of iterations


# init particle states
options = Options()
x = np.random.rand(options.num_particles, 2) * 20 - 10  # initial positions
v = np.random.rand(options.num_particles, 2)            # initial velocity
pBest = np.zeros(shape=(options.num_particles, 2))      # positions of partial best fitness
gBest = np.zeros(shape=(1, 2))                          # position of global best fitness
fBest = np.zeros(shape=(options.num_particles, ))       # partial best fitness

# main loop
plt.ion()
for i in range(options.iterations):
    fitness = gaussion_func(x[:,0], x[:, 1])
    pBest_update = (fitness >= fBest)
    fBest[pBest_update] = fitness[pBest_update]
    pBest[pBest_update] = x[pBest_update]
    gBest = x[fBest.argmax()]
    w = (options.w_init - options.e_end)*(options.iterations - i)/options.iterations + options.e_end

    v = w*v + options.c_1*np.random.rand()*(pBest-x) + options.c_2*np.random.rand()*(gBest-x)
    v = np.clip(v, a_min=-options.v_range, a_max=options.v_range)
    x = x + v
    x = np.clip(x, a_min=-options.x_range, a_max=options.x_range)

    print(f'Best fitness: {np.max(fBest, axis=0):.3f};   Position of best fitness: {x[fBest.argmax()]}')

    # visualization
    plt.clf()
    plt.pcolormesh(xx, yy, z)
    plt.scatter(x[:,0], x[:,1], c='red')
    plt.pause(0.01)
plt.ioff()
