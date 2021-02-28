import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.search import RandomSearch
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#matplotlib.use("TkAgg")

# Set-up choices for the parameters
def fct_1(b):
    """PSO test function.

    Parameters
    ----------
    b : numpy.ndarray
        sets of inputs shape :code:'(n_particles, 2)'

    Returns
    ----------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """

    cost = 20 + b[:, 0] ** 2 + b[:, 1] ** 2 - 10 * np.cos(2 * np.pi * b[:, 0]) +\
        3 * np.cos(2 * np.pi * b[:, 1])

    return cost

n = 10

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO
np.random.seed(1234)
optimizer = ps.single.GlobalBestPSO(n_particles=n, dimensions=2, options=options, init_pos=np.random.random((10, 2)) * 3 - 1.5)
# Perform optimization
best_cost, best_pos = optimizer.optimize(fct_1, iters=60)

m = 100; n = 100
x, y = np.meshgrid(np.linspace(-2,2,m), np.linspace(-2,2,n))
data = np.array((x, y)).reshape(2, -1).T
z = fct_1(data).reshape((n, m))

plt.figure(figsize=(25, 12))

plt.subplot(2, 2, 1)
plt.title("Initilization")
plt.contourf(x, y, z, 15, cmap='ocean',alpha=0.7)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(np.array(optimizer.pos_history)[0, :, 0],
            np.array(optimizer.pos_history)[0, :, 1],
            c='gold', s=50)

plt.subplot(2, 2, 2)
plt.title("First iteration")
plt.contourf(x, y, z, 15, cmap='ocean',alpha=0.7)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(np.array(optimizer.pos_history)[1, :, 0],
            np.array(optimizer.pos_history)[1, :, 1],
            c='gold', s=50)

plt.subplot(2, 2, 3)
plt.title("Last iteration")
plt.contourf(x, y, z, 15, cmap='ocean',alpha=0.7)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(np.array(optimizer.pos_history)[-1, :, 0],
            np.array(optimizer.pos_history)[-1, :, 1],
            c='gold', s=50)

plt.subplot(2, 2, 4)
plt.title("Best position found by PSO")
plt.contourf(x, y, z, 15, cmap='ocean',alpha=0.7)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(best_pos[0], best_pos[1], c='orange', s=150)
plt.annotate("Best point", xy=(best_pos + 0.1), c='white')

plt.show()