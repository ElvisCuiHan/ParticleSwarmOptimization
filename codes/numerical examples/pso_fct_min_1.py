import pyswarms as ps
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

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

def fct_2(b):
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
    cost = 1 + b[:, 0] * np.sin(4 * np.pi * b[:, 1]) - b[:, 1] * np.sin(4 * np.pi * b[:, 0] + np.pi)

    return cost

def fct_3(b):
    """PSO test function: Keane function.

    Parameters
    ----------
    b : numpy.ndarray
        sets of inputs shape :code:'(n_particles, 2)'

    Returns
    ----------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    #min = np.zeros(b.shape)
    #max = 10 * np.ones(b.shape)
    #b = np.minimum(np.maximum(b, min), max)
    cost = -(np.sin(b[:, 0] - b[:, 1]) ** 2) * (np.sin(b[:, 0] + b[:, 1]) ** 2) / np.sqrt(b[:, 0] ** 2 + b[:, 1] ** 2)

    return cost

n = 10

# Set-up hyperparameters
to_optim = fct_3
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO
np.random.seed(123)
optimizer = ps.single.GlobalBestPSO(n_particles=n, dimensions=2, options=options, init_pos=np.random.random((10, 2)) * 2)
# Perform optimization
best_cost, best_pos = optimizer.optimize(to_optim, iters=60)

m = 100; n = 100
x, y = np.meshgrid(np.linspace(-5,5,m), np.linspace(-5,5,n))
data = np.array((x, y)).reshape(2, -1).T
z = to_optim(data).reshape((n, m))

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

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-1.01, -.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()