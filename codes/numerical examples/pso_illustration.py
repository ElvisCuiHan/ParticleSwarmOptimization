import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import scipy

np.random.seed(1)

origin = np.random.random((10, 2)) # origin point
target = [2, 2]


def test_loss(b, x):
    """PSO test function.

    Parameters
    ----------
    b : numpy.ndarray
        sets of inputs shape :code:'(n_particles, 2)'

    x : numpy.ndarray
        target point of shape :code:'(2, )

    Returns
    ----------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """

    dist = ((b - x) ** 2).sum(axis=1)

    return dist

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=len(origin), dimensions=2, options=options, init_pos=origin)
# Perform optimization
best_cost, best_pos = optimizer.optimize(test_loss, iters=35, x=target)

V = np.array(optimizer.velocity_history)
X = np.array(optimizer.pos_history)

m = 100; n = 100
x, y = np.meshgrid(np.linspace(-0.5,2.5,m), np.linspace(-0.5,2.5,n))
data = np.array((x, y)).reshape(2, -1).T
z = test_loss(data, x=[2,2]).reshape((n, m))

plt.figure(figsize=(25, 8))

plt.subplot(131)

plt.contourf(x, y, z, 15, cmap='Pastel1',alpha=0.7)
plt.colorbar()
plt.quiver(*X[0, :].T, V[1,:,0], V[1,:,1], color='blue', scale=8)
plt.scatter(X[0, :, 0], X[0, :, 1])
plt.scatter(x=2, y=2, s=150, c='red')
plt.ylim(-0.5, 2.5)
plt.xlim(-0.5, 2.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Initial state of PSO")

plt.subplot(132)

plt.contourf(x, y, z, 15, cmap='Pastel1',alpha=0.7)
plt.colorbar()
plt.quiver(*X[1, :].T, V[2,:,0], V[2,:,1], color='blue', scale=10)
plt.scatter(X[1, :, 0], X[1, :, 1])
plt.scatter(x=2, y=2, s=150, c='red')
plt.ylim(-0.5, 2.5)
plt.xlim(-0.5, 2.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("PSO after 1st iteration")

plt.subplot(133)
plt.contourf(x, y, z, 15, cmap='Pastel1',alpha=0.7)
plt.colorbar()
plt.quiver(*X[30, :].T, V[31,:,0], V[31,:,1], color='blue', scale=10)
plt.scatter(X[30, :, 0], X[30, :, 1])
plt.scatter(x=2, y=2, s=150, c='red')
plt.ylim(-0.5, 2.5)
plt.xlim(-0.5, 2.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ultimate state of PSO")

plt.show()