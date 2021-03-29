import pyswarms as ps
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib
from matplotlib.ticker import LinearLocator
from pyswarms.utils.plotters.formatters import Mesher, Designer
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface

#matplotlib.use("TkAgg")
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

# Plot the sphere function's mesh for better plots
m = Mesher(func=fct_3,
           limits=[(-20,5), (-20,5)], levels=20, delta=0.2)
# Adjust figure limits
d = Designer(limits=[(-20,5), (-20,5), (-1.1,0.1)],
             label=['x-axis', 'y-axis', 'z-axis'])

options = {'c1':1.2, 'c2':0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=25, dimensions=2, options=options, init_pos=np.random.random((25, 2))-20)
optimizer.optimize(fct_3, iters=300)
# Plot the cost
#plot_cost_history(optimizer.cost_history)
#plt.show()
#plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d, mark=(0,1.39))
print("hi")
pos_history_3d = m.compute_history_3d(optimizer.pos_history) # preprocessing
animation3d = plot_surface(pos_history=pos_history_3d,mesher=m, designer=d,mark=(0,0,0))
plt.show()