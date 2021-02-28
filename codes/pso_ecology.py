import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

def scad(b, **kwargs):
    """SCAD loss function.

    Parameters
    ----------
    b : numpy.ndarray
        sets of inputs shape :code:'(n_particles, dimensions)'

    X : numpy.ndarray
        design matrix of inputs shape:code:'(N_obs, dimensions)'

    y : numpy.ndarray
        response vector of inputs shape :code:'(N_obs)'

    lam : float
        penalty parameter lambda.

    a : float
        penalty parameter a.

    rho : float
        regularitzation parameter

    Returns
    ----------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """

    def scad_penalty(beta_hat, lambda_val, a_val):
        """
        Ref
        ----------
        https://andrewcharlesjones.github.io/posts/2020/03/scad/
        """

        is_linear = (np.abs(beta_hat) <= lambda_val)
        is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
        is_constant = (a_val * lambda_val) < np.abs(beta_hat)

        linear_part = lambda_val * np.abs(beta_hat) * is_linear
        quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat ** 2 - lambda_val ** 2) / (
                2 * (a_val - 1)) * is_quadratic
        constant_part = (lambda_val ** 2 * (a_val + 1)) / 2 * is_constant

        return linear_part + quadratic_part + constant_part

    X, y, lam, a, rho = kwargs.values()

    # e is a nxN matrix, n=n_particles and N=n_obs
    e = (y - X.dot(b.T)).T
    penalty = np.zeros(b.shape)

    for i in range(e.shape[0]):
        penalty[i, :] = scad_penalty(b[i, :], lam, a)

    return 0.5 * (1 / len(y)) * np.linalg.norm(e, axis=1) ** 2 + rho * penalty.sum(axis=1)

def LASSO(b, **kwargs):
    """LASSO loss

    Parameters
    ----------
    b : numpy.ndarray
        sets of inputs shape :code:'(n_particles, dimensions)'

    Returns
    ----------
    numpy.ndarray
        computed cost of size :code:'(n_particles, )'
    """
    X, y, lam = kwargs.values()

    e = (y - X.dot(b.T)).T

    return 0.5 * (1 / len(y)) * np.linalg.norm(e, axis=1) ** 2 + lam * np.abs(b).sum(axis=1)

data = pd.read_csv("../datasets/waterquality.csv")
scaler = StandardScaler()
data_scale = scaler.fit_transform(data.iloc[:, 1:])
X = data_scale[:, :-3]
y = data_scale[:, -2]

#best_cost, best_pos = optimizer.optimize(scad, iters=100, X=X, y=y.reshape((-1, 1)), lam=1., a=2.5, rho=0.1)

best_pos_history = []
rhos = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 10, 100]

for rho in rhos:
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=X.shape[1], options=options)
    best_cost, best_pos = optimizer.optimize(scad, iters=150, X=X, y=y.reshape((-1, 1)), lam=1., a=2.5, rho=rho)
    m = 100; n = 100
    xx, yy = np.meshgrid(np.linspace(-1, 1, m), np.linspace(-1, 1, n))
    rest = np.repeat(best_pos[2:].reshape((-1, 1)), m * n, 1).T
    xxyy = np.array((xx, yy)).reshape(2, -1).T
    xxyy_rest = np.hstack([xxyy, rest])
    points = np.array(optimizer.pos_history)[-1, :, :2]

    best_pos_history.append(best_pos)

    z = scad(xxyy_rest, X=X, y=y.reshape((-1, 1)), lam=1, a=2.5, rho=rho)
    z_ = z.reshape((n, m))
    #plt.contourf(xx, yy, z_, 15, cmap='Pastel2', alpha=0.5)
    #plt.colorbar()
    #plt.scatter(x=points[:, 0], y=points[:, 1], c="blue", s=15)
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)
    #plt.show()

best_pos_history = np.array(best_pos_history)
labels = data.columns[1:-3]
colors = ['red', 'tomato', 'orange', 'gold', 'yellow', 'yellowgreen',
         'lawngreen', 'springgreen', 'turquoise', 'deepskyblue',
         'steelblue', 'dodgerblue', 'blue', 'blueviolet',
         'fuchsia', 'pink', 'black']

#print(np.array(best_pos_history).shape)
plt.figure(figsize=(25, 12))
for t in range(best_pos_history.shape[1]):
    plt.plot(range(len(rhos)), best_pos_history[:, t], c=colors[t], label=labels[t])
plt.legend()
plt.xlabel("Ordering of rho")
plt.ylabel("Estimation")
plt.title("Solution path of SCAD using PSO.")
plt.show()