import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.search import RandomSearch
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

#matplotlib.use("TkAgg")

# Set-up choices for the parameters
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
        regularization parameter lambda.

    a : float
        regularization parameter a.

    Returns
    ----------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    X, y, lam, a = kwargs.values()

    # e is a nxN matrix, n=n_particles and N=n_obs
    e = (y - X.dot(b.T)).T
    penalty = np.zeros(b.shape)

    for i in range(e.shape[0]):
        penalty[i, :] = scad_penalty(b[i, :], lam, a)

    return np.linalg.norm(e, axis=1) ** 2 + penalty.sum(axis=1)

def scad_modified(b, args):
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
        regularization parameter lambda.

    a : float
        regularization parameter a.

    Returns
    ----------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    X, y, lam, a = args

    # e is a Nx1 vector, N=N_obs
    e = (y - X.dot(b))
    penalty = np.zeros(b.shape)

    for i in range(b.shape[0]):
        penalty[i] = scad_penalty(b[i], lam, a)

    return np.linalg.norm(e) ** 2 / X.shape[0] + penalty.sum()

Methods = ["PSO", "Powell", "BFGS", "L-BFGS-B"]
T = 30
result = []

for t in range(T):
    dim=20
    n = 100
    X = np.random.normal(loc=1, scale=1, size=(n, dim))
    b = np.ones((1,dim))
    y = X.dot(b.T) + np.random.normal(loc=1, scale=1, size=(n, 1))

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=dim, options=options)
    # Perform optimization
    best_cost, best_pos = optimizer.optimize(scad, iters=100, X=X, y=y, lam=5, a=3.5)

    x0 = np.zeros((dim))

    print("PSO")
    print("Estimation:", best_pos.round(2))
    print("MSE:", ((best_pos - b) ** 2).mean().round(4))
    result.append([t, "PSO", ((best_pos - b) ** 2).mean()])

    for i in range(1, len(Methods)):
        print(Methods[i])
        res = minimize(scad_modified, x0, [X, y, 5, 3.5], method=Methods[i])
        print("Estimation:", res['x'].round(2))
        print("MSE:", ((res['x'] - b) ** 2).mean().round(4))
        result.append([t, Methods[i], ((res['x'] - b) ** 2).mean()])

df = pd.DataFrame(result, columns=["iter", "method", "MSE"])
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.boxplot(x="method", y="MSE", data=df)
plt.title("MSE of different optimization methods, iteration=30.")
plt.xlabel("Method")
plt.ylabel("MSE")
plt.show()