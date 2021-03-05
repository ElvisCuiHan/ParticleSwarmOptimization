import cvxpy
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.search import RandomSearch
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
import cvxpy as cp

def nnm(b, **kwargs):
    """Nuclear norm minimization.

    Parameters
    ----------
    b : numpy.ndarray
        sets of inputs shape :code:'(n_particles, dimensions)'

    Y : numpy.ndarray
        a completed matrix of shape :code:'(m_genes, n_cells)'

    omega_c : numpy.ndarray
        a list of missing indices of shape :code:'(dimensions, 2)'

    Returns
    ----------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """

    Y, omega_c = kwargs.values()
    m, n = Y.shape  # m genes and n cells
    K, p = b.shape  # K particles and p dimensions
    X = np.repeat(Y[:, :, np.newaxis], K, axis=2)

    for l in range(K):
        X[:, :, l][omega_c[:, 0], omega_c[:, 1]] = b[l, :]

    return np.linalg.norm(X, ord='nuc', axis=(0, 1))


m = 10
n = 10
K = 10

T = 50
result = []
for t in range(T):

    Y = np.random.random((m, n))
    U, L, V = np.linalg.svd(Y, full_matrices=False)
    k = 2
    Y_low = U[:, :k].dot(np.diag(L[:k])).dot(V[:k, :])
    ind_matrix = np.random.binomial(1, 0.5, (m, n))
    omega_c = np.argwhere(ind_matrix == 0)
    omega = np.argwhere(ind_matrix == 1)

    p = len(omega_c)

    b = np.random.random((K, p))

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=p, options=options)
    # Perform optimization
    best_cost, best_pos = optimizer.optimize(nnm, iters=100, Y=Y_low, omega_c=omega_c)

    known_value_indices = tuple(zip(*omega.tolist()))
    known_values = Y_low[omega[:, 0], omega[:, 1]]
    X = cp.Variable((m, n), pos=True)
    objective_fn = cp.normNuc(X)
    constraints = [
        X[known_value_indices] == known_values,
    ]
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    problem.solve(gp=False)
    #print("Solver reconstruct loss: ", np.linalg.norm((np.array(X.value) - Y_low), "fro"))
    Y[omega_c[:, 0], omega_c[:, 1]] = best_pos
    print("PSO reconstruct loss:", np.linalg.norm((Y - Y_low), "fro"))

    result.append([t, np.linalg.norm((Y - Y_low), "fro")/ np.linalg.norm(Y_low, "fro")])
    #result.append([t, "CVXPY", np.linalg.norm((np.array(X.value) - Y_low), "fro")])

df = pd.DataFrame(result, columns=["iter", "MSE"]).iloc[:, 1]
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(df, bins=12)
plt.title("Reconstruction loss of PSO, iteration=50.")
plt.ylabel("Frequency")
plt.xlabel("Reconstruction Loss")
plt.show()
