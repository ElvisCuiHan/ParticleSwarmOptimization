import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from codes.API.EM_Rasch_API import *

def Rasch_MMLE(particles, parameters):
    """
    Parameters
    ----------
    particles : numpy.ndarray
        sets of inputs shape :code:'(n_particles, 5)'

    parameters : numpy.ndarray
        the response matrix Y of inpus shape :code:'(N_participants, J_items)'
        the number of quadrature nodes to approximate integral :code: integer

    Returns
    ----------
    numpy.ndarray
        computed log likelihood of Gamma-Normal mixture of size :code:`(n_particles, )`
    """
    # unpack parameters
    Y, nodes_num = parameters
    N, I = Y.shape

    log_lik = np.zeros((particles.shape[0]))

    for i in range(particles.shape[0]):
        initial_value = particles[i]  # (I+2)

        # set-up parameters to be updated (parameters is an (I+2) vector)
        betas = initial_value[:I]
        sigma = np.abs(initial_value[I])
        mu = initial_value[I + 1]

        # Gaussian-Hermite nodes and weights
        nodes, weights = np.polynomial.hermite.hermgauss(nodes_num)
        weights /= np.sqrt(np.pi)
        nodes *= np.sqrt(2)

        ### Calculate marginal log-likelihood

        # Initialize marginals: Nx1 vector
        marginal = np.zeros(N)

        # Adjust nodes
        nodes = nodes + mu

        # Calculate marginals: Nx1 vector
        marginal = (f_y(Y, betas, sigma, nodes) * H(nodes, fct="one")).dot(weights)

        # Calculate sum of log-likelihood
        log_lik[i] = np.sum(np.log(marginal))

    return -log_lik

data = np.array(pd.read_excel(r'../datasets/Verbal aggression/data verbal aggression matrix dichot.xls'))
Y = data[:, :24]
N, I = np.shape(Y)
G = 1
beta_init = 0.5 * np.ones((I, G)); sigma_init = 2 * np.ones((1, G))
pi_init = [1 / G] * G; mu_init = [0] * G
init_value = np.vstack((beta_init, sigma_init, mu_init))

bounds = [tuple(np.hstack((np.repeat(-np.inf, (I+2))))),
          tuple(np.hstack((np.repeat(np.inf, (I+2)))))]

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
n = 20
init_pos = np.repeat(init_value.reshape((-1,1)), n, axis=1).T
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=n, dimensions=(I+2), options=options, bounds=bounds, init_pos=init_pos)
# Perform optimization
best_cost, best_pos = optimizer.optimize(Rasch_MMLE, iters=100, parameters=(Y, 21))

plt.figure(figsize=(10,6))
plt.plot(range(I), best_pos[:I])
plt.scatter(range(I), best_pos[:I], c="orange", s=45)
plt.show()