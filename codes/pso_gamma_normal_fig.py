import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
import pandas as pd
import scipy
from scipy.stats import norm, gamma

def gamma_normal_pdf(x, lam, alpha, beta, mu, sigma):
    """Probability density of Gamma-Normal mixture distribution.

    Parameters
    ----------
    x : vector or scalar
        Realization of random variable :code:'(N_obs, )'

    lam : vector or scalar
        Proportion of Gamma distribution.

    alpha : vector or scalar
        Location parameter of Gamma.

    beta : vector or scalar
        Scale parameter of Gamma.

    mu : vector or scalar
        Mean parameter of Normal.

    sigma : vector or scalar
        Variance parameter of Normal.

    Returns
    ----------
    numpy.ndarray
        computed density of size :code:`(N_obs, )`
    """

    g = gamma.pdf(x, a=alpha, scale=1 / beta)
    n = norm.pdf(x, loc=mu, scale=sigma)

    return lam * g + (1 - lam) * n

def log_lik_gamma_normal(b, x):
    """
    Parameters
    ----------
    b : numpy.ndarray
        sets of inputs shape :code:'(n_particles, 5)'

    x : numpy.ndarray
        the gene expression vector of inputs shape :code:'(N_obs, )'

    Returns
    ----------
    numpy.ndarray
        computed log likelihood of Gamma-Normal mixture of size :code:`(n_particles, )`
    """

    lams = b[:, 0];
    alphas = b[:, 1];
    betas = b[:, 2];
    mus = b[:, 3];
    sigmas = b[:, 4]
    n_particles = lams.shape[0]
    log_lik = np.zeros(n_particles)

    for i in range(n_particles):
        pdf = gamma_normal_pdf(x, lams[i], alphas[i], betas[i], mus[i], sigmas[i])
        log_lik[i] = np.log(pdf + 1e-100).sum()

    return -log_lik

np.random.seed(12)
b = np.random.random((10, 5))
n = norm.rvs(size=100, loc=3, scale=1)
g = gamma.rvs(size=100, a=2, scale=0.5)
x = np.hstack((n, g))

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=5, options=options)
best_cost, best_pos = optimizer.optimize(log_lik_gamma_normal, iters=100, x=x)
m = 60; n = 90
xx, yy = np.meshgrid(np.linspace(-3, 9, m), np.linspace(0.01, 7, n))
rest = np.repeat(np.array([0.5, 2, 2]).reshape((-1, 1)), m * n, 1).T
xxyy = np.array((xx, yy)).reshape(2, -1).T
xxyy_rest = np.hstack([rest, xxyy])
points = np.array(optimizer.pos_history)

plt.figure(figsize=(25, 8))

z_ = log_lik_gamma_normal(xxyy_rest, x=x).reshape((n, m))

# PLOT normal loss fct
plt.subplot(131)
plt.title("Negative log liklihood of Normal parameters")
plt.contourf(xx, yy, z_, 15, cmap='summer', alpha=0.5)
plt.colorbar()
plt.xlabel("mu")
plt.ylabel("sigma square")
plt.xlim(-3, 9)
plt.ylim(0.01, 7)

xx, yy = np.meshgrid(np.linspace(0.01, 10, m), np.linspace(0.01, 10, n))
rest = np.repeat(np.array([0.5, 3, 1]).reshape((-1, 1)), m * n, 1).T
xxyy = np.array((xx, yy)).reshape(2, -1).T
xxyy_rest = np.hstack([rest[:, 0].reshape((-1, 1)), xxyy, rest[:, 1:]])

z_ = log_lik_gamma_normal(xxyy_rest, x=x).reshape((n, m))

# PLOT gamma loss fct
plt.subplot(132)
plt.title("Negative log likelihood of Gamma parameters")
plt.contourf(xx, yy, z_, 15, cmap='summer', alpha=0.5)
plt.colorbar()
plt.xlabel("Alpha")
plt.ylabel("1 / Beta")
plt.xlim(0.01, 10)
plt.ylim(0.01, 10)

xx = np.linspace(0, 0.95, 100)
rest = np.repeat(best_pos[1:].reshape((-1, 1)), 100, 1).T
xxyy_rest = np.hstack([xx.reshape((-1, 1)), rest])

z_ = log_lik_gamma_normal(xxyy_rest, x=x)

# PLOT lambda loss fct
plt.subplot(133)
plt.title("Negative log liklihood of lambda parameter")
plt.plot(xx, z_, c="green")
plt.xlabel("Lambda")
plt.ylabel("Negative log likelihood")

plt.show()