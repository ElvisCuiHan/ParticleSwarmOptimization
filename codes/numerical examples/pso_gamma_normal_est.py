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

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
n = 50
# Call instance of PSO
bounds = [(0, 0, 0, -np.inf, 0), (1, np.inf, np.inf, np.inf, np.inf)]
optimizer = ps.single.GlobalBestPSO(n_particles=n, dimensions=5, options=options, bounds=bounds, init_pos=0.5*np.ones((n, 5)))
# Perform optimization
best_cost, best_pos = optimizer.optimize(log_lik_gamma_normal, iters=100, x=x)

print("Estimation is:")
print(best_pos.round(3))

points = np.linspace(-1, np.max(x), 300)
densities = gamma_normal_pdf(points, best_pos[0], best_pos[1], best_pos[2], best_pos[3], best_pos[4])
oracle = gamma_normal_pdf(points, 0.5, 2, 2, 3, 1)
plt.plot(points, densities,'--' ,linewidth=4, label="Estimated pdf by PSO", c="seagreen")
plt.plot(points, oracle, '-', linewidth=3, label="True pdf", c="dodgerblue")
plt.hist(x, weights=2.5*np.ones(len(x)) / len(x), bins=25, color="lightseagreen", label="Raw data")
plt.legend()
plt.xlabel("x")
plt.ylabel("density / frequency")
plt.title("Estimated probability density function of Gamma-Normal mixture.")
plt.show()