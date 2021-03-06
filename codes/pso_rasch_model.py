import pyswarms as ps
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
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
        initial_value = particles[i]  # (I+1)

        # set-up parameters to be updated (parameters is an (I+1) vector)
        betas = initial_value[:I]
        sigma = np.abs(initial_value[I])

        # Gaussian-Hermite nodes and weights
        nodes, weights = np.polynomial.hermite.hermgauss(nodes_num)
        weights /= np.sqrt(np.pi)
        nodes *= np.sqrt(2)

        ### Calculate marginal log-likelihood

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
init_value = np.vstack((beta_init, sigma_init))

bounds = [tuple(np.hstack((np.repeat(-np.inf, (I+1))))),
          tuple(np.hstack((np.repeat(np.inf, (I+1)))))]

"""PSO algorithm"""
# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
n = 50
init_pos = np.repeat(init_value.reshape((-1,1)), n, axis=1).T
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=n, dimensions=(I+1), options=options, bounds=bounds, init_pos=init_pos)
# Perform optimization
best_cost, best_pos = optimizer.optimize(Rasch_MMLE, iters=100, parameters=(Y, 21))

"""Bock-Aitkin algorithm"""
beta_init = 0.5 * np.ones((I, 1)); sigma_init = 1.5
init_value = np.vstack((beta_init, sigma_init)); nodes_num = 21
betas_t = beta_init; sigma_t = sigma_init

iter_time = 10

for i in range(iter_time):
    output = fsolve(EM_Rasch, x0=init_value, args=[Y, nodes_num, betas_t, sigma_t])
    betas_t = output[:-1]
    sigma_t = output[-1]
    # print(betas_t)
    # print(sigma_t)

bock_aitkin = Rasch_MMLE(output.reshape((1, -1)), (Y, 21))

"""Visualization"""
plt.figure(figsize=(25,8))
plt.subplot(121)
plt.plot(range(I+1), best_pos[:], c='dodgerblue', linewidth=2, label="PSO")
plt.scatter(range(I+1), best_pos[:], c="dodgerblue", s=45)
plt.scatter(range(I+1), output, c="r", linewidth=1, label="Bock-Aitkin")
plt.plot(range(I+1), output, c="r")
plt.legend()
plt.xlabel("Parameter (beta_1, ..., beta_J, sigma)")
plt.ylabel("Estimation")
plt.title("Parameter estimation by PSO with 100 iterations and Bock-Aitkin")

plt.subplot(122)
plt.plot(optimizer.cost_history, label="PSO", color="dodgerblue")
plt.scatter(range(len(optimizer.cost_history)), optimizer.cost_history, cmap='summer')
plt.axhline(y=bock_aitkin, color='r', linestyle='-', label="Bock-Aitkin")
plt.xlabel("Iteration")
plt.ylabel("Negative log likelihood")
plt.legend()
plt.title("Minimum negative log likelihood found by PSO with 100 iterations and Bock-Aitkin")

plt.show()