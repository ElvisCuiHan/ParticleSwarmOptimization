import pyswarms as ps
from scipy.optimize import minimize
from pyswarms.utils.search import RandomSearch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.stats import nbinom


def link(t, k, t0, mu0):
    return 2 * mu0 / (1 + np.exp(-k * (t - t0)))


def likelihood(y, t, k, t0, mu0, phi):
    mut = link(t, k, t0, mu0)
    phi = np.maximum(np.floor(phi), 1)
    p = mut / (mut + phi)
    return nbinom.pmf(y, phi, 1 - p)


def pso_obj_fct(b, **kwargs):
    N, d = b.shape
    y, t = kwargs.values()

    cost = np.zeros(N)
    for i in range(N):
        k, t0, mu0, phi = b[i, :]
        cost[i] = -np.sum(np.log(likelihood(y, t, k, t0, mu0, phi)))

    return cost

def scipy_obj_fct(b, para):
    y, t = para

    k, t0, mu0, phi = b
    cost = -np.sum(np.log(likelihood(y, t, k, t0, mu0, phi)))

    return cost

parameters = [(np.random.uniform(0, 1, size=400), 6,7,0.4,25),
             (np.random.uniform(0, 1, size=400), 4,-8,0.85,80),
             (np.random.uniform(0, 1, size=400), 1.4, 1.6, 1, 2),
             (np.random.uniform(0, 1, size=100), 6, 7, 0.4, 25),
             (np.random.uniform(0, 1, size=100), 4, -8, 0.85, 80),
             (np.random.uniform(0, 1, size=100), 1.4, 1.6, 1, 2)]
t, mu0, k, t0, phi = parameters[5]
t = np.sort(t)
mut = link(t, k, t0, mu0)
p = mut / (mut + phi)
np.random.seed(1234)
y = nbinom.rvs(phi, 1-p)

n = 100
b = np.random.random((n, 4))
b[:, 2] += 3
x0 = b[:, 0]
#obj_fct(b, y=y, t=t)

# Set-up hyperparameters
options = {'c1': 1.5, 'c2': 0.3, 'w':0.9}
bounds = [tuple([-np.inf, 0, np.min(y), 0]),
          tuple([np.inf, 1, np.max(y) / 2, 100])]
init_pos = b
# Call instance of PSO
gmodel = ps.single.GlobalBestPSO(n_particles=n, dimensions=b.shape[1], options=options, bounds=bounds, init_pos=b)
# Perform optimization
gcost, gbest = gmodel.optimize(pso_obj_fct, iters=150, y=y, t=t)

# Set-up hyperparameters
options = {'c1': 1.5,'c2': 0.3,'w': 0.9,'k': 5,'p': 1}

bounds = [tuple([-np.inf, 0, np.min(y), 0]),
          tuple([np.inf, 1, np.max(y) / 2, 100])]
init_pos = b
# Call instance of PSO
lmodel = ps.single.LocalBestPSO(n_particles=n, dimensions=b.shape[1],
                                   options=options, bounds=bounds, init_pos=b)
# Perform optimization
lcost, lbest = lmodel.optimize(pso_obj_fct, iters=150, y=y, t=t)

print("k_g, t_g, mu_g, phi:\n", [k, t0, mu0, phi])
print("gcost: ", np.round(gcost, 4))
print("gbest:\n", np.round(gbest, 4))
print("lcost: ", np.round(lcost, 4))
print("lbest:\n", np.round(lbest, 4))

plt.figure(figsize=(8, 6))


def plot_result(para, color, label):
    k_fit, t0_fit, mu0_fit, phi_fit = para
    mut_fit = link(np.sort(t), k_fit, t0_fit, mu0_fit)
    plt.plot(np.sort(t), mut_fit, c=color, label=label)


plt.scatter(t, y, s=10)
plt.plot(t, mut, c='orange', label="True mean")

plot_result(gbest, 'red', label="gbest")
plot_result(lbest + 0.01, 'blue', label="lbest")

plt.title("Comparison of gbest and lbest.\n" +
          "C=%d, k=%d, t0=%.1f, mu0=%.1f, phi=%d" % (len(t), k, t0, mu0, phi))
plt.legend()
plt.ylim(np.min(y) - 1, np.max(y) + 1)
plt.show()