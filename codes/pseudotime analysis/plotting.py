import pyswarms as ps
import pandas as pd
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
np.random.seed(1234)
C400 = np.random.uniform(0, 1, size=400)
C100 = np.random.uniform(0, 1, size=100)
parameters = [(C400, 6,7,0.4,25),
             (C400, 4,-8,0.85,80),
             (C400, 1.4, 1.6, 1, 2),
             (C100, 6, 7, 0.4, 25),
             (C100, 4, -8, 0.85, 80),
             (C100, 1.4, 1.6, 1, 2)]

g_para = [(5.4294 , 0.4655 , 6.3844 , 17 ),
          (-9.7579 , 0.8610,3.9048, 66),
          (1.6199 , 0.9789 , 1.5215 , 2),
          ( 6.4729 , 0.3814 , 6.0773 , 15),
          (-3.7905 ,0.7915, 4.6435, 76),
          (2.7108, 0.2148, 0.8032, 2)]
l_para = [(5.1776, 0.4833, 6.5585, 16),
          (-8.3863, 0.8398, 3.9933, 60),
          (1.6983, 0.9364, 1.4810, 2),
          (6.5545, 0.3781, 6.0621, 15),
          (-3.6032, 0.7818, 4.7309, 77),
          (3.0299, 0.2688, 0.8193, 3)]

for j in range(6):
    t, mu0, k, t0, phi = parameters[j]
    t = np.sort(t)
    mut = link(t, k, t0, mu0)
    p = mut / (mut + phi)
    np.random.seed(1234)
    y = nbinom.rvs(phi, 1-p)

    n = 10
    b = np.random.random((n, 4))
    b[:, 2] += 3
    x0 = b[:, 0]
    #obj_fct(b, y=y, t=t)

    gbest = g_para[j]
    lbest = l_para[j]
    plt.figure(figsize=(8, 6))

    def plot_result(para, color, label, shift=0):
        k_fit, t0_fit, mu0_fit, phi_fit = para
        mut_fit = link(np.sort(t), k_fit, t0_fit, mu0_fit) + shift
        plt.plot(np.sort(t), mut_fit, c=color, label=label)

    plt.scatter(t, y, s=10)
    plt.plot(t, mut, c='orange', label="True mean")

    plot_result(gbest, 'red', label="gbest", shift=0.05)
    plot_result(lbest, 'blue', label="lbest")

    plt.title("Comparison of gbest and lbest.\n" +
              "C=%d, k_g=%d, t_g=%.1f, mu_g=%.1f, phi_g=%d" % (len(t), k, t0, mu0, phi))
    plt.legend()
    plt.ylim(np.min(y) - 1, np.max(y) + 1)
    plt.xlabel("t_c")
    plt.ylabel("y_c")

    plt.savefig("pso_trajectory_" + str(j) +".png")