import numpy as np

# Conditional likelihood function
def f_y(Y, betas, sigma, nodes):
    """
    This function calculates the conditional likelihood of matrix Y,
    and the return is a NxM matrix. M is the number of nodes.
    """

    # set-up
    if len(betas.shape) == 1:
        betas = np.array([betas]).T
    X = 1 - Y  # X: NxI matrix
    N, I = np.shape(Y)
    M = len(nodes)

    # sufficient statistics
    r = Y.sum(axis=1)  # Nx1 vector
    q = Y.sum(axis=0)  # Ix1 vector

    ### equation (13) in paper
    # numerator: NxM matrix
    to_exp = (np.outer(I - r, -sigma * nodes) + X.dot(betas))
    numerator = np.exp(to_exp)

    # denomenator: Mx1 vector
    denomenator = (np.outer(np.exp(-sigma * nodes), np.exp(betas)) + 1).prod(axis=1)

    # conditional likelihood: NxM matrix
    CLF = numerator / denomenator
    return CLF

# H function in eq(12)
def H(nodes, fct="identity", betas=None, sigma=None):
    """ This function calculates the H function in eq(12)
    """
    if fct == "identity":
        # Return: Mx1 vector
        return sigma * nodes
    elif fct == "one":
        # Return: a scalar
        return 1
    elif fct == "logistic":
        # Return: an MxI matrix
        return 1 / (1 + np.outer(np.exp(-sigma * nodes), np.exp(betas)))
    elif fct == "logistic_identity":
        # Return: an MxI matrix
        return np.diag(nodes).dot(1 / (1 + np.outer(np.exp(-sigma * nodes), np.exp(betas))))
    elif fct == "hessian_sigma":
        # Return: an MxI matrix
        to_divide = (1 + np.outer(np.exp(-sigma * nodes), np.exp(betas))) * (
                    1 + np.outer(np.exp(sigma * nodes), np.exp(-betas)))
        return np.diag(nodes ** 2).dot(1 / to_divide)
    elif fct == "hessian_betaj":
        # Return: an MxI matrix
        to_divide = (1 + np.outer(np.exp(-sigma * nodes), np.exp(betas))) * (
                    1 + np.outer(np.exp(sigma * nodes), np.exp(-betas)))
        return 1 / to_divide
    elif fct == "hessian_betai_betaj":
        # Return: a scalar
        return 0
    elif fct == "hessian_sigma_betaj":
        # Return: an MxI matrix
        to_divide = (1 + np.outer(np.exp(-sigma * nodes), np.exp(betas))) * (
                    1 + np.outer(np.exp(sigma * nodes), np.exp(-betas)))
        return np.diag(nodes).dot(1 / to_divide)
    else:
        return None

def EAP_estimation(Y, betas, sigma, nodes_num=21):
    ### Generate Gaussian-Hermite quadrature
    nodes, weights = np.polynomial.hermite.hermgauss(nodes_num)
    weights /= np.sqrt(np.pi)
    nodes *= np.sqrt(2)

    # marginal densities: Nx1 vector
    marginal = (f_y(Y, betas, sigma, nodes) * H(nodes, fct="one")).dot(weights)

    # EAP: Nx1 vector
    EAP_est = (f_y(Y, betas, sigma, nodes).dot((H(nodes, fct="identity", sigma=sigma) * weights))) / marginal

    return EAP_est

def standard_error_est(Y, nodes_num, betas, sigma):
    """
    This function calculates standard error estimation of betas and sigma.
    The method is based on Fisher information matrix (or Hessian matrix of marginal log-likelihood).
    The output is a (I+1)x(I+1) matrix where I is the length of betas.
    """

    ## Generating nodes
    N, I = np.shape(Y)
    nodes, weights = np.polynomial.hermite.hermgauss(nodes_num)
    weights /= np.sqrt(np.pi)
    nodes *= np.sqrt(2)

    ## Calculating H functions in equation (12)
    h_betaj = H(nodes, fct="hessian_betaj", sigma=sigma, betas=betas)  # MxI
    h_betai_betaj = H(nodes, fct="hessian_betai_betaj", sigma=sigma, betas=betas)  # MxI
    h_sigma = H(nodes, fct="hessian_sigma", sigma=sigma, betas=betas)  # scalar
    h_sigma_betaj = H(nodes, fct="hessian_sigma_betaj", sigma=sigma, betas=betas)  # MxI

    ## Calculating Hessian matrix based on equation (14) (15) (16) and (17)
    marginal = (f_y(Y, betas, sigma, nodes) * H(nodes, fct="one")).dot(weights)

    # h_sigma_mat
    cache = f_y(Y, betas, sigma, nodes).dot(np.diag(weights).dot(h_sigma))
    h_sigma_scalar = np.diag(1 / marginal).dot(cache).sum()  # scalar
    # print(h_sigma_scalar)

    # h_betaj_mat
    cache = f_y(Y, betas, sigma, nodes).dot(np.diag(weights).dot(h_betaj))
    h_betaj_vec = np.diag(1 / marginal).dot(cache).sum(axis=0)  # Ix1 vector
    # print(h_betaj_vec)

    # h_sigma_betaj
    cache = f_y(Y, betas, sigma, nodes).dot(np.diag(weights).dot(h_sigma_betaj))
    h_sigma_betaj_vec = np.diag(1 / marginal).dot(cache).sum(axis=0)  # Ix1 vector
    # print(h_sigma_betaj_vec)

    hessian = np.zeros((I + 1, I + 1))
    hessian[:I, -1] = hessian[-1, :I] = h_sigma_betaj_vec
    hessian[I, I] = h_sigma_scalar
    for i in range(I):
        hessian[i, i] = h_betaj_vec[i]

    ## Calculating standard error of parameters
    std_est = np.sqrt([np.linalg.inv(hessian)[i, i] for i in range(I + 1)])

    return {"Hessian matrix": hessian, "std_est": std_est}

def EM_Rasch(initial_value, parameters):
    # unpack parameters
    Y, nodes_num, betas_t, sigma_t = parameters
    N, I = Y.shape

    # set-up parameters to be updated
    betas = initial_value[:I]
    sigma = initial_value[-1]

    # Gaussian-Hermite nodes and weights
    nodes, weights = np.polynomial.hermite.hermgauss(nodes_num)
    weights /= np.sqrt(np.pi)
    nodes *= np.sqrt(2)

    # sufficient statistics
    r = Y.sum(axis=1)  # Nx1 vector
    q = Y.sum(axis=0)  # Ix1 vector

    ### E-step

    # (a) marginal densities: Nx1 vector
    marginal = (f_y(Y, betas_t, sigma_t, nodes) * H(nodes, fct="one")).dot(weights)

    # (b) EAP: Nx1 vector
    EAP_est = (f_y(Y, betas_t, sigma_t, nodes).dot((H(nodes, fct="identity", sigma=sigma) * weights))) / marginal

    # (c) Logistic_identity in eq(10): Nx1 vector
    cache = f_y(Y, betas_t, sigma_t, nodes).dot((np.diag(weights).dot(
        H(nodes, fct="logistic_identity", sigma=sigma, betas=betas))))
    logistic_identity_mat = np.diag(1 / marginal).dot(cache)  # NxI matrix
    logistic_identity = logistic_identity_mat.sum(axis=1)

    # (d) Logistic in eq(11): NxI matrix
    logistic = np.diag(1 / marginal).dot(f_y(Y, betas_t, sigma_t, nodes)).dot(
        (np.diag(weights).dot(H(nodes, fct="logistic", sigma=sigma, betas=betas))))

    ### M-step

    eq_10 = q - logistic.sum(axis=0)
    eq_11 = np.mean(r * EAP_est - logistic_identity)
    return np.concatenate((eq_10, np.array([eq_11])))