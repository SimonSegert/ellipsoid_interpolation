import numpy as np
from scipy.special import logsumexp

def mirror_update(logu, X, x, eta=.01, G=None, v=None):
    # performs mirror gradient descent on the objective
    # |x-\sum_i u_ix_i|^2, where u is on unit simplex

    # G=precomputed gram matrix (x_i,x_j)
    # if not provided, then computed on the fly

    # v=vector of dot products (x_j,x)
    # if not provided computed on the flyt
    u = np.exp(logu)

    A = np.einsum('ij,kj,k->i', X, X, u, optimize='optimal') if G is None else np.dot(G, u)
    B = np.dot(X, x) if v is None else v

    grad = A - B

    log_unew = logu - eta * grad
    log_unew = log_unew - logsumexp(log_unew)
    return log_unew


def closest_pt_hull(X, x, eta=.01, n_iters=500, eps=None):
    # eps: used for early termination
    # stops if |\sum_i u_ix_j-x|<epsilon
    N, d = X.shape
    # if N>2000 or so, it is faster to use the einsum
    # otherwise, it is faster to precompute gram matrix and use np.dot
    if N < 2000:
        G = np.dot(X, X.T)
    else:
        G = None
    u = np.ones(N) / N
    logu = np.log(u)
    v = np.dot(X, x)
    for _ in range(n_iters):
        logu = mirror_update(logu, X, x, eta=eta, G=G, v=v)
        xp = np.linalg.norm(np.dot(X.T, np.exp(logu)) - x)
        if eps and xp < eps:
            break
    return np.dot(X.T, np.exp(logu))