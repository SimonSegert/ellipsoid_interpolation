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
    logcoefs=[logu] #keep track of all coeffidients, because loss can sometimes increase
    for _ in range(n_iters):
        logu = mirror_update(logu, X, x, eta=eta, G=G, v=v)
        logcoefs.append(logu)
        if eps:
            xp = np.linalg.norm(np.dot(X.T, np.exp(logu)) - x)
            if xp < eps:
                break
    logcoefs=np.array(logcoefs)
    coefs=np.exp(logcoefs) #n iters x n points
    projs=np.dot(coefs,X)
    dists=np.linalg.norm(projs-x,axis=1)
    min_id=dists.argmin()

    return projs[min_id],dists[min_id],logcoefs[min_id],dists


def mirror_update_batched(logu, X, x, eta=.01, G=None, v=None):
    assert len(logu) == len(x)
    # performs mirror gradient descent on the objective
    # |x-\sum_i u_ix_i|^2, where u is on unit simplex

    # G=precomputed gram matrix (x_i,x_j)
    # if not provided, then computed on the fly

    # v=vector of dot products (x_j,x)
    # if not provided computed on the fly

    # log u: shape (batch size) x (n training examples)
    # x: shape (batch size) x (n dims)

    #eta: either a scalar, or vector of length (batch size)
    u = np.exp(logu)

    A = np.einsum('ai,ij,jk->ak', u, X, X.T, optimize='optimal') if G is None else np.dot(u, G)
    B = np.dot(x, X.T) if v is None else v

    grad = A - B
    try:
        log_unew=logu-eta[:,None]*grad
    except:
        log_unew = logu - eta * grad
    log_unew = log_unew - logsumexp(log_unew, axis=1)[:, None]
    return log_unew


def closest_pt_hull_batched(X, x, eta=.01, n_iters=500,adaptive=False,adapt_factor=.3):
    # eps: used for early termination
    # stops if |\sum_i u_ix_j-x|<epsilon
    #if adaptive, will change learning rate on each step for each input, according to whether loss increased or decreased
    N, d = X.shape
    u = np.ones((len(x), N)) / N
    logu = np.log(u)
    v = np.dot(x, X.T)
    if adaptive:
        losses=np.linalg.norm(X.mean(0)-x,axis=1)

    etas=np.ones(len(x))*eta if adaptive else eta
    for ii in range(n_iters):
        logu = mirror_update_batched(logu, X, x, eta=etas, G=None, v=v)
        print(f'iter ={ii}')
        if adaptive:
            #increase learning rate if loss went down, and decrease if went up
            new_losses=np.linalg.norm(np.dot(np.exp(logu),X)-x,axis=1)
            dl=losses-new_losses
            etas[dl>0]*=1+adapt_factor
            etas[dl<0]*=adapt_factor
            losses=new_losses

    u=np.exp(logu)#batch size x n points
    projs=np.dot(u,X)
    dists=np.linalg.norm(projs-x,axis=1)
    return projs,dists