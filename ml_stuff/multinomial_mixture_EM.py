import numpy as np
import scipy
import pdb

def multinomial_mixture_EM(num_iters, tol, K, X):
    """
    K is number of clusters, X is N by M where N is number samples, M is number of categories
    returns list of K probability vectors, pi over the clusters, likelihood of data given those parameters
    """
    from numpy import linalg
    from scipy.misc import logsumexp
    N,T = X.shape
    thetas_old = np.random.rand(T,K)
    thetas_old = thetas_old / thetas_old.sum(axis=0)[np.newaxis,:]
    pi = np.ones(K) / K
    pi_old = pi
    
    for i in xrange(num_iters):
        log_p_X_z = X.dot(np.log(thetas_old)) + np.log(pi_old).T[np.newaxis,:]
        log_p_z_given_X = log_p_X_z - logsumexp(log_p_X_z, axis=1)[:,np.newaxis]
        p_z_given_X = np.exp(log_p_z_given_X)
        #if i % 1 == 0: print 'likelihood:', logsumexp(log_p_X_z + log_p_z_given_X)
        if i % 1 == 0: print 'likelihood:', (log_p_X_z * p_z_given_X).sum()
        weighted_counts = X.T.dot(p_z_given_X)
        thetas_new = weighted_counts / weighted_counts.sum(axis=0)
        pi_new = p_z_given_X.sum(axis=0)
        pi_new = pi_new / pi_new.sum()
        if linalg.norm(thetas_new - thetas_old, ord='fro') < tol:
            break
        thetas_old = thetas_new
        pi_old = pi_new

    order = sorted(range(thetas_new.shape[1]), key=lambda x:thetas_new[0,x])
    return thetas_new[:,order].T, pi_new[order], (log_p_X_z * p_z_given_X).sum()


def assign_clusters(thetas, pi, counts):
    log_thetas = np.log(thetas)
    log_pi = np.log(pi)
    return np.array([np.argmax([log_theta.dot(count)+log_pi_i for (log_theta, log_pi_i) in zip(log_thetas,log_pi)]) for count in counts])

