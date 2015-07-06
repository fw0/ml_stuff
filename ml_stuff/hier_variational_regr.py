import numpy as np
import itertools
import pdb
import scipy.special
import python_utils.python_utils.caching as caching
import copy

verbose = 0

class ModelSucksError(Exception):
    pass

def debug_print(level, s):
    global verbose
    if level <= verbose:
        print s

        
def simulate_data(L, N, d, x_var, mu, prec_val, lambda_val):

#    z_ns_num = np.random.randint(0, L, N)
    z_ns_num = np.tile(np.arange(0, L, 1), 1 + ((N-1)/L))[0:N]
    z_ns = np.zeros((N,L))
    z_ns[np.arange(0,N,1), z_ns_num] = 1
    x_ns = np.random.multivariate_normal(np.zeros(d), np.eye(d) * x_var, N)
    B_ls = np.random.multivariate_normal(mu, np.eye(d) / prec_val, L)
    lambda_ls = np.tile(lambda_val, L)
    B_ns = B_ls[z_ns_num,:]
    lambda_ns = lambda_ls[z_ns_num]
    y_ns = np.sum(x_ns * B_ns, axis=1) + np.random.normal(0, (1.0 / lambda_ns)**0.5)

    return x_ns, y_ns, z_ns, B_ls


def init_variational_params(x_ns, y_ns, z_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0):
    """
    
    """
    lambda_B = 1.0
    T_sigma = 1.0
    v = 10.0
    c = 10.0
    
    L = z_ns.shape[1]
    d = x_ns.shape[1]
    T = np.eye(d) * T_sigma
    m = np.ones(d)
    #mu_B_ls = np.random.multivariate_normal(np.ones(d)*5, np.eye(d) / .85, L)
    mu_B_ls = np.zeros((L,d))
    prec_B_ls = [np.eye(d) * lambda_B for l in xrange(L)]
#    alpha_lambda_ls = np.ones(L)*1000
#    beta_lambda_ls = np.ones(L)*5000
    alpha_lambda_ls = np.ones(L)#*76
    beta_lambda_ls = np.ones(L)#*6991.21705681
    return v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls


def infer_variational_params(num_iter, x_ns, y_ns, z_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls):
    """
    accepts hyperparameters, observed data, initial value of parameters
    """
    L = z_ns.shape[1]
    d = x_ns.shape[1]

    N = z_ns.shape[0]
    
    #print 'L: %d d: %d' % (L, d)
    
    k_ls = k_ls_f(L)
    eps = -0.00001

    old_evidence_val = None
    
    for i in xrange(num_iter):
        print i, np.sum(mu_B_ls)
        if True:
        
            v = v_0 + L
            c = c_0 + L
            if verbose > 2: print 'old m', m, mu_B_ls, T
            m = (L / (c_0 + L)) * (np.sum(mu_B_ls, axis=0) / L) + (c_0 / (c_0 + L)) * m_0

            #pdb.set_trace()
            ####T = np.linalg.inv(np.linalg.inv(T_0) + np.sum([E_B_l_B_l_T_f(mu_B_l, prec_B_l) for (mu_B_l, prec_B_l) in itertools.izip(mu_B_ls, prec_B_ls)], axis=0))
            third = np.sum([E_B_l_B_l_T_f(mu_B_l, prec_B_l) for (mu_B_l, prec_B_l) in itertools.izip(mu_B_ls, prec_B_ls)], axis=0)
            if verbose > 2: print 'here', np.linalg.inv(T_0), c_0*np.outer(m_0,m_0), third, c*np.outer(m,m)
            if verbose > 2: print 'there', c_0, m_0, c, m, third-c*np.outer(m,m), np.linalg.inv(prec_B_l), 'llllll'
            T = np.linalg.inv(np.linalg.inv(T_0) + c_0*np.outer(m_0,m_0) + np.sum([E_B_l_B_l_T_f(mu_B_l, prec_B_l) for (mu_B_l, prec_B_l) in itertools.izip(mu_B_ls, prec_B_ls)], axis=0) - c*np.outer(m,m))


            if verbose > 2: print 'new m', m, mu_B_ls, T

            #v = 0. + L
            #c = 0. + L
            #m = (L / (0. + L)) * (np.sum(mu_B_ls, axis=0) / L) + (0. / (c_0 + L)) * m_0
            #pdb.set_trace()
            ####T = np.linalg.inv(np.linalg.inv(T_0) + np.sum([E_B_l_B_l_T_f(mu_B_l, prec_B_l) for (mu_B_l, prec_B_l) in itertools.izip(mu_B_ls, prec_B_ls)], axis=0))
            #T = np.linalg.inv(np.linalg.inv(T_0)  + np.sum([E_B_l_B_l_T_f(mu_B_l, prec_B_l) for (mu_B_l, prec_B_l) in itertools.izip(mu_B_ls, prec_B_ls)], axis=0) - c*np.outer(m,m))

            
                
        if True:
            new_evidence_val = log_evidence_lower_bound(x_ns, y_ns, z_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls)
            print '1 step %d' % i, 'mu_B_ls sum: %.3f' % np.sum(mu_B_ls), 'log evidence bound: %.6f' % new_evidence_val
            if not old_evidence_val is None:
                assert new_evidence_val - old_evidence_val > eps
            old_evidence_val = new_evidence_val


        if True:
                        
            for l in xrange(L):
                prec_B_ls[l] = E_prec_f(v, T, c, m) + E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) * np.diag(z_ns[:,l]).dot(x_ns).T.dot(np.diag(z_ns[:,l]).dot(x_ns))
                temp = np.zeros((d,d))
                for n in xrange(N):
                    temp += z_ns[n,l] * np.outer(x_ns[n],x_ns[n])
                #print E_prec_f(v, T, c, m) + E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) * temp
                #print prec_B_ls[l]
                mu_B_ls[l] = np.linalg.inv(prec_B_ls[l]).dot(E_prec_mu_f(v, T, c, m) + E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) * x_ns.T.dot(z_ns[:,l] * y_ns))
                temp = np.zeros(d)
                for n in xrange(N):
                    temp += z_ns[n,l] * y_ns[n] * x_ns[n]
                #print np.linalg.inv(prec_B_ls[l]).dot(E_prec_mu_f(v, T, c, m) + (E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) * temp))
                #print mu_B_ls[l]

        if True:
            new_evidence_val = log_evidence_lower_bound(x_ns, y_ns, z_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls)
            if verbose > 2: print '2 step %d' % i, 'mu_B_ls sum: %.3f' % np.sum(mu_B_ls), 'log evidence bound: %.6f' % new_evidence_val
            if not old_evidence_val is None:
                assert new_evidence_val - old_evidence_val > eps
            old_evidence_val = new_evidence_val

        if True:
            if verbose > 2: print 'before alpha', alpha_lambda_ls
            if verbose > 2: print 'before beta', beta_lambda_ls
            if verbose > 2: print 'before lambda mean:', alpha_lambda_ls / beta_lambda_ls
            if verbose > 2: print 'mu_B_ls', mu_B_ls
            if verbose > 2: print 'prec_B_ls', prec_B_ls    
            for l in xrange(L):
                #alpha_lambda_ls[l] = 0.5 * np.sum(z_ns[:,l]) # FIX
                alpha_lambda_ls[l] = alpha_0 + 0.5 * np.sum(z_ns[:,l])
                E_B_l_B_l_T = E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l])
                #beta_lambda_ls[l] = z_ns[:,l].dot(0.5 * (y_ns**2 - 2 * y_ns * x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])) + np.array([np.sum(np.outer(x_n, x_n) * E_B_l_B_l_T) for x_n in x_ns]))) # FIX
                beta_lambda_ls[l] = beta_0 + z_ns[:,l].dot(0.5 * (y_ns**2 - 2 * y_ns * x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])) + np.array([np.sum(np.outer(x_n, x_n) * E_B_l_B_l_T) for x_n in x_ns])))

            if verbose > 2: print 'after alpha', alpha_lambda_ls
            if verbose > 2: print 'after beta', beta_lambda_ls
            if verbose > 2: print 'after lambda mean:', alpha_lambda_ls / beta_lambda_ls
                
        new_evidence_val = log_evidence_lower_bound(x_ns, y_ns, z_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls)
        if True:
            if verbose > 2: print '3 step %d' % i, 'mu_B_ls sum: %.3f' % np.sum(mu_B_ls), 'log evidence bound: %.6f' % new_evidence_val
            if not old_evidence_val is None:
                assert new_evidence_val - old_evidence_val > eps
            old_evidence_val = new_evidence_val
            
        if False:
            print 'v',v
            print 'T', T
            print 'c', c
            print 'm', m
            print 'mu_B_ls', mu_B_ls
            print 'prec_B_ls', prec_B_ls
            print 'alpha_lambda_ls', alpha_lambda_ls
            print 'beta_lambda_ls', beta_lambda_ls
            
    return v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls


def log_evidence_lower_bound(x_ns, y_ns, z_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls):

    L = z_ns.shape[1]
    d = x_ns.shape[1]
    
    val = 0
    debug_print(1, 'CALCULATING EVIDENCE')
    # p_data
    for l in xrange(L):
        if verbose > 2: print 'constant', - d * np.log(2*np.pi) / 2
        if verbose > 2: print 'E_log_lambda_l', E_log_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l])
        if verbose > 2: print 'E_lambda_l', E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l])
        if verbose > 2: print 'error', z_ns[:,l].dot(y_ns**2 \
                                - 2 * y_ns * x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])) \
                                + np.array([np.sum(np.outer(x_n, x_n) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l])) for x_n in x_ns]))
        if verbose > 2: print 'first:', 0.5*E_log_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) * np.sum(z_ns[:,l])
        if verbose > 2: print 'second:', -0.5 * E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) * z_ns[:,l].dot(y_ns**2 \
                                - 2 * y_ns * x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])) \
                                + np.array([np.sum(np.outer(x_n, x_n) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l])) for x_n in x_ns]))
        if verbose > 2: print 'first+second:', 0.5*E_log_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) * np.sum(z_ns[:,l]) -0.5 * E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) * z_ns[:,l].dot(y_ns**2 \
                                - 2 * y_ns * x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])) \
                                + np.array([np.sum(np.outer(x_n, x_n) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l])) for x_n in x_ns]))
        if verbose > 2: print 'mean squared error long:', 1.0 /np.mean(y_ns**2 \
                                - 2 * y_ns * x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])) \
                                + np.array([np.sum(np.outer(x_n, x_n) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l])) for x_n in x_ns]))
        if verbose > 2: print 'mean squared error:', 1.0 /np.mean((y_ns - x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])))**2)
        if verbose > 2: print 'all', z_ns[:,l].dot(\
                             - d * np.log(2*np.pi) / 2 \
                             + 0.5 * E_log_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) \
                             - 0.5 * E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) \
                             * (y_ns**2 \
                                - 2 * y_ns * x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])) \
                                + np.array([np.sum(np.outer(x_n, x_n) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l])) for x_n in x_ns])))
        new = z_ns[:,l].dot(\
                             - d * np.log(2*np.pi) / 2 \
                             + 0.5 * E_log_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) \
                             - 0.5 * E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) \
                             * (y_ns**2 \
                                - 2 * y_ns * x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])) \
                                + np.array([np.sum(np.outer(x_n, x_n) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l])) for x_n in x_ns])))
        val += new
        if verbose > 2: print 'val', val
        debug_print(2, 'update: %.2f p_data' % new)
        debug_print(1, 'evidence: %.2f p_data' % val)
        
    # p_mu_prec
    if verbose > 2: print 'm', m
    if verbose > 2: print 'T', T
    new = -1. * wishart_log_Z_f(v_0, T_0) \
      + ((v_0-d-1)/2.) * E_log_det_prec_f(v, T, c, m) \
      - 0.5 * np.sum(np.linalg.inv(T_0) * E_prec_f(v, T, c, m)) \
      - (d/2.)*np.log(2*np.pi) \
      + 0.5 * (E_log_det_prec_f(v, T, c, m) + np.log(c_0))\
      - (c_0/2.) * np.sum(np.outer(m_0, m_0) * E_prec_f(v, T, c, m)) \
      - (c_0/2.) * E_trace_prec_mu_mu_T_f(v, T, c, m) \
      + c_0 * E_prec_mu_f(v, T, c, m).T.dot(m_0)
    val += new
    temp = new
    debug_print(1, 'update: %.2f p_mu_prec' % new)
    debug_print(1, 'evidence: %.2f p_mu_prec' % val)

    # p_B_ls
    new = 0.
    new += (-L*d/2.) * np.log(2*np.pi) \
      + (L/2.) * E_log_det_prec_f(v, T, c, m) \
      - (L/2.) * E_trace_prec_mu_mu_T_f(v, T, c, m)
    temp2 = - (L/2.) * E_trace_prec_mu_mu_T_f(v, T, c, m)
    print v, T, c, m
    #pdb.set_trace()
    print 'temp2 update', -(L/2.), - (L/2.) * E_trace_prec_mu_mu_T_f(v, T, c, m)
    for l in xrange(L):
        new += -0.5 * np.sum(E_prec_f(v, T, c, m) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l]))
        new += E_B_l_f(mu_B_ls[l], prec_B_ls[l]).dot(E_prec_mu_f(v, T, c, m))
        temp2 += -0.5 * np.sum(E_prec_f(v, T, c, m) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l]))
        if verbose > 2: print E_prec_f(v, T, c, m), E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l])
        if verbose > 2: print 'temp2 update', -0.5 * np.sum(E_prec_f(v, T, c, m) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l]))
        if verbose > 2: print E_B_l_f(mu_B_ls[l], prec_B_ls[l]), E_prec_mu_f(v, T, c, m)
        temp2 += E_B_l_f(mu_B_ls[l], prec_B_ls[l]).dot(E_prec_mu_f(v, T, c, m))
        if verbose > 2: print 'temp2 update', E_B_l_f(mu_B_ls[l], prec_B_ls[l]).dot(E_prec_mu_f(v, T, c, m))
        if verbose > 2: print mu_B_ls[l], m, v*T
    if verbose > 2: print 'errorsquaredstuff', temp2
    val += new
    temp += new
    debug_print(2, 'update %.4f related_to_NW' % temp)
    debug_print(2, 'update: %.2f p_B_ls' % new)
    debug_print(1, 'evidence: %.2f p_B_ls' % val)
        
    # p_lambda_ls
    new = L * (alpha_0*np.log(beta_0) - scipy.special.gamma(alpha_0)) \
      + (alpha_0-1) * np.sum(E_log_lambda_ls_f(alpha_lambda_ls, beta_lambda_ls)) \
      - beta_0 * np.sum(E_lambda_ls_f(alpha_lambda_ls, beta_lambda_ls))
    val += new
    debug_print(1, 'update: %.2f p_lambda_ls' % new)
    debug_print(1, 'evidence: %.2f p_lambda_ls' % val)

    
    # H_mu_prec
    new = H_mu_prec_f(v, T, c, m)
    val += new
    debug_print(1, 'update: %.4f H_mu_prec' % new)
    debug_print(1, 'evidence: %.4f H_mu_prec' % val)

    # H_B_ls
    for l in xrange(L):
        new = H_B_l_f(mu_B_ls[l], prec_B_ls[l])
        val += new
        debug_print(1, 'update: %.2f H_B_ls' % new)
        debug_print(1, 'evidence: %.2f H_B_ls' % val)
        
    # H_lambda_ls
    for l in xrange(L):
        new = H_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l])
        val += new
        debug_print(1, 'update: %.5f H_lambda_ls' % new)
        debug_print(1, 'evidence: %.2f H_lambda_ls' % val)
        
    return val
        
#@caching.id_cache_fxn_decorator()
def k_ls_f(L):
    return np.array([[i <= j for j in xrange(L)] for i in xrange(L)], dtype=int)

#@caching.id_cache_fxn_decorator()
def k_ls_k_ls_T_f(L):
    return [np.outer(k_l, k_l) for k_l in k_ls_f(L)]
#    return k_ls_f(L).dot(k_ls_f(L).T)

#@caching.id_cache_fxn_decorator()
def E_prec_f(v, T, c, m):
    #print 'prec', v*T
    return v*T

#@caching.id_cache_fxn_decorator()
def E_prec_mu_f(v, T, c, m):
    ans = v*T.dot(m)
#    print 'E_prec_mu_f', ans
    return ans


#@caching.id_cache_fxn_decorator()
def E_log_det_prec_f(v, T, c, m):
    d = T.shape[0]
    ans = d * np.log(2) + np.log(np.linalg.det(T)) + np.sum(scipy.special.psi(np.arange(v+1-d,v,1)*0.5))
    #print 'log det prec', ans, 'prec', E_prec_f(v,T,c,m), 'v', v, 'T', T
    return ans

#@caching.id_cache_fxn_decorator()
def E_trace_prec_mu_mu_T_f(v, T, c, m):
    d = T.shape[0]
    if verbose > 2: print v, np.outer(m,m), T, 'trace prec mumu'
    if verbose > 2: print d/c, np.sum(np.outer(m,m)*T), v*np.sum(np.outer(m,m)*T), (d/c) + v*np.sum(np.outer(m,m)*T)
    return (d/c) + (v * np.sum(np.outer(m,m) * T))

#@caching.id_cache_fxn_decorator()
def H_mu_prec_f(v, T, c, m):
    d = T.shape[0]
    if verbose > 2: print v, T, c, m, d
    if verbose > 2: print 'log c', np.log(c)
    if verbose > 2: print wishart_log_Z_f(v, T)
    if verbose > 2: print  -((v-d-1)/2.) * E_log_det_prec_f(v, T, c, m)
    if verbose > 2: print  + (v*d / 2)
    if verbose > 2: print  + (d/2.) * (1 + np.log(2*np.pi)) 
    if verbose > 2: print  - 0.5 * (E_log_det_prec_f(v, T, c, m))
    if verbose > 2: print 'asdf', v-d-1
    return wishart_log_Z_f(v, T) \
      - ((v-d-1)/2.) * E_log_det_prec_f(v, T, c, m) \
      + (v*d / 2) \
      + (d/2.) * (1 + np.log(2*np.pi)) \
      - 0.5 * (E_log_det_prec_f(v, T, c, m))
#      - 0.5 * (np.log(c) + E_log_det_prec_f(v, T, c, m))

#@caching.id_cache_fxn_decorator()
def E_B_l_f(mu_B_l, prec_B_l):
    return copy.deepcopy(mu_B_l)

#@caching.id_cache_fxn_decorator()
def E_B_ls_f(mu_B_ls, prec_B_ls):
    return copy.deepcopy(mu_B_ls)
                    
#@caching.default_cache_fxn_decorator()
def E_B_l_B_l_T_f(mu_B_l, prec_B_l):
#    return np.outer(mu_B_l, mu_B_l)
    d = len(mu_B_l) #FIX
#    alt = 
    return np.outer(mu_B_l, mu_B_l) + np.linalg.inv(prec_B_l) #+ (np.tile(mu_B_l,(d,1)) - np.tile(mu_B_l,(d,1)).T)**2

#@caching.id_cache_fxn_decorator()
def H_B_l_f(mu_B_l, prec_B_l):
    d = len(mu_B_l)
    ans = (d/2.) * (1 + np.log(2*np.pi)) - 0.5 * np.log(np.linalg.det(prec_B_l))
#    print mu_B_l, prec_B_l, ans, 'H_B_l_f'
    return ans

#@caching.id_cache_fxn_decorator()
def E_lambda_l_f(alpha_lambda_l, beta_lambda_l):
    return alpha_lambda_l / beta_lambda_l

#@caching.id_cache_fxn_decorator()
def E_lambda_ls_f(alpha_lambda_ls, beta_lambda_ls):
    return alpha_lambda_ls / beta_lambda_ls

#@caching.id_cache_fxn_decorator()
def E_log_lambda_l_f(alpha_lambda_l, beta_lambda_l):
    # FIX
    #return np.log(alpha_lambda_l / beta_lambda_l)
    ans = -np.log(beta_lambda_l) + scipy.special.psi(alpha_lambda_l)
    #print 'log lambda:', ans
    return ans

#@caching.id_cache_fxn_decorator()
def E_log_lambda_ls_f(alpha_lambda_ls, beta_lambda_ls):
    return -np.log(beta_lambda_ls) + scipy.special.psi(alpha_lambda_ls)

#@caching.id_cache_fxn_decorator()
def H_lambda_l_f(alpha_lambda_l, beta_lambda_l):
    return alpha_lambda_l - np.log(beta_lambda_l) + scipy.special.gammaln(alpha_lambda_l) + (1.-alpha_lambda_l) * scipy.special.psi(alpha_lambda_l)

#@caching.id_cache_fxn_decorator()
def wishart_log_Z_f(v, T):
    d = T.shape[0]
    if verbose > 2: print 'log det T', T, np.log(np.linalg.det(T))
    return (v*d/2.) * np.log(2) + (v/2.) * np.log(np.linalg.det(T)) + scipy.special.multigammaln(v/2., d)

class trunc_norm(object):

    def __init__(self):
        from rpy2.robjects.packages import importr
        import rpy2.robjects as ro
        self.r = ro.r
        self.r('library(tmvtnorm)')

    def horse(self, mu, prec):
        import string
        import scipy.linalg
        cov = np.linalg.inv(prec)
        eps = .00001
        cov[cov<eps] = 0
        #sigma = scipy.linalg.sqrtm(cov)
        mu_str = string.join(map(str,mu),sep=',')
        d = prec.shape[0]
        sigma_str = string.join(map(str,cov.reshape(d**2)),sep=',')
        low_str = string.join(map(str,np.zeros(d)), sep=',')
        cmd = 'mtmvnorm(mean=c(%s), sigma=matrix(c(%s),ncol=%d,nrow=%d), lower=c(%s))' % (mu_str,sigma_str,d,d,low_str)
        #cmd = 'mtmvnorm(mean=c(%s), sigma=matrix(c(%s),ncol=%d,nrow=%d))' % (mu_str,sigma_str,d,d)
        #print 'cmd', cmd
        import rpy2
        try:
            firstm, secondm = self.r(cmd)
        except rpy2.rinterface.RRuntimeError, asdf:
            raise ModelSucksError
            print mu, np.diagonal(prec)
            #pdb.set_trace()
        firstm_np, secondm_np = np.array(firstm), np.array(secondm)
        if np.sum(np.isnan(firstm_np)) > 0:
            print 'naned'
            raise ModelSucksError
            pdb.set_trace()
            

        return firstm_np, secondm_np

    def mean(self, mu, prec):
        first, second = self.horse(mu, prec)
        return first

    def cov(self, mu, prec):
        first, second = self.horse(mu, prec)
        return second

    def E_xx_T(self, mu, prec):
        first, second = self.horse(mu, prec)
        #print 'first', first, 'second', second
        return np.outer(first, first) + second

    
trunc_norm_instance = trunc_norm()

#@caching.id_cache_fxn_decorator()
def E_delta_f(mu_delta, prec_delta):
    #print 'mu_delta', mu_delta
    ans = trunc_norm_instance.mean(mu_delta, prec_delta)
    #print 'E_delta', ans
#    print ans
    return ans

#@caching.id_cache_fxn_decorator()
def E_delta_delta_T_f(mu_delta, prec_delta):
    ans = trunc_norm_instance.E_xx_T(mu_delta, prec_delta)
#    print 'ddt', mu_delta, ans
    return ans
