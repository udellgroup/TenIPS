import numpy as np
import tensorly as tl
from itertools import combinations
import apgpy
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
from collections import defaultdict
from tensorly import tucker_to_tensor
from tensorly.tenalg import kronecker
from copy import deepcopy
import multiprocessing as mp
import time

def std_logistic_function(x):
    return 1 / (1 + np.exp(-x))

def get_square_set(T):
    """
    A helper function that gets the dimensions in the square set, sizes of these dimensions, 
    and the difference of size multiplications between the square set and its complement.
    """
    sizes = np.array(T.shape)
    diff = np.prod(sizes)
    for n in range(len(sizes) // 2 + 1):
        subsets_n = list(combinations(np.arange(len(sizes)), n))
        for subset in subsets_n:
            diff_new = np.abs(np.prod(sizes[list(subset)]) - np.prod(sizes)/np.prod(sizes[list(subset)]))
            if diff_new < diff:
                diff = diff_new
                subset_selected = subset
    dims_sq = np.array(subset_selected)
    sizes_sq = sizes[list(dims_sq)]    
    
    return dims_sq, sizes_sq, diff

def square_unfolding(T):
    """
    Get the square unfolding of tensor T.
    """
    sizes = T.shape
    sizes_sq = get_square_set(T)[1]
    return np.reshape(tl.unfold(T, mode=0), 
                      (int(np.prod(sizes_sq)), int(np.prod(sizes)/np.prod(sizes_sq))), 
                      order='F')

def normalized_error(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def tenips_general(B_obs, P, r):
    X_bar = np.multiply(1./P, B_obs)
    Q_s = []
    Q_st = []
    for i in range(B_obs.ndim):
        X_n = tl.unfold(X_bar,i)
        q, _, _ = svds(X_n, k=r[i])
        q = np.fliplr(q)
        Q_s.append(q)
        Q_st.append(q.T)
    W = tucker_to_tensor((X_bar, Q_st))
    X_res = tucker_to_tensor((W, Q_s))
    return X_res

def tenips_general_paper1(B_obs, P, r):
    """
    The "paper1" refers to the HOSVD_w method in https://arxiv.org/pdf/2003.08537.pdf.
    """
    P_half = np.power(P,-0.5)
    X_bar = np.multiply(P_half,B_obs)
    Q_s = []
    Q_st = []
    for i in range(B_obs.ndim):
        X_n = tl.unfold(X_bar,i)
        q, _, _ = svds(X_n, k=r[i])
        q = np.fliplr(q)
        Q_s.append(q)
        Q_st.append(q.T)
    W = tucker_to_tensor((X_bar, Q_st))
    X_res = np.multiply(P_half,tucker_to_tensor((W,Q_s)))
    return X_res

def calc_fn(D, n):
    """
    A helper function to calculate the second-order estimator in https://arxiv.org/pdf/1711.04934.pdf.
    """
    D_n = tl.unfold(D,n)
    d,m = D_n.shape[0],D_n.shape[1]
    f = np.zeros((d,d))
    nonzero_is, nonzero_js = np.nonzero(D_n)
    for j in range(m):
        non_zero_rows = nonzero_is[nonzero_js == j]
        for i1 in range(len(non_zero_rows)):
            for i2 in range(i1+1,len(non_zero_rows)):
                val = D_n[non_zero_rows[i1],j]*D_n[non_zero_rows[i2],j]
                f[i1,i2] += val
                f[i2,i1] += val           
    return f
                
def tenips_general_paper2(B_obs, P, r):
    """
    The "paper2" refers to the SO-HOSVD method in https://arxiv.org/pdf/1711.04934.pdf. 
    """
    X_bar = np.multiply(1./P,B_obs)
    m = np.sum(B_obs!=0)
    scaling = 1. /((m-1)*m)
    Q_s = []
    Q_st = []
    for i in range(B_obs.ndim):
        f_n = scaling * calc_fn(X_bar,i)
        q, _, _ = svds(f_n, k=r[i])
        q = np.fliplr(q)
        Q_s.append(q)
        Q_st.append(q.T)
    W = tucker_to_tensor((X_bar, Q_st))
    X_res = tucker_to_tensor((W,Q_s))
    return X_res

def generate_orthogonal_mats(dim):
    """
    Generate an m-by-n column orthonormal random matrix.
    """
    m, n = dim[0], dim[1]
    H = np.random.uniform(-1, 1, (m, n))
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    mat = u
    return tl.tensor(mat)

def tensor_log_loss(M, A):
    """
    Compute the logistic loss with a mask tensor M and a parameter tensor A, with the same size.
    """
    assert M.shape == A.shape
    sigma_A = std_logistic_function(A)
    result = np.sum(- M * np.log(sigma_A) - (1 - M) * np.log(1 - sigma_A))
    return result


# Algorithm 1 (ConvexPE)
def one_bit_MC_fully_observed(M, link, tau, gamma, max_rank=None, init='zero',
                              apg_max_iter=500, apg_eps=1e-12, fixed_step_size=False,
                              apg_use_restart=True):
    """
    Algorithm 1 (ConvexPE): run one-bit matrix completion to estimate $\hat{A}_\square$.
    """
    m = M.shape[0]
    n = M.shape[1]
    tau_sqrt_mn = tau * np.sqrt(m*n)

    def prox(_A, t):
        _A = _A.reshape(m, n)

        # project so nuclear norm is at most tau*sqrt(m*n)
        if max_rank is None:
            U, S, VT = np.linalg.svd(_A, full_matrices=False)
#             U, S, VT = randomized_svd(A, n_components=min(m, n), n_iter=10, random_state=None)
        else:
            U, S, VT = randomized_svd(_A, max_rank)
#             U, S, VT = randomized_svd(A, n_components=max_rank, n_iter=50, random_state=None)
        nuclear_norm = np.sum(S)
        if nuclear_norm > tau_sqrt_mn:
            S *= tau_sqrt_mn / nuclear_norm
            _A = np.dot(U * S, VT)

        # clip matrix entries with absolute value greater than gamma
        mask = np.abs(_A) > gamma
        if mask.sum() > 0:
            _A[mask] = np.sign(_A[mask]) * gamma

        return _A.flatten()

    M_one_mask = (M == 1)
    M_zero_mask = (M == 0)
    def grad(_A):
        _A = _A.reshape(m, n)
        return (std_logistic_function(_A) - M).flatten()
    
    if init == 'zero':
        A_init = np.zeros(m*n)
    elif init == 'uniform':
        A_init = np.random.rand(m*n)
    
    A_hat = apgpy.solve(grad, prox, A_init,
                        max_iters=apg_max_iter,
                        eps=apg_eps,
                        use_gra=True,
                        use_restart=apg_use_restart,
                        fixed_step_size=fixed_step_size,
                        quiet=True)
    
    A_sq_pred = A_hat.reshape(m, n)
    return A_sq_pred


# Algorithm 2 (NonconvexPE)
def A_unfold_grad_U_single_block(U_all, G, target_ranks, n, N, j):
    """
    Helper function: compute a single term in the gradient of A^{(n)} with respect to U_n, given by a single column of U_{n+1}.
    """
    assert target_ranks == [U.shape[1] for U in U_all]
    G_unfold = tl.unfold(G, mode=n)
    r_minus_n_and_n_plus_one = int(np.prod(target_ranks) / (target_ranks[n%N] * target_ranks[(n+1) % N]))
    
    kron = kronecker((U_all[(n+1) % N][:, j].reshape(-1, 1), 
                                          kronecker([U_all[(n+i) % N] for i in range(2, N)])))    
    result = kron \
            @ G_unfold.T[(r_minus_n_and_n_plus_one * j):(r_minus_n_and_n_plus_one * (j+1)), :]    
    del kron   
    return result


def one_bit_TC_fully_observed_gd(M, link, target_ranks, max_iter=10, verbose=True, 
                                 A_true=None, step_size_U=5e-6, step_size_G=5e-6):
    """
    Arguments:
    M:                 the mask tensor.
    link:              the link function.
    target_ranks:      a list of target ranks.
    max_iter:          maximum number of iterations for gradient descent.
    verbose:           whether to print out intermediate details of optimization.
    A_true:            (optional) the true parameter tensor, only used to print out the relative loss.
                        Not required for the optimization itself.
    step_size_U:       step size for the gradient descent update on U.
    step_size_G:       step size for the gradient descent update on G.


    Return:
    A_pred:            the predicted parameter tensor.
    optimization details: a dictionary of optimization details for analysis purposes.
    """
    step_size_U = step_size_U
    step_size_G = step_size_G
    side_lengths = M.shape
    N = len(side_lengths)
    I_all = list(side_lengths)
    losses = []
    optimization_details = defaultdict(list)
    assert len(M.shape) == len(target_ranks)
    
    loss_true = tensor_log_loss(M, A_true)
    if verbose:
        print("true loss: {}".format(loss_true))
        
    start = time.time()
    
    # parameter initialization
    U_all = [2 * (np.random.rand(side_lengths[n], target_ranks[n]) - 0.5) for n in range(N)]
    G = 2 * (np.random.rand(*[target_ranks[n] for n in range(N)]) - 0.5)
    idx_all = []
    for n in range(N):
        for j in range(target_ranks[(n+1) % N]):
            idx_all.append((n, j))
    
    for iter in range(max_iter):
        print("Iteration {} ...".format(iter))
        A = tucker_to_tensor((G, U_all))
        grad_all = []
        
        if verbose: # optional: check the loss every 5 iterations
            if not iter % 5:
                loss = tensor_log_loss(M, A)
                relative_loss = loss / loss_true
                print("relative loss: {}".format(relative_loss))
                optimization_details['relative_losses'].append(deepcopy((iter, relative_loss)))
        
        grad_A = std_logistic_function(A) -  M # gradient of the logistic loss

        # compute the gradient of A^{(n)} with respect to U_n
        p1 = mp.Pool(25)
        result = [p1.apply_async(A_unfold_grad_U_single_block, args=[
                U_all, G, target_ranks, n, N, j]) for n, j in idx_all]
        p1.close()
        p1.join()

        A_unfold_grad_U_all = []
        for n in range(N):
            I_minus_n = int(np.prod(I_all) / I_all[n])
            A_unfold_grad_U = np.zeros((I_minus_n, target_ranks[n]))
            A_unfold_grad_U_all.append(A_unfold_grad_U)    

        for i, (n, j) in enumerate(idx_all):
            A_unfold_grad_U_all[n] += result[i].get()        
        
        def grad_U(mode=n):
            grad_A_unfold = tl.unfold(grad_A, mode=n)
            grad_U = grad_A_unfold @ A_unfold_grad_U_all[n]
            return grad_U

        def grad_G():
            G_unfold = tl.unfold(G, mode=N-1)
            grad_A_unfold = tl.unfold(grad_A, mode=N-1)
            kron_U_except_N = kronecker([U_all[i % N] for i in range(0, N-1)])
            grad_G_unfold = U_all[N-1].T @ grad_A_unfold @ kron_U_except_N
            grad_G = tl.fold(grad_G_unfold, mode=n, shape=G.shape)
            return grad_G
        
        
        grad_U_all = []
        for n in range(N):
            grad_U_all.append(grad_U(n))
        grad_G_fold = grad_G()
        
        if verbose:
            print("grad U: {}".format(np.average(grad_U_all)))
            print("U: {}".format(np.average(U_all)))        
            print("grad G: {}".format(np.average(grad_G_fold)))
            print("G: {}".format(np.average(G)))
        
    #         optimization_details['grad_U'].append(deepcopy(grad_U_all))
    #         optimization_details['U'].append(deepcopy(U_all))
    #         optimization_details['grad_G'].append(deepcopy(grad_G_fold))
    #         optimization_details['G'].append(deepcopy(G))
        
        for n in range(N):
            U_all[n] -= step_size_U * grad_U_all[n]
        G -= step_size_G * grad_G_fold

        optimization_details['grad_U'].append(deepcopy(grad_U_all))
        optimization_details['U'].append(deepcopy(U_all))
        optimization_details['grad_G'].append(deepcopy(grad_G_fold))
        optimization_details['G'].append(deepcopy(G))        

        elapsed = time.time() - start
        if verbose:
            print("cumulated time: {}".format(elapsed))
        optimization_details['cumulated_time'].append(deepcopy(elapsed))
    
    A_pred = tucker_to_tensor((G, U_all))
    return A_pred, optimization_details

def unfolding_based_ips_tensor_completion(B_obs, P, ranks, unfolding='square'):
    """
    On a tensor, use SVD on its unfolding to complete the tensor.
    
    Arguments:
    B_obs:             the observed tensor.
    P:                 the propensity tensor.
    ranks:             a list of target ranks.
    unfolding:         one of {'square', '0'}: whether to complete the tensor in its square unfolding, or rectangular unfolding along mode 0.

    Return:
    X_res:             the predicted tensor.
    """
    X_bar = np.multiply(1./P, B_obs)
    x_bar_mat = tl.unfold(X_bar, 0)
    rank = ranks[0]
    if unfolding == 'square':
        dims_sq = get_square_set(B_obs)[0]
        x_bar_mat = square_unfolding(X_bar)
        rank = min(np.prod(np.array(ranks)[dims_sq]), int(np.prod(np.array(ranks))/np.prod(np.array(ranks)[dims_sq])))
    elif unfolding == '0':
        assert 0 <= int(unfolding) < B_obs.ndim
        rank_idx = int(unfolding)
        x_bar_mat = tl.unfold(X_bar, rank_idx)
        rank = min(ranks[rank_idx], int(np.prod(np.array(ranks))/ranks[rank_idx]))
    
    print("rank is {}".format(rank))
    U, sigma, Vt = svds(x_bar_mat, k=rank)
    U = np.fliplr(U)
    sigma = np.flipud(sigma)
    Vt = np.flipud(Vt)
    Sigma = np.diag(sigma)
    X_res = U.dot(Sigma).dot(Vt)
    return X_res
