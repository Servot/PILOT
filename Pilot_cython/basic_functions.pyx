cimport numpy as cnp
import numpy as np
from libc.stdint cimport int64_t

cnp.import_array()

def random_sample(a, k):
    a = np.random.choice(a, size=k, replace=False)
    return a

def loss_fun(str criteria, cnp.ndarray[cnp.int64_t, ndim=1] num, cnp.ndarray[cnp.float64_t, ndim=1] Rss, cnp.ndarray[cnp.int64_t, ndim = 1] k):
    """
    This function is used to compute the information criteria

    parameters:
    ----------
    criteria: str,
        the information criteria
    num: int,
        total number of samples
    Rss: float,
        the residual sum of squares, can be a vector
    k: ndarray,
        1D int array to describe the degrees of freedom, can be a vector

    return:
    -------
    float: The loss according to the information criteria
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] loss

    if criteria == "AIC":
        loss = num * np.log(Rss / num) + 2 * k
    elif criteria == "AICc":
        loss =  num * np.log(Rss / num) + 2 * k + (2 * k**2 + 2 * k) / (num - k - 1)
    elif criteria == "BIC":
        loss =  num * np.log(Rss / num) + np.log(num) * k

    print(loss)
    return loss

def compute_loss():
    cdef str criteria = 'BIC' # buffer type can only be defined locally!
    cdef cnp.ndarray[cnp.int64_t, ndim=1] num = np.array([100], dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] Rss = np.array([1.5,1.])
    cdef cnp.ndarray[cnp.int64_t, ndim=1] k = np.array([1, 2], dtype=np.int64)
    return(loss_fun(criteria, num, Rss, k))


