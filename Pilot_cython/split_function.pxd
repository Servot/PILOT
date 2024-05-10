cimport numpy as cnp
import numpy as np
from libc.string cimport strcmp

cdef best_split(
    # cnp.ndarray[cnp.int64_t, ndim=1] 
    index, # NEW: regression_nodes deleted
    cnp.int64_t n_features,
    # cnp.ndarray[cnp.int64_t, ndim=2] 
    sorted_X_indices,
    # cnp.ndarray[cnp.float64_t, ndim=2] 
    X,
    # cnp.ndarray[cnp.float64_t, ndim=2] 
    y,
    # NEW: char* split_criterion deleted,
    cnp.int64_t min_sample_leaf,
    # cnp.ndarray[cnp.int64_t, ndim=1] 
    k_con,
    # cnp.ndarray[cnp.int64_t, ndim=1] 
    k_lin,
    # cnp.ndarray[cnp.int64_t, ndim=1] 
    k_split_nodes,
    # cnp.ndarray[cnp.int64_t, ndim=1] 
    k_pconc,
    # cnp.ndarray[cnp.int64_t, ndim=1] 
    categorical,
    cnp.int64_t max_features_considered,
    cnp.int64_t min_unique_values_regression,
)

cdef loss_fun(int num, 
                     # cnp.ndarray[cnp.float64_t, ndim=1] 
                     Rss, 
                     # cnp.ndarray[cnp.int64_t, ndim = 1] 
                     k)

"""
cdef double best_pivot 
cdef char* best_node
cdef double best_loss
cdef int best_feature
cdef cnp.ndarray[cnp.float64_t, ndim=1] lm_L
cdef cnp.ndarray[cnp.float64_t, ndim=1]lm_R
cdef cnp.ndarray[cnp.float64_t, ndim=1] interval

cdef char* regression_nodes[3] # NEW: TO BE TREATED AS AN INPUT

cdef int l
cdef cnp.ndarray[cnp.float64_t, ndim=2] coef 
cdef cnp.ndarray[cnp.float64_t, ndim=2] intercept 

# TODO: define the range of the samples
cdef int feature_id

# NEW: INITIALIZE INTERMEDIATE VARIABLES
cdef cnp.ndarray[cnp.int64_t, ndim=1] idx
cdef cnp.ndarray[cnp.float64_t, ndim=2] X_sorted
cdef cnp.ndarray[cnp.float64_t, ndim=2] y_sorted
cdef cnp.ndarray[cnp.float64_t, ndim=1] possible_p
cdef int lenp
cdef int pointer_c = 0 # NEW: BY DEFAULT CATEGORICAL = NP.ARRAY([X.SHAPE[1]]) 
cdef cnp.ndarray[cnp.int64_t, ndim=1] num
cdef cnp.ndarray[cnp.float64_t, ndim=2] Moments
cdef double intercept_con
cdef double coef_con # NEW: SHOULD BE FLOAT?
cdef double rss_1d
cdef cnp.ndarray[cnp.float64_t, ndim=1] rss
cdef cnp.ndarray[cnp.float64_t, ndim=1] loss
cdef double var
cdef double coef_lin
cdef double intercept_lin
cdef cnp.ndarray[cnp.float64_t, ndim=2] XtX
cdef cnp.ndarray[cnp.float64_t, ndim=2] XtY
cdef double pre_pivot
cdef int p
cdef int i
cdef double pivot
cdef double xi
cdef cnp.ndarray[cnp.float64_t, ndim=1] Moments_add 
cdef cnp.ndarray[cnp.float64_t, ndim=1] coefs
cdef int index_min
# cdef cnp.ndarray[cnp.int8_t, ndim = 1] index_add
cdef cnp.ndarray[cnp.float64_t, ndim=1] X_add 
cdef cnp.ndarray[cnp.float64_t, ndim=2] y_add 

# NEW: INTILIZE VARIABLES FOR THE CATEGORICAL ONLY
cdef cnp.ndarray[cnp.float64_t, ndim=1] mean_vec 
cdef cnp.ndarray[cnp.int64_t, ndim=1] num_vec 
cdef int id # pointer for loops
cdef cnp.ndarray[cnp.int64_t, ndim=1] mean_idx 
cdef cnp.ndarray[cnp.float64_t, ndim=1] sum_vec 
cdef cnp.ndarray[cnp.float64_t, ndim=1] sum_all 
cdef double rss_cat
cdef cnp.ndarray[cnp.float64_t, ndim=1] pivot_c
"""