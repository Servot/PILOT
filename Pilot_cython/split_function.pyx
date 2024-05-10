# cython: profile=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIIL=1

cimport numpy as cnp
import numpy as np
from libc.string cimport strcmp

cnp.import_array()

def random_sample(a, k):
    a = np.random.choice(a, size=k, replace=False)
    return a

cpdef best_split(
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
):
    """
    This function finds the best split as well as the linear
    model on the node.

    parameters:
    -----------
    index: ndarray,
        1D int array. the indices of samples in the current node, which
        would induce sorted_X and sorted_y.
    regression_nodes: list,
        a list indicating which kinds of node is used,
        we have 'lin'/'pcon'/'blin'/'plin'.
    n_features: int,
        number of features.
    sorted_X_indices: ndarray,
        2D int array. Sorted indices of cases, according to each feature.
    X: ndarray,
        2D float array, the predictors.
    y: ndarray,
        2D float array, the response.
    split_criterion: str, TO BE DEPRECATED!!!
        the criterion to split the tree,
        default is 'BIC', can also be 'AIC'/'AICc', etc.
    min_sample_leaf: int,
        the minimal number of samples required
        to be at a leaf node
    k_*: ndarray
        degrees of freedom for each regression node
    categorical: ndarray,
        1D int array, the columns of categorical variable, array.
    max_features_considered: int
        number of features to consider for each split (randomly sampled)
    min_unique_values_regression: int
        minimum number of unique values necessary to consider a linear node

    returns:
    --------
    best_feature: int,
        The feature/predictor id at which the dataset is best split.
        if it is a categorical feature, the second element is a list of values
        indicating the left region.
    best_pivot: float,
        The feature id at which the dataset is best split.
    best_node: str,
        The best regression model.
    lm_L: ndarray,
        1D float array. The linear model on the left node (intercept, coeficents).
    lm_R:  ndarray,
        1D float array. The linear model on the right node (intercept, coeficents).
        for 'lin' and 'con': lm_R is None, all information is included in lm_L
    interval: ndarray,
        1D float array. The range of the training data on this node
    pivot_c: ndarray,
        1D int array. An array of the levels belong to the left node.
        Used if the chosen feature/predictor is categorical.

    Remark:
    -------
    If the input data is not allowed to split, the function will return default
    values.
    """

    # Initialize output variables, should be consistent with the variable type
    cdef double best_pivot = -1.0
    cdef str best_node = ""
    cdef double best_loss = -1.0
    cdef int best_feature = -1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] lm_L = np.array([0.0, 0.0])
    cdef cnp.ndarray[cnp.float64_t, ndim=1]lm_R = np.array([0.0, 0.0])
    cdef cnp.ndarray[cnp.float64_t, ndim=1] interval = np.array([-np.inf, np.inf])

    cdef char* regression_nodes[3] # NEW: TO BE TREATED AS AN INPUT
    regression_nodes[0] = "blin"
    regression_nodes[1] = "pcon"
    regression_nodes[2] = "plin"

    # Initialize the coef and intercept for 'blin'/'plin'/'pcon'
    cdef int l = 3
    cdef cnp.ndarray[cnp.float64_t, ndim=2] coef = np.zeros((l, 2)) * np.nan
    cdef cnp.ndarray[cnp.float64_t, ndim=2] intercept = np.zeros((l, 2)) * np.nan

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
    cdef double coef_con = 0 # NEW: SHOULD BE FLOAT?
    cdef double rss_1d
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rss = np.zeros(3) * np.nan
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
    cdef cnp.ndarray[cnp.float64_t, ndim=1] pivot_c = np.array([0.])

    # search for the best split among all features, negelecting the indices column
    for feature_id in range(1, n_features + 1):
        # get sorted X, y NEW: 1D ARRAY?
        idx = sorted_X_indices[feature_id - 1] # NEW: ANNOTATION
        idx = idx[np.isin(idx, index)]
        X_sorted = X[idx] # NEW: ANNOTATION
        y_sorted = y[idx] # NEW: .COPY DROPPED

        # Initialize possible pivots
        possible_p = np.unique(X_sorted[:, feature_id]) # NEW: ANNOTATION
        lenp = possible_p.shape[0] # NEW: ANNOTATION, DROP LEN()

        if feature_id - 1 < categorical[pointer_c]: # NEW: ALL IN FUNCTION NEEDS TO BE OPTIMIZED,
                                              # EITHER NUMPY.ISIN OR EXPLICIT LOOP OR OTHER IMPLEMENTATIONS
            num = np.array([0, X_sorted.shape[0]], dtype = np.int64)

            # store entries of the Gram and moment matrices
            Moments = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [
                        np.sum(X_sorted[:, feature_id]),
                        np.sum(X_sorted[:, feature_id] ** 2),
                        np.sum(X_sorted[:, feature_id].copy().reshape(-1, 1) * y_sorted),
                        np.sum(y_sorted),
                        np.sum(y_sorted**2),
                    ],
                ]
            )

            # CON: NEW DELETE IF CON
            # print(num)
            intercept_con = Moments[1, 3] / num[1]
            # compute the RSS and the loss according to the information criterion
            rss_1d = (
                Moments[1, 4]
                + (num[1] * intercept_con**2)
                - 2 * intercept_con * Moments[1, 3]
            )
            loss = loss_fun(
                # criteria=split_criterion,
                num=num[1],
                Rss=np.array([rss_1d]),
                k=k_con,
            )
            # update best_loss immediately
            if best_node == "" or loss.item() < best_loss: # NEW: THEY ARE ALL DEFINED
                best_node = "con"
                best_loss = loss.item()
                best_feature = feature_id
                interval = np.array([possible_p[0], possible_p[-1]])
                lm_L = np.array([coef_con, intercept_con])

            # LIN: NEW DELETE IF LIN
            if lenp >= min_unique_values_regression:
                var = num[1] * Moments[1, 1] - Moments[1, 0] ** 2 # NEW: FLOAT
                # in case a constant feature
                if var == 0.0:
                    coef_lin = 0.0
                else:
                    coef_lin = (num[1] * Moments[1, 2] - Moments[1, 0] * Moments[1, 3]) / var # NEW: FLOAT
                intercept_lin = (Moments[1, 3] - coef_lin * Moments[1, 0]) / num[1]
                # compute the RSS and the loss according to the information criterion
                rss_1d = (
                    Moments[1, 4]
                    + (num[1] * intercept_lin**2)
                    + (2 * coef_lin * intercept_lin * Moments[1, 0])
                    + coef_lin**2 * Moments[1, 1]
                    - 2 * intercept_lin * Moments[1, 3]
                    - 2 * coef_lin * Moments[1, 2]
                )
                loss = loss_fun(
                    # criteria=split_criterion,
                    num=num[1],
                    Rss=np.array([rss_1d]),
                    k=k_lin,
                )
                # update best_loss immediately
                if best_node == "" or loss.item() < best_loss:
                    best_node = "lin"
                    best_loss = loss.item()
                    best_feature = feature_id
                    interval = np.array([possible_p[0], possible_p[-1]])
                    lm_L = np.array([coef_lin, intercept_lin])

            # For blin, we need to maintain another Gram/moment matrices and the knot xi
            # NEW: ALWAYS BLIN
            # Moments need to be updated for blin:
            # [sum(x-xi)+, sum[(x-xi)+]**2, sum[x(x-xi)+], sum[y(x-xi)+]]
            XtX = np.array(
                [
                    [
                        np.float64(num.sum()),
                        Moments[:, 0].sum(),
                        Moments[:, 0].sum(),
                    ],
                    [Moments[:, 0].sum(), Moments[:, 1].sum(), Moments[:, 1].sum()],
                    [Moments[:, 0].sum(), Moments[:, 1].sum(), Moments[:, 1].sum()],
                ]
            )
            XtY = np.array([[Moments[1, 3]], [Moments[1, 2]], [Moments[1, 2]]])
            pre_pivot = 0.0

            # pcon, blin and plin: try each possible split and
            # find the best one the last number are never used for split
            for p in range(possible_p.shape[0] - 1):
                # The pointer to select the column of coef and intercept
                i = 0
                pivot = possible_p[p]
                # update cases in the left region
                # NEW: DELETE index_add = (X_sorted[:, feature_id] == pivot).astype(np.bool_)
                X_add = X_sorted[X_sorted[:, feature_id] == pivot, feature_id]
                y_add = y_sorted[X_sorted[:, feature_id] == pivot]

                # BLIN:
                # First maintain xi
                xi = pivot - pre_pivot

                # update XtX and XtY
                XtX += np.array(
                    [
                        [0.0, 0.0, -xi * num[1]],
                        [0.0, 0.0, -xi * Moments[1, 0]],
                        [
                            -xi * num[1],
                            -xi * Moments[1, 0],
                            xi**2 * num[1] - 2 * xi * XtX[0, 2],
                        ],
                    ]
                )
                XtY += np.array([[0.0], [0.0], [-xi * Moments[1, 3]]])

                # useless to check the first pivot or partition that
                # leads to less than min_sample_leaf samples
                if (
                    pivot != possible_p[0]
                    and p >= 1
                    and lenp >= min_unique_values_regression
                    and np.linalg.det(XtX) > 0.001
                    and num[0] + X_add.shape[0] >= min_sample_leaf
                    and num[1] - X_add.shape[0] >= min_sample_leaf
                ):
                    coefs = np.linalg.solve(XtX, XtY).flatten()
                    coef[i, :] = np.array([coefs[1], coefs[1] + coefs[2]])
                    intercept[i, :] = np.array([coefs[0], coefs[0] - coefs[2] * pivot])
                i += 1  # we add a dimension to the coef and intercept arrays， i = 1
                pre_pivot = pivot

                # update num after blin is fitted
                num += np.array([1, -1]) * X_add.shape[0]

                # first update moments then check if this pivot is eligable for a pcon/plin split
                Moments_add = np.array(
                    [
                        np.sum(X_add),
                        np.sum(X_add**2),
                        np.sum(X_add.reshape(-1, 1) * y_add),
                        np.sum(y_add),
                        np.sum(y_add**2),
                    ]
                )
                Moments += Moments_add * np.array([[1.0], [-1.0]])

                # negelect ineligable split
                if num[0] < min_sample_leaf:
                    continue
                elif num[1] < min_sample_leaf:
                    break

                # 'pcon' fit
                coef[i, :] = np.array([0, 0])
                intercept[i, :] = (Moments[:, 3]) / num
                i += 1  # we add a dimension to the coef and intercept arrays i = 2

                # 'plin' for the first split candidate is equivalent to 'pcon'
                if (
                    p
                    >= min_unique_values_regression
                    - 1  # number of unique values smaller than current value
                    and lenp - p
                    >= min_unique_values_regression  # number of unique values larger than current value
                    and ~np.isin(0., num * Moments[:, 1] - Moments[:, 0] ** 2) # remove constant feature?
                ):
                    # coef and intercept are vectors of dimension 1
                    # have to reshape X column in order to get correct cross product
                    # the intercept should be divided by the total number of samples
                    coef[i, :] = (num * Moments[:, 2] - Moments[:, 0] * Moments[:, 3]) / (
                        num * Moments[:, 1] - Moments[:, 0] ** 2
                    )
                    intercept[i, :] = (Moments[:, 3] - coef[i, :] * Moments[:, 0]) / num

                # compute the rss and loss of the above 3 methods
                # The dimension rss is between 1 and 3 (depending on the regression_nodes)
                # TODO: HOW TO DEFINE RSS?
                rss = (
                    Moments[:, 4]
                    + (num * intercept**2)
                    + (2 * coef * intercept * Moments[:, 0])
                    + coef**2 * Moments[:, 1]
                    - 2 * intercept * Moments[:, 3]
                    - 2 * coef * Moments[:, 2]
                ).sum(axis=1)

                # if no fit is done, continue
                if np.isnan(rss).all():
                    continue

                # update the best loss
                rss = np.maximum(10**-8, rss)
                loss = loss_fun(
                    # criteria=split_criterion,
                    num=num.sum(),
                    Rss=rss,
                    k=k_split_nodes,
                )

                if ~np.isnan(loss).all() and (best_node == "" or np.nanmin(loss) < best_loss):
                    best_loss = np.nanmin(loss)
                    index_min = np.where(loss == best_loss)[0][0]
                    best_node = regression_nodes[index_min].decode('utf-8') # NEW: REGRESSION_NODES modified
                    best_feature = feature_id  # asigned but will not be used for 'lin'
                    interval = np.array([possible_p[0], possible_p[-1]])
                    best_pivot = pivot
                    lm_L = np.array([coef[index_min, 0], intercept[index_min, 0]])
                    lm_R = np.array([coef[index_min, 1], intercept[index_min, 1]])

            continue

        # CATEGORICAL VARIABLES
        mean_vec = np.zeros(lenp)
        num_vec = np.zeros(lenp, dtype = np.int64)
        for id in range(lenp):
            # mean values of the response w.r.t. each level
            mean_vec[id] = np.mean(y_sorted[X_sorted[:, feature_id] == possible_p[id]])
            # number of elements at each level
            num_vec[id] = y_sorted[X_sorted[:, feature_id] == possible_p[id]].shape[0]

        # sort unique values w.r.t. the mean of the responses
        mean_idx = np.argsort(mean_vec)
        num_vec = num_vec[mean_idx]
        sum_vec = mean_vec[mean_idx] * num_vec
        possible_p = possible_p[mean_idx]

        # loop over the sorted possible_p and find the best partition
        num = np.array([0, X_sorted.shape[0]], dtype = np.int64)
        sum_all = np.array([0., np.sum(y_sorted)])
        for id in range(lenp - 1):
            # update the sum and num
            sum_all += np.array([1.0, -1.0]) * sum_vec[id]
            num += np.array([1, -1]) * num_vec[id]
            # find the indices of the elements in the left node, NEW: np.isin
            sub_index = np.isin(X_sorted[:, feature_id], possible_p[: id + 1])
            # compute the rss_cat
            rss_cat = np.sum((y_sorted[sub_index] - sum_all[0] / num[0]) ** 2) + np.sum(
                (y_sorted[~sub_index] - sum_all[1] / num[1]) ** 2
            )
            rss_cat = np.maximum(10**-8, rss_cat)
            loss = loss_fun(
                # criteria=split_criterion,
                num=num.sum(),
                Rss=np.array([rss_cat]),
                k=k_pconc,
            )
            if best_node == "" or loss.item() < best_loss:
                best_feature = feature_id
                best_node = "pconc"
                best_loss = loss.item()
                lm_L = np.array([0, sum_all[0] / num[0]])
                lm_R = np.array([0, sum_all[1] / num[1]])
                pivot_c = possible_p[: id + 1].copy()
                # pivot_c = pivot_c.astype(np.int64)

        # update the pointer
        pointer_c += 1


    result = (best_feature, best_pivot, best_node, lm_L, lm_R, interval, pivot_c)
    """
    print(best_feature)
    print(best_pivot)
    print(best_node)
    print(lm_L)
    print(lm_R)
    print(interval)
    print(pivot_c)
    """

    return result

cdef loss_fun(int num, 
              # cnp.ndarray[cnp.float64_t, ndim=1] 
              Rss, 
              # cnp.ndarray[cnp.int64_t, ndim = 1] 
              k):
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
    loss =  num * np.log(Rss / num) + np.log(num) * k

    return loss

def fit_one_step():
    # initilize the variables
    cdef cnp.ndarray[cnp.int64_t, ndim=1] index
    cdef cnp.int64_t n_features = 10
    cdef cnp.ndarray[cnp.int64_t, ndim=2] sorted_X_indices
    cdef cnp.ndarray[cnp.float64_t, ndim=2] X
    cdef cnp.ndarray[cnp.float64_t, ndim=2] y
    cdef cnp.int64_t min_sample_leaf = 5
    cdef cnp.ndarray[cnp.int64_t, ndim=1] k_con = np.array([1], dtype = np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] k_lin = np.array([2], dtype = np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] k_split_nodes = np.array([5, 5, 7], dtype = np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] k_pconc = np.array([5], dtype = np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] categorical = np.array([11], dtype = np.int64) # in total 10 rows
    cdef cnp.int64_t max_features_considered = 10
    cdef cnp.int64_t min_unique_values_regression = 5

    # intermidiate variables
    cdef cnp.ndarray[cnp.int64_t, ndim=2] sorted_indices
    cdef int n_samples = 100


    # return variable, this can not be annotated in Cython
    """
    cdef (cnp.int64_t, 
      cnp.ndarray[cnp.float64_t, ndim=1],
      char*, 
      cnp.ndarray[cnp.float64_t, ndim=1],
      cnp.ndarray[cnp.float64_t, ndim=1],
      cnp.ndarray[cnp.float64_t, ndim=1],
      cnp.ndarray[cnp.int64_t, ndim=1]) result
    """

    # input X and y
    X = 2 * np.random.rand(100, 5000)
    y = (3 * X[:,0]**2 + X[:,8] - X[:,100]**3 ).reshape(-1, 1)
    X = np.c_[np.arange(0, n_samples), X]

    # index and sorted_X_indices
    index = 1 + np.arange(n_samples, dtype = np.int64)
    sorted_indices = np.array(
            [
                np.argsort(X[:, feature_id], axis=0).flatten()
                for feature_id in range(1, n_features + 1)
            ]
        )
    sorted_X_indices = X[:, 0][sorted_indices].astype(np.int64)

    # print(sorted_X_indices)

    # run split function
    result = best_split(index, # NEW: regression_nodes deleted
    n_features,
    sorted_X_indices,
    X,          
    y,
    min_sample_leaf,
    k_con,
    k_lin,
    k_split_nodes,
    k_pconc,
    categorical,
    max_features_considered,
    min_unique_values_regression)

    print(result)
    
    return result