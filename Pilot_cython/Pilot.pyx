cimport numpy as cnp
import numpy as np
import pandas as pd
from Tree cimport tree
from libc.string cimport strcmp
from split_function cimport best_split
from split_function cimport loss_fun

cnp.import_array()

cdef class PILOT():
    """
    This is an implementation of the PILOT method.

    Attributes:
    -----------
    max_depth: int,
        the max depth allowed to grow in a tree.
    split_criterion: str,
        the criterion to split the tree,
        we have 'AIC'/'AICc'/'BIC'/'adjusted R^2', etc.
    regression_nodes: list,
        A list of regression models used.
        They are 'con', 'lin', 'blin', 'pcon', 'plin'.
    min_sample_split: int,
        the minimal number of samples required
        to split an internal node.
    min_sample_leaf: int,
        the minimal number of samples required
        to be at a leaf node.
    step_size: int,
        boosting step size.
    X: ndarray,
        2D float array of the predictors.
    y, y0: ndarray,
        2D float array of the responses.
    sorted_X_indices: ndarray,
        2D int array of sorted indices according to each feature.
    n_feature: int,
        number of features
    categorical: ndarray,
        1D int array indicating categorical predictors.
    model_tree: tree object,
        learned PILOT model tree.
    B1, B2: int
        upper and lower bound for the first truncation,
        learned from y.
    """

    cdef public int max_depth
    cdef public int min_sample_split
    cdef public int min_sample_leaf
    cdef public int step_size
    cdef public double truncation_factor
    cdef public int min_unique_values_regression
    cdef public cnp.int64_t max_features_considered

    cdef cnp.ndarray X
    cdef cnp.ndarray y
    cdef cnp.ndarray y0
    cdef cnp.ndarray sorted_X_indices
    cdef double ymean
    cdef int n_features
    # cdef max_features_considered
    cdef cnp.ndarray categorical
    cdef tree model_tree
    cdef double B1
    cdef double B2
    cdef int tree_depth
    cdef cnp.ndarray k_con
    cdef cnp.ndarray k_lin
    cdef cnp.ndarray k_split_nodes 
    cdef cnp.ndarray k_pconc 

    # TODO: annotation for the init function?
    def __init__(
        self,
        max_depth=12,
        #split_criterion="BIC",
        min_sample_split=10,
        min_sample_leaf=5,
        step_size=1,
        #random_state=42,
        truncation_factor = 1.5,
        #rel_tolerance: float = 0,
        min_unique_values_regression = 5,
    ) -> None:
        """
        Here we input model parameters to build a tree,
        not all the parameters for split finding.

        parameters:
        -----------
        max_depth: int,
            the max depth allowed to grow in a tree.
        split_criterion: str,
            the criterion to split the tree,
            we have 'AIC'/'AICc'/'BIC'/'adjusted R^2', etc.
        min_sample_split: int,
            the minimal number of samples required
            to split an internal node.
        min_sample_leaf: int,
            the minimal number of samples required
            to be at a leaf node.
        step_size: int,
            boosting step size.
        random_state: int,
            Not used, added for compatibility with sklearn framework
        truncation_factor: float,
            By default, predictions are truncated at [-3B, 3B] where B = y_max = -y_min for centered data.
            The multiplyer (3 by default) can be adapted.
        rel_tolerance: float,
            Minimum percentage decrease in RSS in order for a linear node to be added (if 0, there is no restriction on the number of linear nodes).
            Used to avoid recursion errors.
        df_settings:
            Mapping from regression node type to the number of degrees of freedom for that node type.
        regression_nodes:
            List of node types to consider for numerical features. If None, all available regression nodes are considered
        """

        # initialize class attributes
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.step_size = step_size
        # self.random_state = random_state
        self.truncation_factor = truncation_factor
        # self.rel_tolerance = rel_tolerance
        self.min_unique_values_regression = min_unique_values_regression

        # attributes used for fitting
        """
        self.X = None
        self.y = None
        self.y0 = None
        self.sorted_X_indices = None
        self.ymean = None
        self.n_features = None
        self.max_features_considered = None
        self.model_tree = None
        self.B1 = None
        self.B2 = None
        """

        # self.recursion_counter = {"lin": 0, "blin": 0, "pcon": 0, "plin": 0, "pconc": 0}
        self.tree_depth = 0

        """
        # order of preference for regression nodes
        # this cannot be changed as best split relies on this specific order
        self.regression_nodes = [
            node for node in NODE_PREFERENCE_ORDER if node in self.regression_nodes
        ]
        """

        """
        # degrees of freedom for each regression node
        self.k = DEFAULT_DF_SETTINGS.copy()
        if df_settings is not None:
            self.k.update(df_settings)
        """

        """
        # df need to be stored as separate numpy arrays for numba
        self.k = {k: np.array([v], dtype=np.int64) for k, v in self.k.items()}
        """
        
        self.k_con = np.array([1], dtype = np.int64)
        self.k_lin = np.array([2], dtype = np.int64)
        self.k_split_nodes = np.array([5, 5, 7], dtype = np.int64)
        self.k_pconc = np.array([5], dtype = np.int64)

    cdef bint stop_criterion(self, int tree_depth, cnp.ndarray[cnp.float64_t, ndim=2] y):
        """
        Stop splitting when either the tree has reached max_depth or the number of the
        data in the leaf node is less than min_sample_leaf or the variance of the node
        is less than the threshold.

        parameters:
        -----------
        tree_depth: int,
            Current depth.
        y: ndarray,
            2D float array. The response variable.

        returns:
        --------
        boolean:
            whether to stop the recursion.

        """
        if tree_depth >= self.max_depth or y.shape[0] <= self.min_sample_split:
            return False
        return True

    cdef tree build_tree(self, int tree_depth, cnp.ndarray[cnp.int64_t, ndim=1] indices, double rss):
        """
        This function is based on the recursive algorithm. We keep
        growing the tree, until it meets the stopping criterion.
        The parameters root is to save the tree structure.

        parameters:
        -----------
        tree_depth: int,
            the depth of the tree. By definition, the depth
            of the root node is 0.
        indices: ndarray,
            1D array containing data with int type. It gives
            the indices of cases in this node, which will
            induce sorted_X.
        rss: float,
            The rss of the current node before fitting a model.

        return:
        -------
        tree object:
            If it meets stop_criterion or can not be further split,
            return end node (denoted by 'END').

        """

        # cdef double rss_previous
        # cdef double rss_new
        # cdef double improvement
        cdef cnp.ndarray[cnp.float64_t, ndim=2] raw_res
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast=True] cond
        cdef cnp.ndarray[cnp.int64_t, ndim=1] indices_left
        cdef cnp.ndarray[cnp.int64_t, ndim=1] indices_right
        cdef cnp.ndarray[cnp.float64_t, ndim=2] rawres_left
        cdef cnp.ndarray[cnp.float64_t, ndim=2] rawres_right

        cdef int best_feature
        cdef double best_pivot
        cdef str best_node
        cdef str last_node
        cdef cnp.ndarray[cnp.float64_t, ndim=1] lm_l
        cdef cnp.ndarray[cnp.float64_t, ndim=1] lm_r
        cdef cnp.ndarray[cnp.float64_t, ndim=1] interval
        cdef cnp.ndarray[cnp.float64_t, ndim=1] pivot_c

        tree_depth += 1
        # fit models on the node 
        best_feature, best_pivot, best_node, lm_l, lm_r, interval, pivot_c = best_split(
            indices,
            # self.regression_nodes,
            self.n_features,
            self.sorted_X_indices,
            self.X,
            self.y,
            # self.split_criterion,
            self.min_sample_leaf,
            self.k_con,
            self.k_lin,
            self.k_split_nodes,
            self.k_pconc,
            self.categorical,
            self.max_features_considered,
            self.min_unique_values_regression,
        )  # find the best split
        # stop fitting the tree
        # if strcmp(best_node, "") == 0:
        if best_node == '':
            last_node = 'END'
            return tree(node=last_node, Rt=rss)
        # elif strcmp(best_node, "con") == 0 or strcmp(best_node, "lin") == 0:
        elif best_node == 'con' or best_node == 'lin':
            # do not include 'lin' and 'con' in the depth calculation
            tree_depth -= 1

        self.tree_depth = max(self.tree_depth, tree_depth)

        # build tree only if it doesn't meet the stop_criterion
        if self.stop_criterion(tree_depth, self.y[indices]):
            # define a new node
            # best_feature should - 1 because the 1st column is the indices
            node = tree(
                best_node,
                best_feature - 1, 
                best_pivot,
                lm_l,
                lm_r,
                Rt=rss,
                depth=tree_depth + 1,
                interval=interval,
                pivot_c=pivot_c,
            )

            # update X and y by vectorization, reshape them to make sure their sizes are correct
            # if strcmp(best_node, "lin") == 0:
            if best_node == 'lin':
                # rss_previous = np.sum(self.y[indices] ** 2)
                # unpdate y
                raw_res = self.y[indices] - self.step_size * (
                    lm_l[0] * self.X[indices, best_feature].reshape(-1, 1) + lm_l[1]
                )
                # truncate the prediction
                self.y[indices] = self.y0[indices] - np.maximum(
                    np.minimum(self.y0[indices] - raw_res, self.B1), self.B2
                )
                """
                rss_new = np.sum(self.y[indices] ** 2)
                improvement = (rss_previous - rss_new) / rss_previous
                if improvement < self.rel_tolerance:
                    node.left = tree(node="END", Rt=np.sum(self.y[indices] ** 2))
                    return node

                else:
                    self.recursion_counter[best_node] += 1
                """
                # recursion
                node.left = self.build_tree(
                    tree_depth,
                    indices,
                    np.maximum(0, np.sum((self.y[indices] - np.mean(self.y[indices])) ** 2)),
                )

            # elif strcmp(best_node, "con") == 0:
            elif best_node == "con":
                self.y[indices] -= self.step_size * (lm_l[1])

                # stop the recursion
                last_node = 'END'
                node.left = tree(node=last_node, Rt=np.sum(self.y[indices] ** 2))
                return node

            else:
                # find the indices for the cases in the left and right node
                # if strcmp(best_node, "pconc") == 0:
                if best_node == "pconc":
                    cond = np.isin(self.X[indices, best_feature], pivot_c)
                else:
                    cond = self.X[indices, best_feature] <= best_pivot
                indices_left = (self.X[indices][cond, 0]).astype(np.int64)
                indices_right = (self.X[indices][~cond, 0]).astype(np.int64)

                # compute the raw and truncated predicrtion
                rawres_left = (
                    self.y[indices_left]
                    - (lm_l[0] * self.X[indices_left, best_feature].reshape(-1, 1) + lm_l[1])
                ).copy()
                self.y[indices_left] = self.y0[indices_left] - np.maximum(
                    np.minimum(self.y0[indices_left] - rawres_left, self.B1), self.B2
                )
                rawres_right = (
                    self.y[indices_right]
                    - (lm_r[0] * self.X[indices_right, best_feature].reshape(-1, 1) + lm_r[1])
                ).copy()
                self.y[indices_right] = self.y0[indices_right] - np.maximum(
                    np.minimum(self.y0[indices_right] - rawres_right, self.B1), self.B2
                )

                node.left = self.build_tree(
                    tree_depth,
                    indices_left,
                    np.maximum(
                            0,
                            np.sum((self.y[indices_left] - np.mean(self.y[indices_left])) ** 2),
                        ),
                    )

                node.right = self.build_tree(
                    tree_depth,
                    indices_right,
                    np.maximum(
                        0,
                        np.sum((self.y[indices_right] - np.mean(self.y[indices_right])) ** 2),
                    ),
                )

                """
                # recursion
                try:
                    self.recursion_counter[best_node] += 1
                    node.left = self.build_tree(
                        tree_depth,
                        indices_left,
                        np.maximum(
                            0,
                            np.sum((self.y[indices_left] - np.mean(self.y[indices_left])) ** 2),
                        ),
                    )

                    node.right = self.build_tree(
                        tree_depth,
                        indices_right,
                        np.maximum(
                            0,
                            np.sum((self.y[indices_right] - np.mean(self.y[indices_right])) ** 2),
                        ),
                    )
                except RecursionError as re:
                    print(tree_depth, best_node, node.nodes_selected(), self.recursion_counter)
                    raise Exception from re
                """

        else:
            # stop recursion if meeting the stopping criterion
            last_node = 'END'
            return tree(node=last_node, Rt=rss)

        return node

    cpdef void fit(
        self,
        cnp.ndarray[cnp.float64_t, ndim=2] X,
        cnp.ndarray[cnp.float64_t, ndim=2] y,
        object categorical = None
        #max_features_considered: Optional[int] = None,
        #**kwargs,
    ):
        """
        This function is used for model fitting. It should return
        a pruned tree, which includes the location of each node
        and the linear model for it. The results should be saved
        in the form of class attributes.

        parameters:
        -----------
        X: Array-like objects, usually pandas.DataFrame or numpy arrays.
            The predictors.
        y: Array-like objects, usually pandas.DataFrame or numpy arrays.
            The responses.
        categorical: An array of column indices of categorical variables.
                     We assume that they are integer valued.

        return:
        -------
        None
        """

        cdef int n_samples
        cdef cnp.ndarray[cnp.int64_t, ndim=2] sorted_indices
        cdef double padding

        # X and y should have the same size
        assert X.shape[0] == y.shape[0]

        """
        # Switch pandas objects to numpy objects
        if isinstance(X, pd.core.frame.DataFrame):
            X = np.array(X)

        if isinstance(y, pd.core.frame.DataFrame):
            y = np.array(y)
        elif y.ndim == 1:
            y = y.reshape((-1, 1))
        """

        # define class attributes
        self.n_features = X.shape[1]
        
        """
        self.max_features_considered = (
            min(max_features_considered, self.n_features)
            if max_features_considered is not None
            else self.n_features
        )
        """

        n_samples = X.shape[0]
        self.max_features_considered = X.shape[1]
        if categorical is None:
            self.categorical = np.array([X.shape[1] + 1], dtype = np.int64)
        else : 
            self.categorical = np.append(categorical, [X.shape[1] + 1])
        # print(self.categorical)

        # insert indices to the first column of X to memorize the indices
        self.X = np.c_[np.arange(0, n_samples), X]

        # Memorize the indices of the cases sorted along each feature
        # Do not sort the first column since they are just indices
        sorted_indices = np.array(
            [
                np.argsort(self.X[:, feature_id], axis=0).flatten()
                for feature_id in range(1, self.n_features + 1)
            ]
        )
        self.sorted_X_indices = (self.X[:, 0][sorted_indices]).astype(np.int64)
        # ->(n_samples, n_features) 2D array

        # y should be remembered and modified during 'boosting'
        self.y = y.copy()  # calculate on y directly to save memory
        self.y0 = y.copy()  # for the truncation procudure
        padding = (self.truncation_factor - 1.) * ((y.max() - y.min()) / 2)
        self.B1 = y.max() + padding  # compute the upper bound for the first truncation
        self.B2 = y.min() - padding  # compute the lower bound for the second truncation

        # build the tree, only need to take in the indices for X
        self.model_tree = self.build_tree(-1, self.sorted_X_indices[0], np.sum((y - y.mean()) ** 2))

        # if the first node is 'con'
        # if strcmp(self.model_tree.node, "END") == 0:
        if self.model_tree.node == "END":
            self.ymean = y.mean()

        return

    cpdef predict(self, cnp.ndarray[cnp.float64_t, ndim=2] X, tree model=None, int maxd=100):
        """
        This function is used for model predicting. Given a dataset,
        it will find its location and respective linear model.

        parameters:
        -----------
        model: The tree objects
        x: Array-like objects, new sample need to be predicted
        maxd: The maximum depth to be considered for prediction,
              can be less than the true depth of the tree.

        return:
        -------
        y_hat: numpy.array
               the predicted y values
        """
        cdef int n_test_samples = X.shape[0]
        cdef cnp.ndarray[cnp.float64_t, ndim=1] y_hat = np.ones(n_test_samples, dtype = np.float64)
        cdef tree t
        cdef double y_hat_one\
        
        if model is None:
            model = self.model_tree

        """
        if isinstance(X, pd.core.frame.DataFrame):
            X = np.array(X)
        """
        # if strcmp(self.model_tree.node, "END") == 0:
        if self.model_tree.node == "END":
            y_hat = y_hat * self.ymean
            return y_hat

        for row in range(n_test_samples):
            t = model
            y_hat_one = 0.
            while (t.node != 'END') and t.depth < maxd:
                if (t.node == 'pconc'):
                    if np.isin(X[row, t.feature], t.pivot_c):
                        y_hat_one += self.step_size * (t.lm_l[1])
                        t = t.left
                    else:
                        y_hat_one += self.step_size * (t.lm_r[1])
                        t = t.right

                # go left if 'lin'
                elif t.node == 'lin':
                    # truncate both on the left and the right
                    y_hat_one += self.step_size * (
                        t.lm_l[0]
                        * np.min(
                            [
                                np.max([X[row, t.feature], t.interval[0]]),
                                t.interval[1],
                            ]
                        )
                        + t.lm_l[1]
                    )
                    t = t.left

                elif t.node == 'con' or X[row, t.feature] <= t.pivot:
                    # truncate on the left
                    y_hat_one += self.step_size * (
                        t.lm_l[0] * np.max([X[row, t.feature], t.interval[0]]) + t.lm_l[1]
                    )
                    t = t.left

                else:
                    y_hat_one += self.step_size * (
                        t.lm_r[0] * np.min([X[row, t.feature], t.interval[1]]) + t.lm_r[1]
                    )
                    t = t.right

                # truncation
                if y_hat_one > self.B1:
                    y_hat_one = self.B1
                elif y_hat_one < self.B2:
                    y_hat_one = self.B2

            y_hat[row] = y_hat_one

        return y_hat