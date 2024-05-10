""" This is an implementation of the tree data structure"""
cimport numpy as cnp
import numpy as np

cnp.import_array()

cdef class tree:
    
    """
    cdef public tree left
    cdef public tree right
    cdef public double Rt
    cdef public char* node
    cdef public int feature
    cdef public double pivot
    cdef public cnp.ndarray lm_l # a cython type rather than a typed memoryview
    cdef public cnp.ndarray lm_r
    cdef public int depth
    cdef public cnp.ndarray interval
    cdef public cnp.ndarray pivot_c
    """
    
    """
    We use a tree object to save the PILOT model.

    Attributes:
    -----------

    node: str,
        type of the regression model
        'lin', 'blin', 'pcon', 'plin' or 'END' to denote the end of the tree
    pivot: tuple,
        a tuple to indicate where we performed a split. The first
        coordinate is the feature_id and the second one is
        the pivot.
    lm_l: ndarray,
        a 1D array to indicate the linear model for the left child node. The first element
        is the coef and the second element is the intercept.
    lm_r: ndarray,
        a 1D array to indicate the linear model for the right child node. The first element
        is the coef and the second element is the intercept.
    Rt: float,
        a real number indicating the rss in the present node.
    depth: int,
        the depth of the current node/subtree
    interval: ndarray,
        1D float array for the range of the selected predictor in the training data
    pivot_c: ndarry,
        1D int array. Indicating the levels in the left node
        if the selected predictor is categorical
    """

    def __init__(
        self,
        node=None,
        feature=0,
        pivot=0.,
        lm_l=np.array([0.,0.]),
        lm_r=np.array([0.,0.]),
        Rt=None,
        depth=100,
        interval=np.array([0.,1.]),
        pivot_c=np.array([0]),
    ) -> None:
        """
        Here we input the tree attributes.

        parameters:
        ----------
        node: str,
            type of the regression model
            'lin', 'blin', 'pcon', 'plin' or 'END' to denote the end of the tree
        feature: int,
            an interger for the selected best feature
        pivot: int,
            an interger to indicate where we performed a split
        lm_l: ndarray,
            a 1D array to indicate the linear model for the left child node. The first element
            is the coef and the second element is the intercept.
        lm_r: ndarray,
            a 1D array to indicate the linear model for the right child node. The first element
            is the coef and the second element is the intercept.
        Rt: float,
            a real number indicating the rss in the present node.
        depth: int,
            the depth of the current node/subtree
        interval: ndarray,
            1D float array for the range of the selected predictor in the training data
        pivot_c: ndarry,
            1D int array. Indicating the levels in the left node
            if the selected predictor is categorical

        """
        self.left = None  # go left by default if node is 'lin'
        self.right = None
        self.Rt = Rt
        self.node = node
        self.feature = feature
        self.pivot = pivot
        self.lm_l = np.asarray(lm_l, dtype=np.float64)
        self.lm_r = np.asarray(lm_r, dtype=np.float64)
        self.depth = depth
        self.interval = np.asarray(interval, dtype=np.float64)
        self.pivot_c = np.asarray(pivot_c, dtype=np.float64)

# an instance of tree
cpdef test_tree():
    cdef double Rt = 0.985
    cdef char* node = 'blin'
    cdef int feature = 3
    cdef double pivot = 0.5
    cdef cnp.ndarray[cnp.float64_t, ndim=1] lm_l = np.array([1.2, 0.3], dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] lm_r = np.array([-0.5, 1], dtype=np.float64)
    cdef int depth = 1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] interval = np.array([-3.,3.], dtype=np.float64)
    cdef tree tree_example

    tree_example = tree(Rt=Rt, node=node, feature=feature, pivot=pivot, lm_l=lm_l, lm_r=lm_r, depth=depth, interval=interval)
    print(tree_example.depth)
    print(tree_example.lm_l)

    return(tree_example.pivot)