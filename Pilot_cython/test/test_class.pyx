cimport numpy as cnp
import numpy as np

cnp.import_array()

cdef class test():
    cdef public cnp.ndarray X
    cdef public cnp.ndarray y

    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=1] X) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.array([1.,2.], dtype=np.float64)

    cdef void update_X(self):
        self.X += 1.
        self.X = self.X * self.y
        return 

    def compute_X(self):
        self.update_X()
        print('1')
        print(self.X[0])
        self.X[0] += 1
        print(self.X)
        return 

def try_class():
    cdef cnp.ndarray[cnp.float64_t, ndim=1] X = np.array([1.2,1.3], dtype=np.float64)
    cdef test t
    t = test(X)
    t.compute_X()
    print(t.X)
    print(np.sum(t.X))
    return