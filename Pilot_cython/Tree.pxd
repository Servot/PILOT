cimport numpy as cnp
import numpy as np

cdef class tree:
    
    cdef public tree left
    cdef public tree right
    cdef public double Rt
    cdef public str node
    cdef public int feature
    cdef public double pivot
    cdef public cnp.ndarray lm_l # a cython type rather than a typed memoryview
    cdef public cnp.ndarray lm_r
    cdef public int depth
    cdef public cnp.ndarray interval
    cdef public cnp.ndarray pivot_c