from setuptools import setup
from Cython.Build import cythonize
import numpy

"""
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()
directive_defaults['linetrace'] = True
directive_defaults['binding'] = True
"""

setup(
    ext_modules = cythonize("split_function.pyx", annotate=True, compiler_directives={'profile':True, 'linetrace':True}),
    include_dirs=[numpy.get_include()],
    define_macros=[('CYTHON_TRACE_NOGIL', '1')]
)
