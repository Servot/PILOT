from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["Tree.pyx", "split_function.pyx", "Pilot.pyx"], annotate=True),
    include_dirs=[numpy.get_include()]
)