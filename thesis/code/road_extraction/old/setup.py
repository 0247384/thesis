from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["road_extraction/cost_c.pyx", "road_extraction/search_c.pyx"]),
    include_dirs = [numpy.get_include()]
)