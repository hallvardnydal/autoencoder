from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
        name = "Rolling Ball",
        ext_modules = cythonize('rolling_ball.pyx'),
        include_dirs=[numpy.get_include()] 
)
