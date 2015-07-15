from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Collaborative filtering experiment",
    ext_modules = cythonize('cofi.pyx'),  # accepts a glob pattern
)
