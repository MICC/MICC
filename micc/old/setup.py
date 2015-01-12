"""
This is NOT for use to build releases, but rather for compiling via Cython

Usage:
    python setup.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='MICC',
    ext_modules=cythonize(["graph.py"])
)

