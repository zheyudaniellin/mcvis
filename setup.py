# we need this file to compile the cython code
# based on the setup.py from pdspy
# python3 setup.py build_ext --inplace

from setuptools import setup
from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize

libinterferometry = cythonize([\
        Extension('mcvis.interferometry.libinterferometry',\
            ["mcvis/interferometry/libinterferometry.pyx"],\
            libraries=["m"], extra_compile_args=['-ffast-math'])])[0]

# now define the setup
setup(
    packages=[
        "mcvis", 
        "mcvis.interferometry", 
        "mcvis.parameters"],
    ext_modules=[libinterferometry],
)

