import time
import cppyy, numba, warnings
import cppyy.numba_ext
import os
import numpy as np
from cppyy.gbl.std import vector

cppyy.load_library("/usr/lib/x86_64-linux-gnu/libiomp5.so")

cppyy.include('/home/maximus/cern/jax/jax/_src/cppyy_utils/omp_demo.cpp')

cppyy.gbl.ompdemo()