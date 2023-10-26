import time
import cppyy, numba, warnings
import cppyy.numba_ext
import os
import numpy as np
from cppyy.gbl.std import vector

cppyy.load_library("/usr/lib/x86_64-linux-gnu/libiomp5.so")

cppyy.include('/home/maximus/cern/jax/jax/_src/cppyy_utils/matmul.cpp')

# @numba.njit()
def mul_njit(d, shape1, shape2):
    d.multiplyMatrices(shape1, shape2)
    
   
def std_vecmul(qy, db):
    shape_qy = vector[int](qy.shape)
    shape_db = vector[int](db.shape)
    
    res = vector['double'](np.zeros(shape_qy[0]*shape_db[1]))
    qy = vector['double'](qy.flatten())
    db = vector['double'](db.flatten())
    d = cppyy.gbl.MatrixDot(qy, db, res)
    
    # t0 = time.time()
    # mul_njit(d, shape1=shape_qy, shape2=shape_db)
    
    t0 = time.time()
    mul_njit(d, shape1=shape_qy, shape2=shape_db)
    print("MUL TIME:", time.time() -t0)
    return np.array(d.result).reshape(shape_qy[0], shape_db[1])

# qy_shape=(200, 128)
# db_shape=(128, 30000)

# qy_test = np.arange(200 * 128, dtype=np.float64).reshape(200, 128)
# db_test = np.arange(128 * 500, dtype = np.float64).reshape(128, 500)

# std_vecmul(qy_test, db_test)