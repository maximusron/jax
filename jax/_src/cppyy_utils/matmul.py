import time
import cppyy, numba, warnings
import cppyy.numba_ext
import os
import numpy as np

inc_paths = [os.path.join(os.path.sep, 'usr', 'include'),
             os.path.join(os.path.sep, 'usr', 'local', 'include')]

eigen_path = None
for p in inc_paths:
    p = os.path.join(p, 'eigen3')
    if os.path.exists(p):
        eigen_path = p

cppyy.add_include_path(eigen_path)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    cppyy.include('Eigen/Dense')


cppyy.cppdef('''
    #pragma cling optimize(2)
    template<typename T>
    void MatrixProduct(T& m1,
                        T& m2,
                        T& result) {             
        if (m1.cols() != m2.rows()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
        }

        result = m1 * m2;
    }

        ''')

@numba.njit()
def mul_njit(m1, m2, m3):
    cppyy.gbl.MatrixProduct(m1, m2, m3)

cppyy.gbl.MatrixProduct(cppyy.gbl.Eigen.MatrixXd(), cppyy.gbl.Eigen.MatrixXd(), cppyy.gbl.Eigen.MatrixXd())

def eigen_mul(qy, db):

    shape_qy = qy.shape
    shape_db = db.shape

    qy = qy.flatten()
    db = db.flatten()

    dtype = type(qy[0])

    assert shape_qy[1] == shape_db[0] 
    
    mat1 = cppyy.gbl.Eigen.MatrixXd(shape_qy[0], shape_qy[1])
    mat2 = cppyy.gbl.Eigen.MatrixXd(shape_db[0], shape_db[1])
    result_njit = cppyy.gbl.Eigen.MatrixXd()

    c = mat1 << qy[0]
    d = mat2 << db[0]

    max_len = max(len(qy), len(db))

    t0 = time.time()
    for i in range(1, max_len):
        if i < len(qy):
            c = c.__comma__(qy[i])
        if i < len(db):
            d = d.__comma__(db[i])
    print("COMMA INIT TIME:", time.time() -t0)


    t0 = time.time()
    mul_njit(mat1, mat2, result_njit)
    print("MUL TIME:", time.time() -t0)

    t0 = time.time()
    a = np.zeros((result_njit.rows(), result_njit.cols()), dtype)
    for i in range(result_njit.rows()):
        for j in range(result_njit.cols()):
            a[i][j] = result_njit[i, j]
    print("CONVERSION TIME:", time.time() -t0)

    return a
    