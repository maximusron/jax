#include <vector>
#include <omp.h>
#include <thread>

class MatrixDot {
private:
    const std::vector<double>* matrix1;
    const std::vector<double>* matrix2;

public:
    std::vector<double>* result;
    MatrixDot(const std::vector<double>* m1, const std::vector<double>* m2, std::vector<double>* res)
        : matrix1(m1), matrix2(m2), result(res) {}

    void multiplyMatrices(const std::vector<int>& shape1, const std::vector<int>& shape2) {
        int rows1 = shape1[0];
        int cols1 = shape1[1];
        int rows2 = shape2[0];
        int cols2 = shape2[1];

        #pragma omp parallel for
        for (int i = 0; i < rows1; ++i) {
            for (int j = 0; j < cols2; ++j) {
                double sum = 0.0;
                for (int k = 0; k < cols1; ++k) {
                    sum += (*matrix1)[i * cols1 + k] * (*matrix2)[k * cols2 + j];
                }
                (*result)[i * cols2 + j] = sum;
            }
        }
    }
};
