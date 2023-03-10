#ifndef RUN_HPP
#define RUN_HPP

#include "General.hpp"
#include "Matrix.hpp"

template <class T>
bool checkIf3Diagonal (const Matrix<T> &matrix) {
    //std::cout << "===CHECKING IF NOT 3-DIAGONAL===\n";
    uint64_t n = matrix.size().n;
    T null = T(0);
    bool result = true;
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t left, right;
        left = i - 1 > n ? 0 : i - 1;
        right = i + 2 > n ? n : i + 2;
        for (uint64_t j = 0; j < left; ++j) {
            if (!isEqual(matrix(i, j), null)) {
                result = false;
                break;
            }
        }
        for (uint64_t j = left; j < right; ++j) {
            if (isEqual(matrix(i, j), null)) {
                result = false;
                break;
            }
        }
        for (uint64_t j = right; j < n; ++j) {
            if (!isEqual(matrix(i, j), null)) {
                result = false;
                break;
            }
        }
    }
    //std::cout << (result ? "OK\n" : "Not OK.\n");
    //std::cout << "==============DONE==============\n";
    return result;
}

template <class T>
std::vector<T> RUNsolveSLAE (const Matrix<T> &matrix, const std::vector<T> &ans) {
    uint64_t n = matrix.size().n;
    std::vector<T> P(n), Q(n), x(n);

    //std::cout << "Ax = b\n\nMatrix A:\n" << matrix << "\n";
    //printVector("b", ans);
    //std::cout << "\n";

    if (!matrix.isSquare() || n != ans.size()) {
        //std::cout << "Matrix is not square. Stop working.\n";
        return {};
    }
    if (n != ans.size()) {
        //std::cout << "Matrix and vector have different sizes. Stop working.\n";
        return {};
    }
    if (!checkIf3Diagonal(matrix)) {
        //std::cout << "Matrix is not 3-diagonal. Stop working.\n";
        return {};
    }

    T a = 0, b = matrix(0, 0), c = matrix(0, 1), d = ans[0];
    P[0] = -c / b;
    Q[0] = d / b;
    //std::cout << "------------\n";
    //std::cout << "Iteration 0:\nP[0] = -c[0] / b[0] = " << P[0] << "\nQ[0] = d[0] / b[0] = " << Q[0] << "\n";
    for (uint64_t i = 1; i < n; ++i) {
        a = matrix(i, i - 1);
        b = matrix(i, i);
        c = i + 1 < n ? matrix(i, i + 1) : 0;
        d = ans[i];
        P[i] = -c / (b + a * P[i - 1]);
        Q[i] = (d - a * Q[i - 1]) / (b + a * P[i - 1]);
        //std::cout << "------------\n";
        //std::cout << "Iteration " << i << ":\nP[" << i << "] = -c[" << i << "] / (b[" << i << "] + a[" << i << "] * P[" << i - 1 << "]) = " << P[i] << "\n"; 
        //std::cout << "Q[" << i << "] = (d[" << i << "] - a[" << i << "] * Q[" << i - 1 << "]) / (b[" << i << "] + a[" << i << "] * P[" << i - 1 << "]) = " << Q[i] << "\n";
    }
    x[n - 1] = Q[n - 1];
    //std::cout << "Solving x:\nx[" << n - 1 << "] = " << Q[n - 1] << "\n";
    for (uint64_t i = n - 2; i < n; --i) {
        x[i] = P[i] * x[i + 1] + Q[i];
        //std::cout << "x[" << i << "] = P[" << i << "] * x[" << i + 1 << "] + Q[" << i << "] = " << x[i] << "\n";
    }
    //std::cout << "\n"; 
    //printVector("x", x);
    //std::cout << "\n";
    return x;
}

#endif