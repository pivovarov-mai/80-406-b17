#ifndef SIMPLE_ITERATION_HPP
#define SIMPLE_ITERATION_HPP

#include "General.hpp"

enum class SIMethod {
    SI_YAKOBI_METHOD,
    SI_ZEIDEL_METHOD
};

template <class T>
T SINorma (const Matrix<T> &matrix) {
    uint64_t n = matrix.size().n;
    T max = 0;
    for (uint64_t i = 0; i < n; ++i) {
        T tmp = 0;
        for (uint64_t j = 0; j < n; ++j) {
            tmp += std::abs(matrix(i, j));
        }
        if (tmp > max) {
            max = tmp;
        }
    }
    return max;
}

template <class T>
T FindEpsilon (const std::vector<T> &oldX, const std::vector<T> &newX, T a) {
    T eps = 0;
    for (uint64_t i = 0; i < oldX.size(); ++i) {
        T tmp = std::abs(oldX[i] - newX[i]);
        if (tmp > eps) {
            eps = tmp;
        }
    }
    return a < T(1) ? eps * a / (T(1) - a) : eps;
}

template <class T>
std::vector<double> SISolveSLAE (const Matrix<T> &matrix, const std::vector<T> &b, T approx, SIMethod method) {
    uint64_t n = matrix.size().n;
    std::vector<T> beta(n), x(n);
    Matrix<T> alpha(n, n);

    if (!matrix.isSquare() || n != b.size()) {
        return;
    }
    if (n != b.size()) {
        return;
    }

    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < i; ++j) {
            alpha(i, j) = -matrix(i, j) / matrix(i, i);
        }
        for (uint64_t j = i + 1; j < n; ++j) {
            alpha(i, j) = -matrix(i, j) / matrix(i, i);
        }
        beta[i] = b[i] / matrix(i, i);
        x[i] = beta[i];
    }
    T a = SINorma(alpha);
    uint64_t iteration = 0;
    T epsilon = T(0);

    while (1) {
        ++iteration;
        std::vector<T> newX(x);
        for (uint64_t i = 0; i < n; ++i) {
            newX[i] = beta[i];
            if (method == SIMethod::SI_ZEIDEL_METHOD) {
                for (uint64_t j = 0; j < i; ++j) {
                    newX[i] += alpha(i, j) * newX[j];
                }
                for (uint64_t j = i; j < n; ++j) {
                    newX[i] += alpha(i, j) * x[j];
                }
            } else {
                for (uint64_t j = 0; j < n; ++j) {
                    newX[i] += alpha(i, j) * x[j];
                }
            }
        }
        epsilon = FindEpsilon(x, newX, a);
        x = newX;
        if (epsilon < approx || iteration > ITERATION_CAP) {
            break;
        }
    }
    return x;
}

#endif