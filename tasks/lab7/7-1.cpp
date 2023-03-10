#include "7-1.hpp"

double Point2Order1 (const std::vector<double> &coeff, double h, double u1, double f, uint64_t i) {
    double alpha = coeff[0], beta = coeff[1];
    double ans = 0;
    if (i == 0) {
        ans += f - alpha * u1 / h;
        ans /= beta - alpha / h;
    } else {
        ans += f + alpha * u1 / h;
        ans /= beta + alpha / h;
    }
    return ans;
}

double Point2Order2 (const std::vector<double> &ux, const std::vector<double> &coeff, double h, double t, double u0, double u1, double f, uint64_t i) {
    double alpha = ux[0], beta = ux[1];
    double a = coeff[0], b = coeff[1], c = coeff[2];
    double ans = 0;
    if (i == 0) {
        ans += h / t * u0 - f * (2 * a - b * h) / alpha + 2 * u1 * a / h;
        ans /= 2 * a / h + h / t - c * h - (beta / alpha) * (2 * a - b * h);
    } else {
        ans += h / t * u0 + f * (2 * a + b * h) / alpha + 2 * u1 * a / h;
        ans /= 2 * a / h + h / t - c * h + (beta / alpha) * (2 * a + b * h);
    }
    return ans;
}

double Point3Order2 (const std::vector<double> &coeff, double h, double u1, double u2, double f, uint64_t i) {
    double alpha = coeff[0], beta = coeff[1];
    double ans = 0;
    if (i == 0) {
        ans += f - alpha * (4 * u1 - u2) / (2 * h);
        ans /= beta - 3 * alpha / (2 * h);
    } else {
        ans += f + alpha * (4 * u1 - u2) / (2 * h);
        ans /= beta + 3 * alpha / (2 * h);
    }
    return ans;
}

void borderApprox (std::vector<std::vector<double>> &u, const std::vector<std::vector<double>> &ux, const std::vector<double> &coeff, double xh, double yh, const std::vector<ApproxLevel> &border) {
        uint64_t left = 0, right = u[0].size() - 1, down = 0, up = u.size() - 1;
        for (uint64_t j = 1; j < u.size(); ++j) {
            switch (border[0]) {
                case ApproxLevel::POINT2_ORDER1:
                    u[j][left] = Point2Order1(ux[0], xh, u[j][left + 1], u[j][left], 0);
                    break;
                case ApproxLevel::POINT2_ORDER2:
                    u[j][left] = Point2Order2(ux[0], coeff, xh, yh, u[j - 1][left], u[j][left + 1], u[j][left], 0);
                    break;
                case ApproxLevel::POINT3_ORDER2:
                    u[j][left] = Point3Order2(ux[0], xh, u[j][left + 1], u[j][left + 2], u[j][left], 0);
                    break;
                default:
                    break;
            }
        }
        for (uint64_t j = 1; j < u.size(); ++j) {
            switch (border[1]) {
                case ApproxLevel::POINT2_ORDER1:
                    u[j][right] = Point2Order1(ux[1], xh, u[j][right - 1], u[j][right], 0);
                    break;
                case ApproxLevel::POINT2_ORDER2:
                    u[j][right] = Point2Order2(ux[1], coeff, xh, yh, u[j - 1][right], u[j][right - 1], u[j][right], 0);
                    break;
                case ApproxLevel::POINT3_ORDER2:
                    u[j][right] = Point3Order2(ux[1], xh, u[j][right - 1], u[j][right - 2], u[j][right], 0);
                    break;
                default:
                    break;
            }
        }
        for (uint64_t j = 1; j < u[0].size(); ++j) {
            switch (border[2]) {
                case ApproxLevel::POINT2_ORDER1:
                    u[down][j] = Point2Order1(ux[2], xh, u[down + 1][j], u[down][j], 0);
                    break;
                case ApproxLevel::POINT2_ORDER2:
                    u[down][j] = Point2Order2(ux[2], coeff, xh, yh, u[down][j - 1], u[down + 1][j], u[down][j], 0);
                    break;
                case ApproxLevel::POINT3_ORDER2:
                    u[down][j] = Point3Order2(ux[2], xh, u[down + 1][j], u[down + 2][j], u[down][j], 0);
                    break;
                default:
                    break;
            }
        }
        for (uint64_t j = 1; j < u[0].size(); ++j) {
            switch (border[3]) {
                case ApproxLevel::POINT2_ORDER1:
                    u[up][j] = Point2Order1(ux[3], xh, u[up - 1][j], u[up][j], 0);
                    break;
                case ApproxLevel::POINT2_ORDER2:
                    u[up][j] = Point2Order2(ux[3], coeff, xh, yh, u[up][j - 1], u[up - 1][j], u[up][j], 0);
                    break;
                case ApproxLevel::POINT3_ORDER2:
                    u[up][j] = Point3Order2(ux[3], xh, u[up - 1][j], u[up - 2][j], u[up][j], 0);
                    break;
                default:
                    break;
            }
        }
}

double norma (const std::vector<std::vector<double>> &v1, const std::vector<std::vector<double>> &v2) {
    double ans = 0;
    uint64_t size1 = std::min(v1.size(), v2.size());
    uint64_t size2 = std::min(v1[0].size(), v2[0].size());
    for (uint64_t i = 0; i < size1; ++i) {
        for (uint64_t j = 0; j < size2; ++j) {
            ans = std::max(ans, std::abs(v1[i][j] - v2[i][j]));
        }
    }
    return ans;
}

void UInit (std::vector<std::vector<double>> &u) {
    uint64_t height = u.size(), width = u[0].size();
    for (uint64_t i = 1; i < height - 1; ++i) {
        double left = u[i].front(), right = u[i].back();
        double step = (right - left) / width;
        for (uint64_t j = 1; j < width - 1; ++j) {
            u[i][j] = left + step * j;
        }
    }
}

void SIIteration (std::vector<std::vector<double>> &u, double xh, double yh, const std::vector<double> &coeff, const std::function<double(double, double)> &f, double eps, SIMethod method) {
    uint64_t n = u[0].size() - 2;
    Matrix<double> matrix(n, n);
    std::vector<double> ans(n);
    double a = coeff[0], b = coeff[1], c = coeff[2];
        for (uint64_t i = 1; i < u.size() - 1; ++i) {

            matrix(0, 0) = a*xh*yh*yh + b*xh*xh*yh - c*xh*xh*yh*yh - 2*yh*yh - 2*xh*xh;
            matrix(0, 1) = yh*yh - a*xh*yh*yh;
            ans[0] = -yh*yh*u[i][0] + (b*xh*xh*yh - xh*xh)*u[i + 1][1] - xh*xh*u[i - 1][1] + xh*xh*yh*yh*f(0, i*yh);
            for (uint64_t j = 1; j < n - 1; ++j) {
                matrix(j, j - 1) = yh*yh;
                matrix(j, j) = a*xh*yh*yh + b*xh*xh*yh - c*xh*xh*yh*yh - 2*yh*yh - 2*xh*xh;
                matrix(j, j + 1) = yh*yh - a*xh*yh*yh;
                ans[j] = (b*xh*xh*yh - xh*xh)*u[i + 1][j + 1] - xh*xh*u[i - 1][j + 1] + xh*xh*yh*yh*f(j*xh, i*yh);
            }
            matrix(n - 1, n - 2) = yh*yh;
            matrix(n - 1, n - 1) = a*xh*yh*yh + b*xh*xh*yh - c*xh*xh*yh*yh - 2*yh*yh - 2*xh*xh;
            ans[n - 1] = -(yh*yh - a*xh*yh*yh)*u[i][n + 1] + (b*xh*xh*yh - xh*xh)*u[i + 1][n] - xh*xh*u[i - 1][n] + xh*xh*yh*yh*f(n*xh, i*yh);
            ans = SISolveSLAE(matrix, ans, eps, method);
            for (uint64_t j = 0; j < ans.size(); ++j) {
                u[i][j + 1] = ans[j];
            }
        }
}

void SI (std::vector<std::vector<double>> &u, const std::vector<std::vector<double>> &ux, const std::vector<double> &coeff, double xh, double yh, const std::function<double(double, double)> &f, const std::vector<ApproxLevel> &border, double eps, SIMethod method) {
    UInit(u);
    SIIteration(u, xh, yh, coeff, f, eps, method);
    borderApprox(u, ux, coeff, xh, yh, border);
    auto prev = u;
    do {
        prev = u;
        SIIteration(u, xh, yh, coeff, f, eps, method);
        borderApprox(u, ux, coeff, xh, yh, border);
    } while (norma(u, prev) > eps);
}

double LibmanIteration (std::vector<std::vector<double>> &u, double xh, double yh, const std::vector<double> &coeff, const std::function<double(double, double)> &f) {
    double maxAbs = 0;
    double a = coeff[0], b = coeff[1], c = coeff[2];
    uint64_t height = u.size(), width = u[0].size();
    for (uint64_t i = 1; i < height - 1; ++i) {
        for (uint64_t j = 1; j < width - 1; ++j) {
            double currU = 0;
            currU += u[i + 1][j] * (a*xh*yh*yh - yh*yh);
            currU -= u[i - 1][j] * yh*yh;
            currU += u[i][j + 1] * (b*xh*xh*yh - xh*xh);
            currU -= u[i][j - 1] * xh*xh;
            currU += xh*xh*yh*yh*f(i*xh, j*yh);
            currU /= a*xh*yh*yh + b*xh*xh*yh - c*xh*xh*yh*yh - 2*yh*yh - 2*xh*xh;
            maxAbs = std::max(maxAbs, std::abs(u[i][j] - currU));
            u[i][j] = currU;
        }
    }
    return maxAbs;
}

void Libman (std::vector<std::vector<double>> &u, const std::vector<std::vector<double>> &ux, const std::vector<double> &coeff, double xh, double yh, const std::function<double(double, double)> &f, const std::vector<ApproxLevel> &border, double eps) {
    UInit(u);
    LibmanIteration(u, xh, yh, coeff, f);
    borderApprox(u, ux, coeff, xh, yh, border);
    auto prev = u;
    do {
        prev = u;
        LibmanIteration(u, xh, yh, coeff, f);
        borderApprox(u, ux, coeff, xh, yh, border);
    } while (norma(u, prev) > eps);
}

std::vector<std::vector<double>> SolveIBVP (const Task &task, double xh, double yh, Method method) {
    double l1 = task.X[1], l2 = task.Y[1];
    std::vector<std::vector<double>> u(uint64_t(l2 / yh), std::vector<double>(uint64_t(l1 / xh), 0));
    std::vector<ApproxLevel> border(4, ApproxLevel::NONE);
    for (uint64_t i = 0; i < 4; ++i) {
        if (task.ux[i][0] != 0) {
            border[i] = ApproxLevel::POINT2_ORDER2;
        }
    }
    auto f = [&] (double x, double y) -> double {
        return task.trees[0]({0, 0, 0, x, y});
    };
    auto fx0 = [&] (double y) -> double {
        return task.trees[1](y);
    };
    auto fxl = [&] (double y) -> double {
        return task.trees[2](y);
    };
    auto fy0 = [&] (double x) -> double {
        return task.trees[3](x);
    };
    auto fyl = [&] (double x) -> double {
        return task.trees[4](x);
    };
    for (uint64_t i = 0; i < u[0].size(); ++i) {
        u.front()[i] = fy0(i * xh);
        u.back()[i] = fyl(i * xh);
    }
    for (uint64_t i = 1; i < u.size() - 1; ++i) {
        u[i].front() = fx0(i * yh);
        u[i].back() = fxl(i * yh);
    }
    switch (method) {
        case Method::YAKOBI:
            SI(u, task.ux, task.coeff, xh, yh, f, border, 0.01, SIMethod::SI_YAKOBI_METHOD);
            break;
        case Method::ZEIDEL:
            SI(u, task.ux, task.coeff, xh, yh, f, border, 0.01, SIMethod::SI_ZEIDEL_METHOD);
            break;
        case Method::LIBMAN:
            Libman(u, task.ux, task.coeff, xh, yh, f, border, 0.01);
            break;
        default:
            break;
    }
    return u;
}