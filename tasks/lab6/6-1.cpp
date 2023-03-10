#include "6-1.hpp"

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

void Time1Approx (std::vector<std::vector<double>> &u, const std::function<double(double)> &f, double xh, double th, double a) {
    for (uint64_t i = 0; i < u[0].size(); ++i) {
        u[1][i] = u[0][i] + u[1][i] * th + a * derivative(f, xh * i, 2) * th * th / 2;
    }
}

void TIteration (std::vector<std::vector<double>> &u, const std::vector<std::vector<double>> &ux, double X0, double xh, double th, const std::vector<double> &coeff, const std::function<double(double, double)> &f, ApproxLevel left, ApproxLevel right, uint64_t i) {
    double a = coeff[0], b = coeff[1], c = coeff[2], d = coeff[3];
    double alpha[2] = {ux[0][0], ux[1][0]}, beta[2] = {ux[0][1], ux[1][1]};
    double coff;
    uint64_t n = u[0].size() - 2;
    uint64_t start = 0, end = n;
    if (left != ApproxLevel::NONE) {
        ++n;
        start = 2;
    } else {
        start = 1;
    }
    if (right != ApproxLevel::NONE) {
        ++n;
        end = n - 2;
    } else {
        end = n - 1;
    }
    Matrix<double> M(n, n);
    std::vector<double> ans(n, 0);
    if (left != ApproxLevel::NONE) {
        M(1, 0) = 2*a*th*th + b*th*th*xh;
        M(1, 1) = 2*th*th*xh*xh*c - 4*a*th*th - 4*b*th*th*xh - 2*xh*xh - 3*d*th*xh*xh;
        M(1, 2) = 2*a*th*th + 3*b*th*th*xh;
        ans[1] = u[i - 1][1] * (-4*xh*xh - 4*d*th*xh*xh) + u[i - 2][1] * (2*xh*xh + d*th*xh*xh) - 2*th*th*xh*xh*f(X0 + xh, i*th);
        switch (left) {
            case ApproxLevel::POINT2_ORDER1:
                M(0, 0) = beta[0]*xh - alpha[0];
                M(0, 1) = alpha[0];
                ans[0] = u[i][0] * xh;
                break;
            case ApproxLevel::POINT2_ORDER2:
                break;
            case ApproxLevel::POINT3_ORDER2:
                coff = -alpha[0] / M(1, 2);
                M(0, 0) = 2*beta[0]*xh - 3*alpha[0] - M(1, 0) * coff;
                M(0, 1) = 4*alpha[0] - M(1, 1) * coff;
                ans[0] = u[i][0] * 2*xh - ans[1] * coff;
                break;
            default:
                break;
        }
    } else {
        M(0, 0) = 2*th*th*xh*xh*c - 4*a*th*th - 4*b*th*th*xh - 2*xh*xh - 3*d*th*xh*xh;
        M(0, 1) = 2*a*th*th + 3*b*th*th*xh;
        ans[0] = u[i - 1][1] * (-4*xh*xh - 4*d*th*xh*xh) + u[i - 2][1] * (2*xh*xh + d*th*xh*xh) - 2*th*th*xh*xh*f(X0 + xh * 0, i*th) - u[i][0] * (2*a*th*th + b*th*th*xh);
    }

    for (uint64_t j = start; j < end; ++j) {
        M(j, j - 1) = 2*a*th*th + b*th*th*xh;
        M(j, j) = 2*th*th*xh*xh*c - 4*a*th*th - 4*b*th*th*xh - 2*xh*xh - 3*d*th*xh*xh;
        M(j, j + 1) = 2*a*th*th + 3*b*th*th*xh;
        ans[j] = u[i - 1][j + 1] * (-4*xh*xh - 4*d*th*xh*xh) + u[i - 2][j + 1] * (2*xh*xh + d*th*xh*xh) - 2*th*th*xh*xh*f(X0 + j*xh, i*th);
    }

    if (right != ApproxLevel::NONE) {
        M(n - 2, n - 3) = 2*a*th*th + b*th*th*xh;
        M(n - 2, n - 2) = 2*th*th*xh*xh*c - 4*a*th*th - 4*b*th*th*xh - 2*xh*xh - 3*d*th*xh*xh;
        M(n - 2, n - 1) = 2*a*th*th + 3*b*th*th*xh;
        ans[n - 2] = u[i - 1][n - 2] * (-4*xh*xh - 4*d*th*xh*xh) + u[i - 2][n - 2] * (2*xh*xh + d*th*xh*xh) - 2*th*th*xh*xh*f(X0 + (n - 2) * xh, i*th);
        switch (right) {
            case ApproxLevel::POINT2_ORDER1:
                M(n - 1, n - 2) = -alpha[1];
                M(n - 1, n - 1) = beta[1]*xh + alpha[1];
                ans[n - 1] = u[i][n - 1] * xh;
                break;
            case ApproxLevel::POINT2_ORDER2:
                break;
            case ApproxLevel::POINT3_ORDER2:
                coff = alpha[1] / M(1, 2);
                M(n - 1, n - 2) = -4*alpha[1] - M(n - 2, n - 2) * coff;
                M(n - 1, n - 1) = 2*beta[1]*xh + 3*alpha[1] - M(n - 2, n - 1) * coff;
                ans[n - 1] = u[i][n - 1] * 2*xh - ans[n - 2] * coff;
                break;
            default:
                break;
        }
    } else {
        M(n - 1, n - 2) = 2*a*th*th + b*th*th*xh;
        M(n - 1, n - 1) = 2*th*th*xh*xh*c - 4*a*th*th - 4*b*th*th*xh - 2*xh*xh - 3*d*th*xh*xh;
        ans[n - 1] = u[i - 1][n] * (-4*xh*xh - 4*d*th*xh*xh) + u[i - 2][n] * (2*xh*xh + d*th*xh*xh) - 2*th*th*xh*xh*f(X0 + n*xh, i*th) - u[i][n - 1] * (2*a*th*th + b*th*th*xh);
    }
    ans = RUNsolveSLAE(M, ans);
    for (uint64_t j = 0; j < ans.size(); ++j) {
        u[i][j + 1] = ans[j];
    }
}

void T (std::vector<std::vector<double>> &u, const std::vector<std::vector<double>> &ux, double X0, double xh, double th, const std::vector<double> &coeff, const std::function<double(double, double)> &f, ApproxLevel left, ApproxLevel right) {
    for (uint64_t i = 2; i < u.size(); ++i) {
        TIteration(u, ux, X0, xh, th, coeff, f, left, right, i);
    }
}

void CrossIteration (std::vector<std::vector<double>> &u, double X0, double xh, double th, const std::vector<double> &coeff, const std::function<double(double, double)> &f, uint64_t i) {
    double a = coeff[0], b = coeff[1], c = coeff[2], d = coeff[3];
    for (uint64_t j = 1; j < u[i].size() - 1; ++j) {
        double tmp = 0;
        tmp += u[i - 1][j + 1] * (2*a*th*th + 3*b*xh*th*th);
        tmp += u[i - 1][j] * (2*c*th*th*xh*xh - 4*a*th*th - 4*b*xh*th*th + 4*xh*xh + 4*d*th*xh*xh);
        tmp += u[i - 1][j - 1] * (2*a*th*th + b*xh*th*th);
        tmp += u[i - 2][j] * (-2*xh*xh - d*th*xh*xh);
        tmp += 2*th*th*xh*xh * f(X0 + j*xh, i*th);
        tmp /= (2*xh*xh + 3*d*th*xh*xh);
        u[i][j] = tmp;
    }
}

void Cross (std::vector<std::vector<double>> &u, const std::vector<std::vector<double>> &ux, double X0, double xh, double th, const std::vector<double> &coeff, const std::function<double(double, double)> &f, ApproxLevel left, ApproxLevel right) {

    for (uint64_t i = 2; i < u.size(); ++i) {
        --i;
        uint64_t start = 0, end = u[i].size() - 1;
        switch (left) {
            case ApproxLevel::POINT2_ORDER1:
                u[i][start] = Point2Order1(ux[0], xh, u[i][start + 1], u[i][start], 0);
                break;
            case ApproxLevel::POINT2_ORDER2:
                u[i][start] = Point2Order2(ux[0], coeff, xh, th, u[i - 1][start], u[i][start + 1], u[i][start], 0);
                break;
            case ApproxLevel::POINT3_ORDER2:
                u[i][start] = Point3Order2(ux[0], xh, u[i][start + 1], u[i][start + 2], u[i][start], 0);
                break;
            default:
                break;
        }
        switch (right) {
            case ApproxLevel::POINT2_ORDER1:
                u[i][end] = Point2Order1(ux[1], xh, u[i][end - 1], u[i][end], 0);
                break;
            case ApproxLevel::POINT2_ORDER2:
                u[i][end] = Point2Order2(ux[1], coeff, xh, th, u[i - 1][end], u[i][end - 1], u[i][end], 0);
                break;
            case ApproxLevel::POINT3_ORDER2:
                u[i][end] = Point3Order2(ux[1], xh, u[i][end - 1], u[i][end - 2], u[i][end], 0);
                break;
            default:
                break;
        }
        ++i;
        CrossIteration(u, X0, xh, th, coeff, f, i);
    }
}

std::vector<std::vector<double>> SolveIBVP (const Task &task, uint64_t xCount, uint64_t tCount, double Tmax, Method method, ApproxLevel approx) {
    double X0 = task.X[0], Xl = task.X[1];
    double th = Tmax / tCount, xh = (Xl - X0) / xCount;
    std::vector<std::vector<double>> u(tCount, std::vector<double>(xCount, 0));
    ApproxLevel left = ApproxLevel::NONE, right = ApproxLevel::NONE;
    if (task.ux[0][0] != 0) {
        left = approx;
    }
    if (task.ux[1][0] != 0) {
        right = approx;
    }
    auto f = [&] (double x, double t) -> double {
        return task.trees[0]({0, 0, 0, x, t});
    };
    auto fx0 = [&] (double t) -> double {
        return task.trees[1](t);
    };
    auto fxl = [&] (double t) -> double {
        return task.trees[2](t);
    };
    auto ft0 = [&] (double x) -> double {
        return task.trees[3](x);
    };
    auto ftt0 = [&] (double x) -> double {
        return task.trees[4](x);
    };
    for (uint64_t i = 0; i < u[0].size(); ++i) {
        u[0][i] = ft0(i * xh);
        u[1][i] = ftt0(i * xh);
    }
    Time1Approx(u, ft0, xh, th, task.coeff[0]);
    for (uint64_t i = 2; i < u.size(); ++i) {
        u[i].front() = fx0(i * th);
        u[i].back() = fxl(i * th);
    }
    switch (method) {
        case Method::EXPLICIT:
            Cross(u, task.ux, X0, xh, th, task.coeff, f, left, right);
            break;
        case Method::NOT_EXPLICIT:
            T(u, task.ux, X0, xh, th, task.coeff, f, left, right);
            break;
        default:
            break;
    }
    return u;
}