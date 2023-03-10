#include "General.hpp"

void printVector(const std::vector<double> &vec) {
    for (double el : vec) {
        std::cout << el << " ";
    }
    std::cout << "\n";
}

double findEpsillon () {
    return std::pow(__DBL_EPSILON__, 1.0 / 3) * 10;
}

bool isEqual(double x, double y) {
    return std::fabs(x - y) < std::numeric_limits<double>::epsilon();
}

double derivative (const std::function<double(double)> &f, double x, uint64_t degree) {
    static double eps = findEpsillon();
    if (degree == 1) {
        return (f(x + eps) - f(x - eps)) / (2 * eps);
    }
    return (derivative(f, x + eps, degree - 1) - derivative(f, x - eps, degree - 1)) / (2 * eps);
}

std::string toString (double val, uint64_t precision) {
    return std::to_string(val).substr(0, std::to_string(val).find(".") + precision + 1);
}

double stringFix (std::string &str) {
    double val = 0;
    std::string tmp;
    std::swap(tmp, str);
    uint64_t count = 0;
    for (uint64_t i = 0; i < tmp.size(); ++i) {
        if (tmp[i] == 'u') {
            std::string valStr;
            while (tmp[i] != '(') {
                str += tmp[i];
                ++i;
            }
            ++count;
            ++i;
            while (1) {
                if (tmp[i] == ')') {
                    --count;
                }
                if (tmp[i] == '(') {
                    ++count;
                }
                if (count == 0 || i >= tmp.size()) {
                    break;
                }
                valStr += tmp[i];
                ++i;
            }
            val = FunctionalTree(valStr.c_str()).calculate();
        } else {
            str += tmp[i];
        }
    }
    return val;
}

Task getTaskInfo(const std::vector<std::string> &system) {
    uint64_t idx = 0, size = 0;
    std::string tmp;
    Task task;

    task.X.resize(2);
    task.Y.resize(2);
    task.T.resize(2);
    task.coeff = std::vector<double>(5, 0);

    idx = system[0].find('=');
    size = system[0].size();
    task.trees.push_back(std::move(FunctionalTree(system[0].substr(idx + 1, size - idx), {"uxx", "ux", "uyy", "uy", "u", "x", "y"})));
    for (uint64_t i = 0; i < task.coeff.size() - 1; ++i) {
        task.coeff[i] = task.trees[0].getCoeff(i).calculate();
    }
    std::vector<std::string> vars = {"y", "t"};
    for (uint64_t i = 1; i < 3; ++i) {
        FunctionalTree tmpTree;
        idx = system[i].find('=');
        size = system[i].size();
        tmp = system[i].substr(0, idx);
        task.X[i - 1] = stringFix(tmp);

        tmpTree.reset(tmp, {"ux", "u"});
        task.ux.push_back({tmpTree.getCoeff(0).calculate(), tmpTree.getCoeff(1).calculate()});
        task.trees.push_back(std::move(FunctionalTree(system[i].substr(idx + 1, size - idx), vars)));
    }
    vars[0] = "x";
    for (uint64_t i = 3; i < 5; ++i) {
        FunctionalTree tmpTree;
        idx = system[i].find('=');
        size = system[i].size();
        tmp = system[i].substr(0, idx);
        task.Y[i - 3] = stringFix(tmp);

        tmpTree.reset(tmp, {"uy", "u"});
        task.ux.push_back({tmpTree.getCoeff(0).calculate(), tmpTree.getCoeff(1).calculate()});
        task.trees.push_back(std::move(FunctionalTree(system[i].substr(idx + 1, size - idx), vars)));
    }
    vars[1] = "y";
    for (uint64_t i = 5; i < 6; ++i) {
        FunctionalTree tmpTree;
        idx = system[i].find('=');
        size = system[i].size();
        tmp = system[i].substr(0, idx);
        task.T[i - 5] = stringFix(tmp);

        tmpTree.reset(tmp, {"ut", "u"});
        task.ux.push_back({tmpTree.getCoeff(0).calculate(), tmpTree.getCoeff(1).calculate()});
        task.trees.push_back(std::move(FunctionalTree(system[i].substr(idx + 1, size - idx), vars)));
    }
    return task;
}

std::string readLine () {
    std::string str;
    while (str.empty()) {
        std::getline(std::cin, str);
        if (!str.empty() && str[0] == '#') {
            str = "";
        }
    }
    return str;
}