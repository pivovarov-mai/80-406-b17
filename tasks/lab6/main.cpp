#include "6-1.hpp"

int main () {
    std::cout.precision(PRECISION);
    std::cout.setf(std::ios_base::fixed);

    std::vector<std::string> system;
    std::string str;
    uint64_t xCount, tCount;
    Task task;
    FunctionalTree check;
    const double Tmax = 1.0;

    std::cout << "=====6.1=====\n";
    std::cout << "Введите начально-краевую задачу:\n";

    str = readLine();
    system.push_back(str);
    for (uint64_t i = 0; i < 4; ++i) {
        str = readLine();
        system.push_back(str);
    }
    for (uint64_t i = 0; i < system.size(); ++i) {
        std::cout << i << ": " << system[i] << "\n";
    }
    std::cout << "Введите количество шагов для \"x\" и для \"t\":\n";
    std::cin >> xCount >> tCount;
    std::cout << "Введите функцию для сравнения:\n";
    str = readLine();
    task = getTaskInfo(system);
    double X0 = task.X[0], Xl = task.X[1];
    double xh = (Xl - X0) / xCount, th = Tmax / tCount;
    
    std::cout << "Размер шага для x: " << xh << "\n" << "Размер шага для t: " << th << "\n";
    check.reset(str, {"x", "t"});

    uint64_t S = xCount * tCount; //вычисляем площадь таблицы U
    std::cout << "Размер таблицы функции U: " << xCount << "x" << tCount << "\n";

    for (uint64_t i = 0; i < 3; ++i) {
        ApproxLevel approx = static_cast<ApproxLevel>(i);
        switch (approx) {
            case ApproxLevel::POINT2_ORDER1:
                std::cout << "Point 2 Order 1\n";
                break;
            case ApproxLevel::POINT2_ORDER2:
                std::cout << "Point 2 Order 2\n";
                break;
            case ApproxLevel::POINT3_ORDER2:
                std::cout << "Point 3 Order 2\n";
                break;
            default:
                std::cout << "Error\n";
                break;
        }
        auto ans1 = SolveIBVP(task, xCount, tCount, Tmax, Method::EXPLICIT, approx);
        auto ans2 = SolveIBVP(task, xCount, tCount, Tmax, Method::NOT_EXPLICIT, approx);
        double mistakeSum1 = 0, mistakeSum2 = 0;

        for (uint64_t i = 0; i < ans1.size(); ++i) {
            for (uint64_t j = 0; j < ans1[i].size(); ++j) {
                mistakeSum1 += std::abs(ans1[i][j] - check({X0 + j * xh, i * th}));
                mistakeSum2 += std::abs(ans2[i][j] - check({X0 + j * xh, i * th}));
            }
        }
        std::cout << "Средняя погрешность явной схемы: " << mistakeSum1 / S << "\n";
        std::cout << "Средняя погрешность неявной схемы: " << mistakeSum2 / S << "\n";
    }

    return 0;
}