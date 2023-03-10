#include "8-1.hpp"

int main () {
    std::cout.precision(PRECISION);
    std::cout.setf(std::ios_base::fixed);

    std::vector<std::string> system;
    std::string str;
    double xh, yh, th;
    Task task;
    FunctionalTree check;

    std::cout << "=====8.1=====\n";
    std::cout << "Введите начально-краевую задачу:\n";

    str = readLine();
    system.push_back(str);
    for (uint64_t i = 0; i < 5; ++i) {
        str = readLine();
        system.push_back(str);
    }
    for (uint64_t i = 0; i < system.size(); ++i) {
        std::cout << i << ": " << system[i] << "\n";
    }
    std::cout << "Введите размер шага для \"x\", \"y\" и \"t\":\n";
    std::cin >> xh >> yh >> th;
    std::cout << "Введите функцию для сравнения:\n";
    str = readLine();
    task = getTaskInfo(system);
    task.T[1] = 5;

    auto ans1 = SolveIBVP(task, xh, yh, th, Method::VARIABLE_DIRECTIONS);
    auto ans2 = SolveIBVP(task, xh, yh, th, Method::FRACTIONAL_STEPS);
    double mistakeSum1 = 0, mistakeSum2 = 0;
    double X0 = task.X[0], Y0 = task.Y[0], T0 = task.T[0];
    check.reset(str, {"x", "y", "t"});

    uint64_t xSize = uint64_t(task.X[1] / xh), ySize = uint64_t(task.Y[1] / yh), tSize = uint64_t(task.T[1] / th);
    uint64_t V = xSize * ySize * tSize; //вычисляем площадь таблицы U
    std::cout << "Размер таблицы функции U: " << xSize << "x" << ySize << "x" << tSize << "\n";
    std::cout << ans1.size() << " " << ans1[0].size() << " " << ans1[0][0].size() << "\n";
    for (uint64_t i = 0; i < ans1.size(); ++i) {
        for (uint64_t j = 0; j < ans1[i].size(); ++j) {
            for (uint64_t k = 0; k < ans1[i][j].size(); ++k) {
                mistakeSum1 += std::abs(ans1[i][j][k] - check({X0 + k * xh, Y0 + j * yh, T0 + i * th}));
                mistakeSum2 += std::abs(ans2[i][j][k] - check({X0 + k * xh, Y0 + j * yh, T0 + i * th}));
            }
        }
    }
    std::cout << "Средняя погрешность метода переменных направлений: " << mistakeSum1 / V << "\n";
    std::cout << "Средняя погрешность метода дробных шагов: " << mistakeSum2 / V << "\n";

    return 0;
}