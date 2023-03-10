#ifndef A_HPP
#define A_HPP

#include "General.hpp"
#include "SI.hpp"

enum class Method {
    YAKOBI,
    ZEIDEL,
    LIBMAN
};

enum SIDE {
    LEFT,
    RIGHT,
    DOWN,
    UP
};

enum class ApproxLevel {
    POINT2_ORDER1,
    POINT2_ORDER2,
    POINT3_ORDER2,
    NONE
};

std::vector<std::vector<double>> SolveIBVP (const Task &task, double xh, double yh, Method method);

#endif