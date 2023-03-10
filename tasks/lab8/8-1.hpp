#ifndef A_HPP
#define A_HPP

#include "General.hpp"
#include "RUN.hpp"

using Table = std::vector<std::vector<std::vector<double>>>;

enum class Method {
    VARIABLE_DIRECTIONS,
    FRACTIONAL_STEPS
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

Table SolveIBVP (const Task &task, double xh, double yh, double th, Method method);

#endif