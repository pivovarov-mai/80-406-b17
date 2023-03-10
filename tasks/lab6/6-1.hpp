#ifndef A_HPP
#define A_HPP

#include "General.hpp"
#include "RUN.hpp"

enum class Method {
    EXPLICIT,
    NOT_EXPLICIT
};

enum class SIDE {
    LEFT,
    RIGHT
};

enum class ApproxLevel {
    POINT2_ORDER1,
    POINT2_ORDER2,
    POINT3_ORDER2,
    NONE
};

std::vector<std::vector<double>> SolveIBVP (const Task &task, uint64_t xCount, uint64_t tCount, double Tmax, Method method, ApproxLevel approx);

#endif