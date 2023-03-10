cmake_minimum_required(VERSION 3.8)

project(lab4 VERSION 1.0 LANGUAGES CXX)

set(MAIN main)
set(MAIN_SOURCE General.hpp General.cpp FuncMaker.hpp FuncMaker.cpp Matrix.hpp SI.hpp 8-1.hpp 8-1.cpp main.cpp)

add_executable(${MAIN} ${MAIN_SOURCE})

add_definitions(-std=c++14 -Wextra -Wall -Wno-sign-compare -Wno-unused-result -O3)