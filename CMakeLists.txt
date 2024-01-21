cmake_minimum_required(VERSION 3.5)
project(libtensor VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(libtensor main.cpp tensor.cpp MathOperations.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
