cmake_minimum_required(VERSION 3.5)
project(libtensor VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

# 添加 xtl 和 xtensor 相关配置
include(FetchContent)

FetchContent_Declare(
        xtl
        GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
        GIT_TAG 0.7.7
)

FetchContent_Declare(
        xtensor
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
        GIT_TAG 0.24.7
)

FetchContent_MakeAvailable(xtl)
FetchContent_MakeAvailable(xtensor)


# 设置源文件
add_executable(libtensor tensor.cpp MathOperations.cpp main.cpp)

# 添加 xtensor 和 xtl 的头文件目录
target_include_directories(libtensor PRIVATE ${xtl_SOURCE_DIR}/include)
target_include_directories(libtensor PRIVATE ${xtl_BINARY_DIR}/include)
target_include_directories(libtensor PRIVATE ${xtensor_SOURCE_DIR}/include)
target_include_directories(libtensor PRIVATE ${xtensor_BINARY_DIR}/include)

# 链接 xtensor 和 xtl 库
target_link_libraries(libtensor PRIVATE xtl xtensor)

# 设置其他项目信息
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
