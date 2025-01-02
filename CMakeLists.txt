cmake_minimum_required(VERSION 3.11)

# 设置项目名称和版本
project(lmsvd VERSION 1.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译器（使用 gnu12 编译器）
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "gcc")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "g++")
endif()

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# 设置源文件
set(SOURCE_FILES src/main.cpp src/lmsvd.cpp)

# 添加可执行文件
add_executable(main ${SOURCE_FILES})

# 添加头文件目录
target_include_directories(main PUBLIC "include")

# 查找 OpenBLAS 和 LAPACK 库
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# 检查 OpenBLAS 是否找到
if(BLAS_FOUND AND LAPACK_FOUND)
    message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
    message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
    target_link_libraries(main ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
    message(FATAL_ERROR "BLAS or LAPACK not found. Please load the appropriate modules.")
endif()

# 打印包含目录
get_property(INCLUDE_DIRS TARGET main PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "Include directories: ${INCLUDE_DIRS}")

# 设置环境变量（可选）
# set(ENV{OPENBLAS_NUM_THREADS} 6)

