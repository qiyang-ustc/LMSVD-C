set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)

cmake_minimum_required(VERSION 3.29)
project(lmsvd)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
include_directories("include/")
set(SOURCE_FILES main.cpp lmsvd.cpp)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(CMAKE_PREFIX_PATH "/usr/local/opt")
set(CMAKE_INSTALL_PREFIX "/usr/local/opt")

add_executable(main ${SOURCE_FILES})

set(OpenBLAS_INCLUDE_DIR "/usr/local/opt/openblas/include")
set(OpenBLAS_LIB_DIR "/usr/local/opt/openblas/lib")
include_directories(${OpenBLAS_INCLUDE_DIR})

set(LAPACK_INCLUDE_DIR "/usr/local/opt/lapack/include")
set(LAPACK_LIB_DIR "/usr/local/opt/lapack/lib")
include_directories(${LAPACK_INCLUDE_DIR})

#find_package(openblas REQUIRED)
#find_package(lapack REQUIRED)

#target_link_libraries(main blas lapack)
target_link_libraries(main ${OpenBLAS_LIB_DIR}/libopenblas.dylib ${LAPACK_LIB_DIR}/liblapack.dylib)
#target_link_libraries(main openblas)

#export OPENBLAS_NUM_THREADS=6

