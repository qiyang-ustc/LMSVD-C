cmake_minimum_required(VERSION 2.8)
project(lmsvd)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories("include/")
set(SOURCE_FILES main.cpp lmsvd.cpp)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


add_executable(main ${SOURCE_FILES})

#link_directories("/opt/OpenBLAS/lib/")
#target_link_libraries(main blas lapack)
target_link_libraries(main openblas)
#target_link_libraries(main openblas_sandybridgep-r0.2.14)
#export OPENBLAS_NUM_THREADS=4
