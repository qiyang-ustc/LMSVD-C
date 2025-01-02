cmake_minimum_required(VERSION 3.29)
set(CMAKE_C_COMPILER /opt/homebrew/opt/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(SOURCE_FILES src/main.cpp src/lmsvd.cpp)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O3 -march=native -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -march=native -Wall")

project(lmsvd)

add_executable(main ${SOURCE_FILES})

target_include_directories(main PUBLIC "include")
get_property(INCLUDE_DIRS TARGET main PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "Include directories: ${INCLUDE_DIRS}")

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

