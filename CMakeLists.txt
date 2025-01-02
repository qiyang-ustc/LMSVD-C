cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_COMPILER /opt/homebrew/opt/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++)

project(lmsvd VERSION 1.0 LANGUAGES CXX)

set(SOURCE_FILES src/main.cpp src/lmsvd.cpp)

add_executable(lmsvd ${SOURCE_FILES})

set(CMAKE_PREFIX_PATH "/opt/homebrew/opt")
target_include_directories(lmsvd PUBLIC "include")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O3 -march=native -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -march=native -flto -Wall")

find_package(OpenBLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(OpenBLAS_FOUND)
    message(STATUS "OpenBLAS found: ${OpenBLAS_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenBLAS not found")
endif()

if(LAPACK_FOUND)
    message(STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
else()
    message(FATAL_ERROR "LAPACK not found")
endif()

# ARPACK
set(ARPACK_DIR /opt/homebrew/opt/arpack)
set(ARPACK_INCLUDE_DIRS /opt/homebrew/opt/arpack/include)
set(ARPACK_LIBRARIES /opt/homebrew/opt/arpack/lib/libarpack.dylib)

target_include_directories(lmsvd PRIVATE ${ARPACK_INCLUDE_DIRS})
target_link_libraries(lmsvd PRIVATE ${ARPACK_LIBRARIES})
target_link_libraries(lmsvd PRIVATE ${OpenBLAS_LIBRARIES} ${LAPACK_LIBRARIES})