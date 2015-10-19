
# LMSVD

## 1. File Descrpition

*data*: Test data, a 2000*2000 dense matrix

### MATLAB

*lmsvd.m*: A MATLAB implementation of LMSVD 

*demo_lmsvd.m*: A test program for lmsvd.m

### C++

*include*: Armadillo-6.100.0

*CMakeLists.txt*: cmake project file

*common.h*: Header files, definitions of structures and constant values

*lmsvd.h, lmvd.cpp*: A C++ implementation of LMSVD

*main.cpp*: A test program for LMSVD


## 2. Configuration

### blas + lapack

* Install blas and lapack libraries

* For CMakeLists.txt, uncomment "target_link_libraries(main blas lapack)", comment "target_link_libraries(main openblas)", "target_link_libraries(main openblas_sandybridgep-r0.2.14)" and "export OPENBLAS_NUM_THREADS=4".

### openblas

* Install openblas

* For CMakeLists.txt, comment "target_link_libraries(main blas lapack)", uncomment "target_link_libraries(main openblas)"

* Uncomment and change "export OPENBLAS_NUM_THREADS=4" to use different numbers of threads to run the program. 

* Library files for "target_link_libraries(main openblas_sandybridgep-r0.2.14)" is generated on the platform of sandy bridge. This command will be different in other platforms, and can replace the command "target_link_libraries(main openblas)". The details of these optional files has not been investigated.


## 3. Usage

1. Install the libraries needed (such as blas+lapack or openblas)

2. Change CMakeLists.txt according to the libraries used 

3. Use the command "cmake ." and "make" to build the project 

4. Run "./main" to conduct the test program
