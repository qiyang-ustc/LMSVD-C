#ifndef _COMMON_H_
#define _COMMON_H_

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "Fastor/Fastor.h"

// Define necessary structures
struct PARAMETERS {
  double tol;
  int maxit;
};

struct OUTTYPE {
  Fastor::Tensor<double> X;
  Fastor::Tensor<double> Y;
  int memo;
  int iter;
  Fastor::Tensor<double> chgv;
  Fastor::Tensor<double> xtrm;
  Fastor::Tensor<double> kktc;
  Fastor::Tensor<double> svk;
  Fastor::Tensor<double> hrvs;
};

// Constants
const double EPS = DBL_EPSILON;
const double INF = std::numeric_limits<double>::infinity();

#endif
