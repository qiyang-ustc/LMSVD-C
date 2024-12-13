
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

#include "armadillo-14.2.2/include/armadillo"

typedef struct {
  double tol;
  int    maxit;
} PARAMETERS;

typedef struct {
  arma::mat X;
  arma::mat Y;
  int       memo;
  int       iter;
  arma::vec chgv;
  arma::vec xtrm;
  arma::vec kktc;
  arma::vec svk;
  arma::mat hrvs;
} OUTTYPE;

// need to be rewritten
const double EPS = DBL_EPSILON;
const double INF = std::numeric_limits<double>::infinity();

#endif
