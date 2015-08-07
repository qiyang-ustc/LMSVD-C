
#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>

#include "../armadillo-5.200.2/include/armadillo"

typedef struct
{
	double tol;
	int maxit;
} PARAMETERS;

typedef struct
{
	arma::mat X;
	arma::mat Y;
	int memo;
	int iter;
	arma::vec chgv;
	arma::vec xtrm;
	arma::vec kktc;
	arma::vec svk;
	arma::mat hrvs;
} OUTTYPE;

// need to be rewritten
const double EPS = 0.00000001;
const double INF = 1000000000;

#endif
