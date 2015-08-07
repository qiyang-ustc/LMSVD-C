
#ifndef _LMSVD_H_
#define _LMSVD_H_

#include "common.h"

void lmsvd(arma::mat A, int r, PARAMETERS opts, arma::mat U, arma::mat S, arma::mat V, OUTTYPE out);

//void set_param();
//void get_svd();
//void check_matrix();
void lm_lbo();

#endif
