#ifndef _LMSVD_H_
#define _LMSVD_H_

#include "Fastor/Fastor.h" // Include Fastor header
#include "common.h" // Include any other necessary headers

// Function declaration
void lmsvd(
    const Fastor::Tensor<double> &A, int r, PARAMETERS opts,
    Fastor::Tensor<double> &U, Fastor::Tensor<double> &S,
    Fastor::Tensor<double> &V, OUTTYPE &out);

// void set_param();
// void get_svd();
// void check_matrix();
// void lm_lbo();

#endif
