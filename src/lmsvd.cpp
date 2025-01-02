#include "Fastor.h"
#include <cmath>
#include <vector>
#include <algorithm>

using namespace Fastor;

// Define necessary structures
struct PARAMETERS {
    double tol;
    int maxit;
    int memo;
    double idisp;
};

struct OUTTYPE {
    Tensor<double> X;
    Tensor<double> Y;
    int memo;
    int iter;
    Tensor<double> chgv;
    Tensor<double> kktc;
    Tensor<double> xtrm;
    Tensor<double> hrvs;
    Tensor<double> svk;
};

//*****************************
// LMSVD Limited Memory Block Mrylov Subspace Optimization for Computing
// Principal Singular Value Decompositions Input:
//  A -- a numeric matrix
//  r -- number of leading singular triplets
//  opts -- option structure with fields
//    tol -- tolerance [1.e-8]
//    maxit -- maximal number of iteration [300]
//    memo -- number of block subspaces [default is set from relations between r
//    and m,n] qvk -- number of additional guard vectors [10] initY -- initial
//    guess of a n by r matrix [randn(n,r)] idisp -- detailed information
//    display options [0] U,S,V -- principal SVD of A with the r singular
//    triplets Out -- output information
//****************************
void lmsvd(const Tensor<double> &A, int r, PARAMETERS opts, Tensor<double> &U, Tensor<double> &S, Tensor<double> &V, OUTTYPE &Out) {
    int m = A.dimension(0);
    int n = A.dimension(1);
    if (r > std::min(m, n) / 2) {
        std::cout << "Too Many SVS requested r > min(m,n)/2" << std::endl;
    }

    // set parameters
    double tol   = 1e-8;
    int    maxit = 3;
    double idisp = 0;
    int    mn    = std::min(m, n);
    int    memo;
    if (r <= (mn * 0.02)) {
        memo = 5;
    } else if (r <= (mn * 0.03)) {
        memo = 4;
    } else {
        memo = 3;
    }

    int tau = 10;

    // working size
    std::vector<int> temp{2 * r, r + tau, m, n};
    int              k = *(std::min_element(std::begin(temp), std::end(temp)));

    // initial guess
    Tensor<double> Y = random_normal<double>(n, k);

    // need to be rewritten
    tol   = opts.tol;
    maxit = opts.maxit;

    // initialize
    Tensor<double> X = matmul(A, Y);
    Tensor<double> R;
    qr(X, R, X);
    Y = transpose(matmul(transpose(X), A));
    // initialize end

    // bound memo
    // need to be rewritten
    // bound memo end

    // call solver
    int m_X   = X.dimension(0);
    int n_Y   = Y.dimension(0);
    int mn_XY = std::min(m_X, n_Y);
    int k_Y   = Y.dimension(1);

    if (k_Y < r) {
        std::cout << "working size too small" << std::endl;
    }

    Tensor<double> Xm = zeros<double>(m_X, (1 + memo) * k_Y);
    Tensor<double> Ym = zeros<double>(n_Y, (1 + memo) * k_Y);
    Xm(slice(k_Y, 2 * k_Y - 1)) = X;
    Ym(slice(k_Y, 2 * k_Y - 1)) = Y;
    int Lm = k_Y;

    Tensor<double> rvr     = zeros<double>(r, 1);
    int chg_rvr = 1;
    Tensor<double> chgv    = zeros<double>(maxit, 1);
    Tensor<double> xtrm    = zeros<double>(maxit, 1);
    Tensor<double> hrvs    = zeros<double>(maxit, r);
    Tensor<double> kktc    = zeros<double>(maxit, 1);

    //// set_tolerance
    double tmp  = std::min(mn_XY * 1.0 / 40 / k_Y, 1.0);
    double qtol = std::pow(std::numeric_limits<double>::epsilon(), tmp);
    double rtol = 5 * std::max(std::sqrt(tol * qtol), 5 * std::numeric_limits<double>::epsilon());
    double ptol = 5 * std::max(tol, std::sqrt(std::numeric_limits<double>::epsilon()));
    //// set_tolerance end

    int iter;
    for (iter = 1; iter <= maxit; ++iter) {
        Tensor<double> SX = X;

        Tensor<double> AY = matmul(A, Y);
        Tensor<double> temp;
        qr(X, temp, AY);
        Y = transpose(matmul(transpose(X), A));

        //// calculating terminating rules
        if (Lm == 0 || iter <= 3) {
            Tensor<double> SYTY = matmul(transpose(SX), AY);
            SYTY     = 0.5 * (SYTY + transpose(SYTY));
            Tensor<double> tU;
            Tensor<double> tE;
            eig(SYTY, tE, tU);
            Tensor<double> rvr0           = rvr;
            Tensor<double> rv_sort        = sort(tE, "ascend");
            rvr            = rv_sort(slice(rv_sort.size() - r, rv_sort.size() - 1));
            chg_rvr            = norm(rvr0 - rvr, 2) / norm(rvr, 2);
            hrvs(iter - 1) = transpose(rvr);
            AY                 = matmul(AY, tU);
            SX                 = matmul(SX, tU);
        }
        xtrm(iter - 1) = Lm / k_Y;
        chgv(iter - 1) = chg_rvr;
        //// calculating terminating rules end

        //// check terminating criterion
        if (chg_rvr < rtol) {
            Tensor<double> kkt = AY(slice(AY.dimension(1) - r, AY.dimension(1) - 1)) -
                                  matmul(SX(slice(SX.dimension(1) - r, SX.dimension(1) - 1)), diag(rvr));
            Tensor<double>    kktcheck   = sqrt(matmul(transpose(pow(kkt, 2)), ones<double>(m, 1)));
            double kktcheck_d = max(kktcheck) / std::max(tol, rvr(rvr.size() - 1));
            if (kktcheck_d < ptol) {
                break;
            }
            kktc(iter - 1) = kktcheck_d;
        } else {
            if (iter == 1) {
                kktc(iter - 1) = std::numeric_limits<double>::infinity();
            } else {
                kktc(iter - 1) = kktc(iter - 2);
            }
        }
        //// check terminating criterion end

        //// look-back optimization
        xtrm(iter - 1) = Lm / k_Y;
        if (Lm == 0) {
            continue;
        }
        Xm(slice(0, k_Y - 1)) = X;
        Ym(slice(0, k_Y - 1)) = Y;
        ////// projection
        int Im_start = k_Y;
        int Im_end   = k_Y + Lm;
        Tensor<double> T        = matmul(transpose(X), Xm(slice(Im_start, Im_end - 1)));
        Tensor<double> Px       = Xm(slice(Im_start, Im_end - 1)) - matmul(X, T);
        Tensor<double> Py       = Ym(slice(Im_start, Im_end - 1)) - matmul(Y, T);
        T            = matmul(transpose(Px), Px);
        ////// remove small vectors
        if (Lm > 50) {
            Tensor<double> dT  = diag(T);
            Tensor<double> sdT = sort(dT, "descend");
            Tensor<size_t> idx = argsort(dT, "descend");
            double L   = sum(sdT > (5e-8) * ones<double>(sdT.size()));
            if (L < 0.95 * Lm) {
                Lm             = L;
                int Icut_start = 0;
                int Icut_end   = Lm;
                Py             = Py(slice(Icut_start, Icut_end - 1));
                T              = T(slice(Icut_start, Icut_end - 1), slice(Icut_start, Icut_end - 1));
            }
        }
        ////// orthonormalize Px
        Tensor<double> ev;
        Tensor<double> U;
        eig(T, ev, U);
        Tensor<size_t> idx       = argsort(ev, "ascend");
        double e_tol     = std::min(std::sqrt(std::numeric_limits<double>::epsilon()), tol);
        Tensor<double> e_tol_vec = ones<double>(idx.size()) * e_tol;
        Tensor<size_t> cut_vec   = find(ev(idx) > e_tol_vec, 1);
        if (cut_vec.size() == 0) {
            Lm = 0;
            continue;
        }
        int Icut_start     = cut_vec(0);
        int Icut_end       = idx.size();
        int L              = Lm - cut_vec(0);
        Tensor<double> ones_vec       = ones<double>(Icut_end - Icut_start);
        Tensor<double> dv             = ones_vec / sqrt(ev(idx(slice(Icut_start, Icut_end - 1))));
        Tensor<double> dv_diag_mat    = zeros<double>(L, L);
        dv_diag_mat.diag() = dv;
        T                  = matmul(U(slice(Icut_start, Icut_end - 1)), dv_diag_mat);
        ////// subspace optimization
        Tensor<double> Yo = concatenate(Y, matmul(Py, T), 1);
        T      = matmul(transpose(Yo), Yo);
        Tensor<double> ev_T;
        eig(T, ev_T, U);
        Tensor<double> rv_sort_T = sort(ev_T, "ascend");
        Tensor<size_t> idx_T     = argsort(ev_T, "ascend");
        ////// this 'U' might be 'v' in (10)
        Y  = matmul(Yo, U(slice(idx_T.size() - k_Y, idx_T.size() - 1)));
        Lm = std::max(0.0, std::round(1.0 * L / k_Y)) * k_Y;
        if (iter < memo) {
            Lm = Lm + k_Y;
        }
        if (Lm > 0) {
            Xm(slice(k_Y, Lm + k_Y - 1)) = Xm(slice(0, Lm - 1));
            Ym(slice(k_Y, Lm + k_Y - 1)) = Ym(slice(0, Lm - 1));
        }
        Tensor<double> rvr0_          = rvr;
        rvr                = rv_sort_T(slice(rv_sort_T.size() - r, rv_sort_T.size() - 1));
        chg_rvr            = norm(rvr - rvr0_, 2) / norm(rvr, 2);
        hrvs(iter - 1) = transpose(rvr);
        //// look-back optimization end
    }
    // iter end
    Out.X    = X;
    Out.Y    = Y;
    Out.memo = memo;
    Out.iter = iter;
    // iter = maxit + 1
    Out.chgv = chgv(slice(0, iter - 2));
    Out.kktc = kktc(slice(0, iter - 2));
    Out.xtrm = xtrm(slice(0, iter - 2));
    Out.hrvs = hrvs(slice(0, iter - 2));
    // call solver end

    // generate svd
    bool method = 0;
    Tensor<double> W;
    Tensor<double> Z;
    Tensor<double> s_vec;
    if (method) {
        svd(Y, s_vec, W, V);
        U = matmul(X, W);
    } else {
        qr(Y, R, V);
        svd(R, s_vec, Z, W);
        U = matmul(X, W);
        V = matmul(V, Z);
    }
    Out.svk                           = s_vec;
    Tensor<double> svk_temp                      = pow(Out.svk(slice(0, r - 1)), 2);
    Out.hrvs(Out.hrvs.size() - 1) = transpose(flip(svk_temp));
    // generate svd end

    // output principal SVD
    U = U(slice(0, r - 1));
    V = V(slice(0, r - 1));
    S = s_vec(slice(0, r - 1));
    // output principal SVD end
}
