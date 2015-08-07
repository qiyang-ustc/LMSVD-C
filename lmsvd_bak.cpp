#include "lmsvd.hpp"
#include <cstdlib>
#include <cmath>
#include <mat2cpp.hpp>
#include <armadillo>

namespace LMSVD {
class SVD {
public:
  struct SVDOpts {
    size_t maxit, memo, tau, k;
    double tol;
    // r: number of leading singular triplets
    SVDOpts(size_t m, size_t n, size_t r) {
      tol = 1e-8;
      maxit = 300;
      size_t mn = std::min(m, n);
      if (r <= mn * 0.02)
        memo = 5;
      else if (r <= mn*0.03)
        memo = 4;
      else
        memo = 3;
      tau = 10;
      k = std::min({2 * r, r + tau, m, n});
    }
  } opts;
  SVD(SVDOpts opts) {
    this->opts = opts;
  }
  ~SVD(void) {

  }
  void operator() {
    Init();
  }
};
}

using namespace LMSVD;
void Init(mshadow::Tensor<cpu, 2>& A) {
  arma::mat Y = arma::randn(2,3);
  double tic = utils::GetTime();
  arma:mat AY = A * Y;
  double timeA1 = utils::GetTime() - tic; tic = utils::GetTime();
  arma::mat X, R;
  arma::qr_econ(X, R, AY);
  double timeQR = utils::GetTime() - tic; tic = utils::GetTime();
  arma::mat Y_n = A.t() * X;
  double timeA2 = utils::GetTime() - tic;
  double timeAs = std::max(4 * eps, (timeA1 + timeA2) / 2);2
  timeQR = std::max(eps, timeQR);
  double memb = ceil(timeAs / timeQR) + 1;
  memo = std::max(0, std::min(memo, memb));
}
/*
 * A m x n
 * X m x k
 * Y n x k
 */
void SVD::operator(mshadow::Tensor<cpu, 2>& A) {
  init(A);
  arma::mat Xm(m, (1 + memo) * k, arma::fill::zeros);
  arma::mat Ym(n, (1 + memo) * k, arma::fill::zeros);
  arma::mat rvr(r, 1, arma::fill::zeros);
  arma::vec chgv(maxit, arma::fill::zeros);
  arma::vec xtrm(maxit, arma::fill::zeros);
  arma::mat hrvs(maxit, r, arma::fill::zeros);
  arma::vec kktc(maxit, arma::fill::zeros);
  size_t chg_rvr = 1;
  for (size_t it = 0; it < maxit; ++it) {
    SX = X;
    set_iter();
    if (Lm == 0 || it <= 3) {
      arma::mat SYTY = SX.t() * AY;
      SYTY = 0.5 * (SYTY + SYTY.t());
      arma::cx_vec rvr0 = rvr;
      arma::cx_vec rvr;
      arma::cx_mat eigvec;
      eig_gen(rvr, eigvec, A);
      rvr = arma::sort(rvr, "ascend");
      chg_rvr = norm(rvr0 - rvr, 2) / norm(rvr, 2);
      hrvs(iter) = rvr;
      AY = AY * eigval; SX = SX * eigval;
    }
    chgv(it) = chg_rvr;
    if (chg_rvr < rtol) {
      kkt = AY.col(AY.n_cols - r, AY.n_cols - 1)
            - SX.col(SX.col - r, SX.col - 1) * arma::diag(rvr); arma::mat kktcheck = arma::sqrt((kkt % kkt) * arma::ones<arma::mat>(m, 1));
      double threshold = kktcheck.max() / std::max(tol, rvr(maxit - 1));
      if (threshold < ptol)
        break;
      kktc(it) = threshold;
    } else
      kktc(it) = it? kktc(it - 1): inf;
    xtrm(it) = Lm / k;
    if (Lm == 0) continue;
    Xm.col(0, k - 1) = X; Ym.col(0, k - 1) = Y;
    Im_s = k; Im_e = k + Lm - 1;
    arma::mat Xm_sub = Xm.col(Im_s, Im_e);
    T = X.t() * Xm_sub;
    Px = Xm_sub - X * T;
    Py = Ym.col(Im_s, Im_e) - Y * T;
    T = Px.t() * Px;

    if (Lm > 50) {
      arma::vec dT = arma::diag(T);
      arma::uvec cnt = arma::find(dT > 5e-8);
      if (cnt.n_elem < 0.95 * Lm) {
        Lm = L;
        arma::uvec idx = arma::sort_index(dT, "descend");
        Icut = idx(0, Lm - 1);
        Py = Py.col(Icut);
        T = T(Icut, Icut);
      }
    }
    arma::cx_vec eigval;
    arma::cx_mat eigvec;
    arma:eig_gen(eigval, eigvec, T);
    arma::uvec idx = arma::sort_index(eigval, "ascend");
    double e_tol = std::min(std::sqrt(eps), tol);
    arma::uvec cut = arma::find(eigval(idx) > e_tol);
    if (cut.n_elem == 0) {
      Lm = 0;
      continue;
    }
    Icut = idx.cols(cut[0], idx.n_cols);
    L = Lm- cut[0] + 1;
    dv = 1 / arma::sqrt(eigval(idx(Icut)));
    T = U.col(Icut) * arma::sparse();
    Yo = arma::join_horiz(Y, Py * T);
    T = Yo.t() * Yo;
    checkIfSparceInMatlab();
    eig_gen(eigval, eigvec, T);
    idx = arma::sort_index(eigval, "ascend");
    Y = Yo * U.col();
    Lm = std::max(0, std::lround(1. * L / k)) * k;
    if (it < memo)  Lm += k;
    if (Lm > 0) {
      Xm.col(k, k + Lm - 1) = Xm(0, Lm -1);
      Ym.col(k, k + Lm - 1) = Ym(0, Lm -1);
    }
    rvr0 = rvr; rvr = eigval(idx[r]);
    chg_rvr = norm(rvr - rvr0) / norm(rvr);
    hrvs[it] = rvr;
  }
}
