#include "lmsvd.h"
#include "armadillo-14.2.2/include/armadillo"

#define ARMA_USE_ARPACK
using namespace arma;

double ferr(mat U, mat V) {
  vec u_vector = reshape(U, U.n_rows * U.n_cols, 1);
  vec v_vector = reshape(V, V.n_rows * V.n_cols, 1);
  return norm(u_vector - v_vector, 2) / norm(v_vector, 2);
}

int main() {
  // problem size
  int m = 5000;
  int n = 5000;

  int nR    = 5;
  int pcent = 20;

  int Ranks_factor = std::ceil(pcent / 1000.0 * std::min(m, n) / nR);
  vec Ranks        = zeros(nR, 1);
  for (int i = 0; i < nR; i++) {
    Ranks(i) = (i + 1) * Ranks_factor;
  }

  // generate random A
  arma::arma_rng::set_seed_random();
  mat A = arma::mat(m, n, arma::fill::randu);

  // set options
  PARAMETERS opts;
  opts.tol = 1e-14;
  //	opts.maxit = 4;
  opts.maxit = 150;

  mat U1;
  mat V1;
  vec S1;

  mat     U2;
  mat     V2;
  vec     S2;
  OUTTYPE out;

  using namespace std::chrono;

  high_resolution_clock::time_point time_point_1;
  high_resolution_clock::time_point time_point_2;

  vec T1 = zeros(nR, 1);
  vec E1 = zeros(nR, 1);

  vec T2 = zeros(nR, 1);
  vec E2 = zeros(nR, 1);

  duration<double> t1;
  duration<double> t2;

  double e1;
  double e2;

  for (int j = 1; j <= nR; ++j) {
    int r = Ranks(j - 1);
    int k = r + 10;

    // svd ====== Comment this if slows

     time_point_1 = high_resolution_clock::now();
     //svds(U1, S1, V1, A_temp, r, opts.tol);
     svds(U1, S1, V1, A, r);
     time_point_2 = high_resolution_clock::now();
     t1 = time_point_2 - time_point_1;
     // Truncate U1, S1, V1 to rank r for fair comparison
     e1 = ferr(U1 * diagmat(S1) * trans(V1), A);
     std::cout << "svd: res = \t\t" << e1 << '\t' << "t = " << t1.count() <<
     '\n';

    // T1(j-1) = t1.count();
    // E1(j-1) = e1;

    // lmsvd

    time_point_1 = high_resolution_clock::now();
    lmsvd(A, r, opts, U2, S2, V2, out);
    time_point_2 = high_resolution_clock::now();
    t2           = duration_cast<duration<double>>(time_point_2 - time_point_1);
    e2           = ferr(U2 * diagmat(S2) * trans(V2), A);
    std::cout << "lmsvd(r) " << r << ": res =\t" << e2 << '\t' << "t = " << t2.count() << std::endl;

    // Bench between them
    // U1 = U1.head_cols(r);
    // S1 = S1.head(r);
    // V1 = V1.head_cols(r);
    // auto esvd = ferr(U2 * diagmat(S2) * trans(V2), U1 * diagmat(S1) *
    // trans(V1)); std::cout << "diff res = " << esvd << std::endl << '\n';

    T2(j - 1) = t2.count();
    E2(j - 1) = e2;
  }

  return 0;
}
