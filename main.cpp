
#include "lmsvd.h"

using namespace arma;


double ferr(mat u, mat v)
{
	vec u_vector = reshape(u, size(u,1)*size(u,2), 1);
	vec v_vector = reshape(v, size(v,1)*size(v,2), 1);
	return norm(u_vector - v_vector, 2) / norm(v_vector, 2);
}

int main()
{
	// problem size
	int m = 2000;
	int n = 2000;

	int nR = 5;
	int pcent = 5;

	int Ranks_factor = std::ceil(pcent/100.0 * std::min(m,n) / nR);
	vec Ranks = zeros(1, nR);
	for(int i = 0; i < nR; i++)
	{
	  Ranks(i) = (i+1) * Ranks_factor;	
	}

	// generate random A
	double tau = as_scalar(randu(1,1));
	double base = 1 + 0.1 * tau;
	rowvec d = zeros(1, n);
	for(int i = 0; i < n; i++)
	{
		d(i) = std::max(std::pow(base, -i), EPS);
	}
	sp_mat A_temp = speye(n, n);
	A_temp.diag() = d;
	A_temp = randn(m, n) * A_temp;

	std::srand(unsigned(std::time(0)));

	std::vector<int> permutation_m(m);
	for(int i = 0; i < m; i++)
	{
		permutation_m.push_back(i);
	}
  std::random_shuffle(permutation_m.begin(), permutation_m.end());

	std::vector<int> permutation_n(n);
	for(int i = 0; i < n; i++)
	{
		permutation_n.push_back(i);
	}
  std::random_shuffle(permutation_n.begin(), permutation_n.end());

	mat A = zeros(m, n);
	for(int i = 0; i < m; i++)
	{
		for(int j = 0; j < n; j++)
		{
			A(i,j) = A_temp(permutation_m[i], permutation_n[j]);
		}
	}
	
	// set options
	PARAMETERS opts;
	opts.tol = 1e-8;
	opts.maxit = 150;

	mat T1 = zeros(1, nR);
	mat E1 = T1;
	mat T2 = T1;
	mat E2 = E1;


	for (int j = 1; j <= nR; ++j)
	{
		int r = Ranks(j-1);
		int k = r + 10;

		// svds


		// lmsvd
		mat U2;
		mat V2;
		mat S2;
		OUTTYPE out;
		lmsvd(A, r, opts, U2, S2, V2, out);
	}

	return 0;
}
