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
	vec Ranks = zeros(nR, 1);
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

	std::ifstream input("data", std::ios::in);
	//std::ifstream input("/Users/abner/ClionProjects/test/data", std::ios::in);
	if(!input)
		std::cerr << "error";
	//input.open("data", std::ios::in);

	double temp = 0.0;
	for(int i = 0; i < 2000; ++i)
	{
		for (int j = 0; j < 2000; ++j)
		{
			input >> temp;
			//std::cout << temp << '\t';
			A(i, j) = temp;
		}
	}

	input.close();

	// set options
	PARAMETERS opts;
	opts.tol = 1e-8;
//	opts.maxit = 4;
	opts.maxit = 150;

	mat T1 = zeros(1, nR);
	mat E1 = T1;
	mat T2 = T1;
	mat E2 = E1;

	mat U2;
	mat V2;
	vec S2;
	OUTTYPE out;

	for (int j = 1; j <= nR; ++j)
	{
		int r = Ranks(j-1);
		int k = r + 10;

		// svds


		// lmsvd

		lmsvd(A, r, opts, U2, S2, V2, out);
	}

/*
	mat A = ones(2000, 2000);

	std::ifstream input("/Users/abner/ClionProjects/test/data", std::ios::in);
	if(!input)
	std::cout << "error";
	//input.open("data", std::ios::in);

	double temp = 0.0;
    for(int i = 0; i < 2000; ++i)
    {
        for (int j = 0; j < 2000; ++j)
        {
            input >> temp;
			std::cout << temp << '\t';
			A(i, j) = temp;
        }
    }

    for(int i = 0; i < 20; ++i)
    {
        for (int j = 0; j < 20; ++j)
        {
			std::cout << A(i, j) << '\t';
        }
		std::cout << '\n';
    }


	input.close();
*/
	return 0;
}
