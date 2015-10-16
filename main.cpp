#include "lmsvd.h"

using namespace arma;

double ferr(mat U, mat V)
{
	vec u_vector = reshape(U, U.n_rows*U.n_cols, 1);
	vec v_vector = reshape(V, V.n_rows*V.n_cols, 1);
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

  /*

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
	
  */

	input.close();

	// set options
	PARAMETERS opts;
	opts.tol = 1e-8;
//	opts.maxit = 4;
	opts.maxit = 150;

	mat U1;
	mat V1;
	vec S1;

	mat U2;
	mat V2;
	vec S2;
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

	for (int j = 1; j <= nR; ++j)
	{
		int r = Ranks(j-1);
		int k = r + 10;

		// svd
    
		
		time_point_1 = high_resolution_clock::now();
		//svds(U1, S1, V1, A_temp, r, opts.tol);
		svd(U1, S1, V1, A);
		time_point_2 = high_resolution_clock::now();
		t1 = time_point_2 - time_point_1;
		e1 = ferr(U1 * diagmat(S1) * trans(V1), A);
		std::cout << "lmsvd: res = " << e1 << '\t' << "t = " << t1.count() << '\n';

		T1(j-1) = t1.count();
		E1(j-1) = e1;

		// lmsvd

		time_point_1 = high_resolution_clock::now();
		lmsvd(A, r, opts, U2, S2, V2, out);
		time_point_2 = high_resolution_clock::now();
		t2 = duration_cast<duration<double>>(time_point_2 - time_point_1);
		e2 = ferr(U2 * diagmat(S2) * trans(V2), A);
		std::cout << "lmsvd: res = " << e2 << '\t' << "t = " << t2.count() << '\n';
		
		T2(j-1) = t2.count();
		E2(j-1) = e2;
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
