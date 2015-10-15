
#include "lmsvd.h"

using namespace arma;

//*****************************
//LMSVD Limited Memory Block Mrylov Subspace Optimization for Computing Principal Singular Value Decompositions
//Input:
//  A -- a numeric matrix
//  r -- number of leading singular triplets
//  opts -- option structure with fields 
//    tol -- tolerance [1.e-8]
//    maxit -- maximal number of iteration [300]
//    memo -- number of block subspaces [default is set from relations between r and m,n]
//    qvk -- number of additional guard vectors [10]
//    initY -- initial guess of a n by r matrix [randn(n,r)]
//    idisp -- detailed information display options [0]
//    U,S,V -- principal SVD of A with the r singular triplets
//    Out -- output information
//****************************
void lmsvd(mat A, int r, PARAMETERS opts, mat & U, vec & S, mat & V, OUTTYPE & Out)
{
	int m = A.n_rows;
	int n = A.n_cols;
	if (r > std::min(m, n)/2)
	{
		std::cout << "Too Many SVS requested r > min(m,n)/2" << std::endl;
	}

  // set parameters
	double tol = 1e-8;
	int maxit = 3;
	//int maxit = 300;
	double idisp = 0;
	int mn = std::min(m, n);
	int memo;
	if(r <= (mn*0.02))
	{
		memo = 5;
	}
	else if(r <= (mn*0.03))
	{
		memo = 4;
	}
	else
	{
		memo = 3;
	}

	int tau = 10;

	// working size
	std::vector<int> temp{2*r, r+tau, m, n};
	int k = *(std::min_element(std::begin(temp), std::end(temp)));

	// initial guess
	mat Y = randn(n, k);

	// need to be rewritten
	tol = opts.tol;
	maxit = opts.maxit;
	//memo = opts.memo;
	//idisp = opts.idisp;

	// set parameters end
	
	// initialize
	mat X = A * Y;
  mat R;
	arma::qr_econ(X, R, X);
	Y = trans(trans(X) * A);
  // initialize end

	// bound memo
  // need to be rewritten
	// bound memo end

  // call solver
	int m_X = X.n_rows;
	int n_Y = Y.n_rows;
	int mn_XY = std::min(m_X, n_Y);
	int k_Y = Y.n_cols;

	if (k_Y < r)
	{
		std::cout << "working size too small" << std::endl;
	}

	mat Xm = zeros(m_X, (1+memo)*k_Y);
	mat Ym = zeros(n_Y, (1+memo)*k_Y);
	Xm.cols(k_Y, 2*k_Y-1) = X;
	Ym.cols(k_Y, 2*k_Y-1) = Y;
	//Xm.submat(span(0, Xm.n_rows-1), span(k_Y+1,2*k_Y)) = X;
	//Ym.submat(span(0, Ym.n_rows-1), span(k_Y+1,2*k_Y)) = Y;
	int Lm = k_Y;

	vec rvr = zeros(r, 1);
	int chg_rvr = 1;
	vec chgv = zeros(maxit, 1);
	vec xtrm = zeros(maxit, 1);
	mat hrvs = zeros(maxit, r);
	vec kktc = zeros(maxit, 1);

  //// set_tolerance
  double tmp = std::min(mn_XY*1.0/40/k_Y, 1.0);
	double qtol = std::pow(EPS, tmp);
	double rtol = 5 * std::max(std::sqrt(tol*qtol), 5*EPS);
	double ptol = 5 * std::max(tol, std::sqrt(EPS));
  //// set_tolerance end
	
	int iter;
  for (iter = 1; iter <= maxit; ++iter)
	{
		mat SX = X;

		mat AY = A * Y;
		mat temp;
		qr_econ(X, temp, AY);
		Y = trans(trans(X)*A);
	
  	//// calculating terminating rules
	  if (Lm == 0 || iter <= 3)
		{
      mat SYTY = trans(SX) * AY;
			SYTY = 0.5 * (SYTY + trans(SYTY));
      mat tU;
			vec tE;
			eig_sym(tE, tU, SYTY);
      vec rvr0 = rvr;
			vec rv_sort = sort(vec(tE), "ascend");
			vec rvr = rv_sort.rows(rv_sort.n_rows-r, rv_sort.n_rows-1);
			chg_rvr = norm(rvr0-rvr, 2) / norm(rvr, 2);
		  hrvs.row(iter-1) = trans(rvr);
		  AY = AY * tU;
		  SX = SX * tU;
		}
		xtrm(iter-1) = Lm / k_Y;
		chgv(iter-1) = chg_rvr;
	  //// calculating terminating rules end
	

	  //// check terminating criterion
	  if (chg_rvr < rtol)
		{
			mat kkt = AY.cols(AY.n_cols-r, AY.n_cols-1) - SX.cols(SX.n_cols-r, SX.n_cols-1) * diagmat(rvr);
			vec kktcheck = sqrt(trans(pow(kkt, 2)) * ones(m, 1));
			double kktcheck_d = max(kktcheck) / std::max(tol, rvr(rvr.n_rows-1));
			if (kktcheck_d < ptol)
			{
				break;
			}
			kktc(iter-1) = kktcheck_d;
		}
		else
		{
			if (iter == 1)
			{
				kktc(iter-1) = INF;
			}
			else
			{
				kktc(iter-1) = kktc(iter-2);
			}
		}
	  //// check terminating criterion end
	
	  //// look-back optimization
		xtrm(iter-1) = Lm / k_Y;
		if (Lm == 0)
		{
			continue;
		}
		Xm.cols(0, k_Y-1) = X;
		Ym.cols(0, k_Y-1) = Y;
		////// projection
	  int Im_start = k_Y;
		int Im_end = k_Y + Lm;
		mat T = trans(X) * Xm.cols(Im_start, Im_end-1);
		mat Px = Xm.cols(Im_start, Im_end-1) - X * T;
		mat Py = Ym.cols(Im_start, Im_end-1) - Y * T;	
		T = trans(Px) * Px;
		////// remove small vectors
		if (Lm > 50)
		{
			vec dT = T.diag();
			vec sdT = sort(dT, "descend");
			uvec idx = sort_index(dT, "descend");
			double L = sum(sdT > (5e-8)*ones(sdT.n_rows, sdT.n_cols));
			if (L < 0.95 * Lm)
			{
				Lm = L;
				int Icut_start = 0;
				int Icut_end = Lm;
				Py = Py.cols(Icut_start, Icut_end-1);
				T = T.submat(span(Icut_start, Icut_end-1), span(Icut_start, Icut_end-1));
			}
		}
		////// orthonormalize Px
		vec ev;
		eig_sym(ev, U, T);
		uvec idx = sort_index(ev, "ascend");
		double e_tol = std::min(std::sqrt(EPS), tol);
		vec e_tol_vec = ones(idx.n_rows, ev.n_cols) * e_tol;
    uvec cut_vec = find(ev.rows(idx) > e_tol_vec, 1);
		if (cut_vec.n_rows == 0)
		{
			Lm = 0;
			continue;
		}
		int Icut_start = cut_vec(0);
		int Icut_end = idx.n_rows;
	  int L = Lm - cut_vec(0);	
		vec ones_vec = ones(Icut_end-Icut_start, 1);
		vec dv = ones_vec / sqrt(ev.rows(idx.rows(Icut_start, Icut_end-1)));
		mat dv_diag_mat = zeros(L, L);
		dv_diag_mat.diag() = dv;
		T = U.cols(Icut_start, Icut_end-1) * dv_diag_mat;
		////// subspace optimization
		mat Yo = join_rows(Y, Py*T);
		T = trans(Yo) * Yo;
		vec ev_T;
		eig_sym(ev_T, U, T);
		vec rv_sort_T = sort(ev_T, "ascend");
    uvec idx_T = sort_index(ev_T, "ascend");
		////// this 'U' might be 'v' in (10)
		Y = Yo*U.cols(idx_T.rows(idx_T.n_rows-k_Y, idx_T.n_rows-1));
		Lm = std::max(0.0, std::round(1.0*L/k_Y)) * k_Y;
		if (iter < memo)
		{
			Lm = Lm + k_Y;
		}
		if (Lm > 0)
		{
			Xm.cols(k_Y, Lm+k_Y-1) = Xm.cols(0, Lm-1);
			Ym.cols(k_Y, Lm+k_Y-1) = Ym.cols(0, Lm-1);
		}
		vec rvr0_ = rvr;
		rvr = rv_sort_T.rows(rv_sort_T.n_rows-r, rv_sort_T.n_rows-1);
		chg_rvr = norm(rvr-rvr0_, 2) / norm(rvr, 2);
		hrvs.row(iter-1) = trans(rvr);
	  //// look-back optimization end
	}
  // iter end
	Out.X = X;
	Out.Y = Y;
	Out.memo = memo;
	Out.iter = iter;
	// iter = maxit + 1
	Out.chgv = chgv.rows(0, iter-2);
	Out.kktc = kktc.rows(0, iter-2);
	Out.xtrm = xtrm.rows(0, iter-2);
	Out.hrvs = hrvs.rows(0, iter-2);
  // call solver end
	
	// generate svd
	bool method = 0;
	mat W;
	mat Z;
	vec s_vec;
	if(method)
	{
		svd_econ(V, s_vec, W, Y);
		U = X * W;
	}
	else
	{
		qr_econ(V, R, Y);
		svd(W, s_vec, Z, trans(R));
		U = X * W;
		V = V * Z;
	}
  Out.svk = s_vec;
	vec svk_temp = pow(Out.svk.rows(0, r-1), 2);
	Out.hrvs.row(Out.hrvs.n_rows-1) = trans(flipud(svk_temp));
	//generate svd end
	
	// output principal SVD
  U = U.cols(0, r-1);
	V = V.cols(0, r-1);
	S = s_vec.rows(0, r-1);
	//S = S.submat(span(0, r-1), span(0, r-1));
  // output principal SVD end
	
}


void lm_lbo()
{
}




