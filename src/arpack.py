import numpy as np
import scipy.sparse.linalg as spla
import time

# Parameters
n = 5000  # Matrix size (n x n)
k_values = [20, 40, 60, 80, 100]  # Number of leading singular values to compute

# Generate a random dense matrix
np.random.seed(42)
A = np.random.randn(n, n).astype(np.float64)

# Benchmark SVDS for different k values
for k in k_values:
    print(f"Calculating {k} leading singular values...")

    # Measure the time taken
    start_time = time.time()

    # Compute the k leading singular values and vectors
    U, s, Vt = spla.svds(A, k=k)

    # Measure the elapsed time
    elapsed_time = time.time() - start_time

    # Output the results
    print(f"Time taken for k={k}: {elapsed_time:.4f} seconds")
