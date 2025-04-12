#ifndef HOST_LOGIC_H
#define HOST_LOGIC_H

#include <vector>
#include <string> // For file paths or other string parameters if needed

// Define data type consistently (e.g., float or double)
// Use 'real' throughout the host code for easy switching between float/double.
using real = float;


// --- Helper Function Declaration ---
// Calculates the product of dimensions in a vector.
long long product(const std::vector<int>& dims);


/**
 * @brief Performs Tucker decomposition for a 4D tensor using the
 *        Higher-Order Orthogonal Iteration (HOOI) algorithm implemented with CUDA.
 *
 * @param h_X Host vector containing the input 4D tensor (row-major).
 * @param X_dims Dimensions {I1, I2, I3, I4}.
 * @param h_A Host vector of vectors for factor matrices {A1, A2, A3, A4} (row-major).
 *            Initial values provide the starting guess. Must be pre-allocated to
 *            clamped sizes: {I1xR1_clamped, ..., I4xR4_clamped}.
 * @param h_G Host vector for the core tensor G (row-major). Must be pre-allocated to
 *            clamped size: {R1_clamped x ... x R4_clamped}.
 * @param R_dims Target ranks {R1, R2, R3, R4}. Ranks are clamped internally:
 *               R_n_clamped = min(R_n, I_n).
 * @param tolerance Convergence tolerance based on change in factor norms.
 * @param max_iterations Maximum HOOI iterations.
 */
void tucker_hooi_cuda(
    const std::vector<real>& h_X,
    const std::vector<int>& X_dims,
    std::vector<std::vector<real>>& h_A,
    std::vector<real>& h_G,
    const std::vector<int>& R_dims,
    real tolerance = 1e-5,
    int max_iterations = 100
);


#endif // HOST_LOGIC_H 