#ifndef HOST_LOGIC_H
#define HOST_LOGIC_H

#include <vector>
#include <string>

// Consistent data type for computations
using real = float;

// Calculates product of dimensions
long long product(const std::vector<int>& dims);

/**
 * @brief Performs 4D Tucker decomposition via CUDA-accelerated HOOI.
 *
 * @param h_X Input tensor (host, row-major).
 * @param X_dims Dimensions {I1, I2, I3, I4}.
 * @param h_A Input/Output factor matrices {A1, ..., A4} (host, row-major).
 *            Provides initial guess, overwritten with result.
 *            Must be pre-allocated.
 * @param h_G Output core tensor (host, row-major).
 *            Must be pre-allocated.
 * @param R_dims Target ranks {R1, R2, R3, R4}. Clamped internally.
 * @param tolerance Convergence tolerance (change in factor Frobenius norms).
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