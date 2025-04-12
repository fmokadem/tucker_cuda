#ifndef TUCKER_KERNELS_H
#define TUCKER_KERNELS_H

#include <vector>
#include <cuda_runtime.h> // For cudaStream_t
#include <cstdio>
#include <cstdlib> // For exit

// --- CUDA Error Checking Macro ---
// Provides a simple way to check for CUDA API errors after calls.
// Terminates the program on error. Consider using exceptions for library code.
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
#endif

/**
 * @brief Launches CUDA kernel for n-Mode Product: Y = T x_n A or Y = T x_n A^T.
 */
void launch_nModeProductKernel(
    const float* d_input_tensor,
    float* d_output_tensor,
    const float* d_factor_matrix,
    const std::vector<int>& input_dims,
    const std::vector<int>& output_dims,
    const std::vector<int>& factor_dims,
    int mode,
    bool use_transpose,
    cudaStream_t stream
);

/**
 * @brief Launches CUDA kernel for Matricization (Unfolding): M = T_(n).
 */
void launch_MatricizeKernel(
    const float* d_input_tensor,
    float* d_output_matrix,
    const std::vector<int>& input_dims,
    int mode,
    cudaStream_t stream
);

/**
 * @brief Launches CUDA kernel to copy first columns: Dest[:, 0:N] = Src[:, 0:N].
 */
void launch_CopyColumnsKernel(
    float* d_dest_matrix,
    const float* d_src_matrix,
    int num_rows,
    int num_cols_to_copy,
    int src_cols,
    cudaStream_t stream
);


#endif // TUCKER_KERNELS_H 