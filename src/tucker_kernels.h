#ifndef TUCKER_KERNELS_H
#define TUCKER_KERNELS_H

#include <vector>
#include <cuda_runtime.h> // For cudaStream_t
#include <cstdio>
#include <cstdlib> // For exit

// CUDA Error Checking
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
 * @brief Launches n-Mode Product kernel: Y = T x_n A or Y = T x_n A^T.
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
 * @brief Launches Matricization (Unfolding) kernel: M = T_(n).
 */
void launch_MatricizeKernel(
    const float* d_input_tensor,
    float* d_output_matrix,
    const std::vector<int>& input_dims,
    int mode,
    cudaStream_t stream
);

/**
 * @brief Launches kernel to copy first N columns: Dest[:, 0:N-1] = Src[:, 0:N-1].
 */
void launch_CopyColumnsKernel(
    float* d_dest_matrix,
    const float* d_src_matrix,
    int num_rows,
    int num_cols_to_copy, // N
    int src_leading_dim,  // Leading dimension of Src
    int dest_leading_dim, // Leading dimension of Dest
    cudaStream_t stream
);


#endif // TUCKER_KERNELS_H 