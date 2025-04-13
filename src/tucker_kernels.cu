#include "tucker_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>

using real = float;

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
#endif

// --- Device Helper Functions for 4D Row-Major Indexing ---

__device__ __forceinline__ long long get_linear_idx_4d_rowmajor(
    const int i0, const int i1, const int i2, const int i3,
    const int D0, const int D1, const int D2, const int D3
) {
    // idx = i3 + D3 * (i2 + D2 * (i1 + D1 * i0))
    return i3 + (long long)D3 * (
           i2 + (long long)D2 * (
           i1 + (long long)D1 * i0
           ));
}

__device__ __forceinline__ void get_multi_idx_4d_rowmajor(
    long long linear_idx,
    const int D0, const int D1, const int D2, const int D3,
    int& i0, int& i1, int& i2, int& i3
) {
    long long D1D2D3 = (long long)D1*D2*D3;
    long long D2D3 = (long long)D2*D3;
    long long D3_ll = D3;

    // Handle potential zero strides if dimensions are 1
    if (D1D2D3 > 0) {
        i0 = linear_idx / D1D2D3;
        linear_idx %= D1D2D3;
    } else {
        i0 = 0;
    }

    if (D2D3 > 0) {
        i1 = linear_idx / D2D3;
        linear_idx %= D2D3;
    } else {
        i1 = 0;
    }

    if (D3_ll > 0) {
        i2 = linear_idx / D3_ll;
        linear_idx %= D3_ll;
    } else {
        i2 = 0;
    }
    i3 = linear_idx;
}


// --- 1. n-Mode Product Kernel Implementation --- //
__global__ void nModeProductKernelImpl(
    const float* __restrict__ input_tensor,
    float* __restrict__ output_tensor,
    const float* __restrict__ factor_matrix,
    const int D0, const int D1, const int D2, const int D3, // Input T dimensions
    const int O0, const int O1, const int O2, const int O3, // Output Y dimensions
    const int I_n,
    const int R_n,
    const int mode,
    const bool use_transpose,
    const long long output_total_elements)
{
    long long thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < output_total_elements) {

        int o0, o1, o2, o3;
        get_multi_idx_4d_rowmajor(thread_id, O0, O1, O2, O3, o0, o1, o2, o3);

        float sum = 0.0f;

        const int K_limit = I_n;

        int output_mode_idx;
        switch(mode) {
            case 0: output_mode_idx = o0; break;
            case 1: output_mode_idx = o1; break;
            case 2: output_mode_idx = o2; break;
            case 3: output_mode_idx = o3; break;
            default: return; // Should not happen
        }

        for (int k = 0; k < K_limit; ++k) {
            int t0 = o0, t1 = o1, t2 = o2, t3 = o3;
            switch(mode) {
                case 0: t0 = k; break;
                case 1: t1 = k; break;
                case 2: t2 = k; break;
                case 3: t3 = k; break;
            }

            long long input_tensor_idx = get_linear_idx_4d_rowmajor(t0, t1, t2, t3, D0, D1, D2, D3);

            // Factor matrix indexing assumes row-major (I_n x R_n).
            long long factor_matrix_idx;
            if (use_transpose) {
                // Access A(k, r_n)
                factor_matrix_idx = (long long)k * R_n + output_mode_idx;
            } else {
                // Access A(r_n, k)
                factor_matrix_idx = (long long)output_mode_idx * R_n + k;
            }

            sum += input_tensor[input_tensor_idx] * factor_matrix[factor_matrix_idx];
        }

        output_tensor[thread_id] = sum;
    }
}


// --- 2. Matricization (Unfolding) Kernel Implementation --- //
__global__ void MatricizeKernelImpl(
    const float* __restrict__ input_tensor,
    float* __restrict__ output_matrix,
    const int D0, const int D1, const int D2, const int D3,
    const int mode,
    const long long total_elements)
{
    long long thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < total_elements) {
        int i0, i1, i2, i3;
        get_multi_idx_4d_rowmajor(thread_id, D0, D1, D2, D3, i0, i1, i2, i3);

        int row;
        switch(mode) {
            case 0: row = i0; break;
            case 1: row = i1; break;
            case 2: row = i2; break;
            case 3: row = i3; break;
            default: return;
        }

        // Calculate column index by flattening other modes
        long long col = 0;
        long long stride = 1;
        if (mode != 0) { col += i0 * stride; stride *= D0; }
        if (mode != 1) { col += i1 * stride; stride *= D1; }
        if (mode != 2) { col += i2 * stride; stride *= D2; }
        if (mode != 3) { col += i3 * stride; stride *= D3; }

        long long num_cols = 0;
        int mode_dim_size = (mode==0) ? D0 : ((mode==1) ? D1 : ((mode==2) ? D2 : D3));
        if (mode_dim_size > 0) {
             num_cols = total_elements / mode_dim_size;
        } else {
             return; // Avoid division by zero if mode dimension is zero
        }

        long long output_matrix_idx = (long long)row * num_cols + col;

        output_matrix[output_matrix_idx] = input_tensor[thread_id];
    }
}


// --- 3. Copy Columns Kernel Implementation --- //
__global__ void CopyColumnsKernelImpl(
    float* __restrict__ dest_matrix,
    const float* __restrict__ src_matrix,
    const int num_rows,
    const int num_cols_to_copy,
    const int src_ld,
    const int dest_ld
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows && col < num_cols_to_copy) {
        long long dest_idx = (long long)row * dest_ld + col;
        long long src_idx = (long long)row * src_ld + col;
        dest_matrix[dest_idx] = src_matrix[src_idx];
    }
}


// === Host Wrapper Functions === //

// --- 1. n-Mode Product Wrapper --- //
void launch_nModeProductKernel(
    const float* d_input_tensor,
    float* d_output_tensor,
    const float* d_factor_matrix,
    const std::vector<int>& input_dims,
    const std::vector<int>& output_dims,
    const std::vector<int>& factor_dims,
    int mode,
    bool use_transpose,
    cudaStream_t stream)
{
    if (input_dims.size() != 4 || output_dims.size() != 4 || factor_dims.size() != 2) {
        fprintf(stderr, "Error: nModeProductKernel requires 4D tensors and 2D factor matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
    if (mode < 0 || mode > 3) {
        fprintf(stderr, "Error: nModeProductKernel mode must be between 0 and 3.\n");
        exit(EXIT_FAILURE);
    }

    // Calculate total elements for output tensor for grid size calculation
    long long output_total_elements = 1;
    for(int dim : output_dims) { output_total_elements *= dim; }
    if (output_total_elements == 0) return; // Nothing to compute

    // Kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (output_total_elements + threads_per_block - 1) / threads_per_block;

    // Call the kernel
    nModeProductKernelImpl<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_input_tensor,
        d_output_tensor,
        d_factor_matrix,
        input_dims[0], input_dims[1], input_dims[2], input_dims[3],
        output_dims[0], output_dims[1], output_dims[2], output_dims[3],
        factor_dims[0], // I_n (rows of A)
        factor_dims[1], // R_n (cols of A)
        mode,
        use_transpose,
        output_total_elements
    );
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
}


// --- 2. Matricization (Unfolding) Wrapper --- //
void launch_MatricizeKernel(
    const float* d_input_tensor,
    float* d_output_matrix,
    const std::vector<int>& input_dims,
    int mode,
    cudaStream_t stream)
{
     if (input_dims.size() != 4) {
        fprintf(stderr, "Error: MatricizeKernel requires 4D tensor dimensions.\n");
        exit(EXIT_FAILURE);
    }
    if (mode < 0 || mode > 3) {
        fprintf(stderr, "Error: MatricizeKernel mode must be between 0 and 3.\n");
        exit(EXIT_FAILURE);
    }

    long long total_elements = 1;
    for(int dim : input_dims) { total_elements *= dim; }
    if (total_elements == 0) return; // Nothing to compute

    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    MatricizeKernelImpl<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_input_tensor,
        d_output_matrix,
        input_dims[0], input_dims[1], input_dims[2], input_dims[3],
        mode,
        total_elements
    );
    CHECK_CUDA(cudaGetLastError());
}


// --- 3. Copy Columns Wrapper --- //
void launch_CopyColumnsKernel(
    float* d_dest_matrix,
    const float* d_src_matrix,
    int num_rows,
    int num_cols_to_copy,
    int src_ld,
    int dest_ld,
    cudaStream_t stream)
{
    if (num_rows <= 0 || num_cols_to_copy <= 0) return; // Nothing to copy

    dim3 threads_per_block(16, 16); // Use 2D blocks (256 threads)
    dim3 blocks_per_grid;
    blocks_per_grid.x = (num_cols_to_copy + threads_per_block.x - 1) / threads_per_block.x;
    blocks_per_grid.y = (num_rows + threads_per_block.y - 1) / threads_per_block.y;

    CopyColumnsKernelImpl<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_dest_matrix,
        d_src_matrix,
        num_rows,
        num_cols_to_copy,
        src_ld,
        dest_ld
    );
    CHECK_CUDA(cudaGetLastError());
} 