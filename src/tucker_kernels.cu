#include "tucker_kernels.h" // Include the header we just defined
#include <cuda_runtime.h>
#include <device_launch_parameters.h> // For blockIdx, threadIdx etc.
#include <cstdio> // For printf inside kernel (debugging)
#include <vector>
#include <numeric> // For std::accumulate
#include <algorithm> // For std::min

using real = float;

// Re-define CHECK_CUDA for convenience, or include a common utilities header.
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        /* Consider throwing an exception instead of exiting */ \
        exit(EXIT_FAILURE); \
    } \
}
#endif

// --- Device Helper Functions for 4D Row-Major Indexing ---

// Calculates the linear memory index for a 4D tensor element
// given its multi-index (i0, i1, i2, i3) and the tensor dimensions (D0-D3).
// Assumes row-major storage order.
__device__ __forceinline__ long long get_linear_idx_4d_rowmajor(
    const int i0, const int i1, const int i2, const int i3,
    const int D0, const int D1, const int D2, const int D3 // Using D0-D3 consistently
) {
    // idx = i3 + D4 * (i2 + D3 * (i1 + D2 * i0))
    return i3 + (long long)D3 * (
           i2 + (long long)D2 * (
           i1 + (long long)D1 * i0
           ));
}

// Calculates the 4D multi-index (i0, i1, i2, i3) for a tensor element
// given its linear memory index and the tensor dimensions (D0-D3).
// Assumes row-major storage order.
__device__ __forceinline__ void get_multi_idx_4d_rowmajor(
    long long linear_idx,
    const int D0, const int D1, const int D2, const int D3,
    int& i0, int& i1, int& i2, int& i3
) {
    // Ensure valid dims to prevent division by zero if a dim is 1
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
// CUDA kernel to compute the n-mode product: Y = T x_n A or Y = T x_n A^T.
// Each thread calculates one element of the output tensor Y.
__global__ void nModeProductKernelImpl(
    const float* __restrict__ input_tensor,    // Input tensor T (device pointer)
    float* __restrict__ output_tensor,         // Output tensor Y (device pointer)
    const float* __restrict__ factor_matrix,   // Factor matrix A (I_n x R_n, row-major, device pointer)
    const int D0, const int D1, const int D2, const int D3, // Input T dimensions
    const int O0, const int O1, const int O2, const int O3, // Output Y dimensions
    const int I_n, // Dimension size of input tensor T along mode 'n' (must match factor matrix rows)
    const int R_n, // Size of the resultant dimension after multiplication (factor matrix columns)
    const int mode, // The mode 'n' (0-based)
    const bool use_transpose, // If true, compute T x_n A^T, else T x_n A
    const long long output_total_elements)
{
    long long thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < output_total_elements) {

        // Determine the multi-index (o0, o1, o2, o3) in the output tensor Y
        // corresponding to this thread's linear ID.
        int o0, o1, o2, o3;
        get_multi_idx_4d_rowmajor(thread_id, O0, O1, O2, O3, o0, o1, o2, o3);

        float sum = 0.0f;

        // Summation index k runs over the dimension of T corresponding to mode n
        const int K_limit = I_n;

        // Extract the specific index from the output multi-index that corresponds
        // to the mode being operated on. This is needed for indexing the factor matrix.
        int output_mode_idx;
        switch(mode) {
            case 0: output_mode_idx = o0; break;
            case 1: output_mode_idx = o1; break;
            case 2: output_mode_idx = o2; break;
            case 3: output_mode_idx = o3; break;
            default: return; // Should not happen
        }


        for (int k = 0; k < K_limit; ++k) { // Loop index k runs from 0 to I_n - 1
            // Determine input tensor indices (t0, t1, t2, t3)
            int t0 = o0, t1 = o1, t2 = o2, t3 = o3;
            // Overwrite the index corresponding to the current 'mode' with the loop variable 'k'.
            switch(mode) {
                case 0: t0 = k; break;
                case 1: t1 = k; break;
                case 2: t2 = k; break;
                case 3: t3 = k; break;
            }

            // Calculate linear index for Input T(t0, t1, t2, t3)
            long long input_tensor_idx = get_linear_idx_4d_rowmajor(t0, t1, t2, t3, D0, D1, D2, D3);

            // Calculate linear index for the element in the factor matrix A.
            // Indexing depends on whether we use A or A^T.
            // Assumes A is stored row-major (I_n x R_n).
            long long factor_matrix_idx;
            if (use_transpose) {
                // Access A(k, r_n), where r_n is the index from the output tensor's 'mode' dimension.
                factor_matrix_idx = (long long)k * R_n + output_mode_idx;
            } else {
                // Access A(i_n, k), where i_n is the index from the output tensor's 'mode' dimension.
                // This case is less common for Tucker HOOI.
                factor_matrix_idx = (long long)output_mode_idx * R_n + k;
            }

            // Accumulate the product.
            sum += input_tensor[input_tensor_idx] * factor_matrix[factor_matrix_idx];
        }

        // Write the final sum to the corresponding element in the output tensor Y.
        output_tensor[thread_id] = sum;
    }
}


// --- 2. Matricization (Unfolding) Kernel Implementation --- //
// CUDA kernel to perform matricization (unfolding) of a 4D tensor T
// into a 2D matrix M = T_(mode), assuming row-major storage.
// Each thread copies one element from the input tensor to the output matrix.
__global__ void MatricizeKernelImpl(
    const float* __restrict__ input_tensor, // Input tensor T (device pointer)
    float* __restrict__ output_matrix,     // Output matrix M (device pointer)
    const int D0, const int D1, const int D2, const int D3, // Input tensor dimensions
    const int mode, // Mode 'n' to unfold (0-based)
    const long long total_elements) // Total elements in input tensor
{
    long long thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < total_elements) {
        // Determine the multi-index (i0, i1, i2, i3) in the input tensor T
        // corresponding to this thread's linear ID.
        int i0, i1, i2, i3;
        get_multi_idx_4d_rowmajor(thread_id, D0, D1, D2, D3, i0, i1, i2, i3);

        // Row index is simply the index of the unfolded mode.
        int row;
        switch(mode) {
            case 0: row = i0; break;
            case 1: row = i1; break;
            case 2: row = i2; break;
            case 3: row = i3; break;
            default: return;
        }

        // Calculate the column index for the output matrix.
        // This involves flattening the indices of the *other* modes (modes != 'mode').
        // The order of flattening (strides) determines the column layout.
        // Here, the order is D0, D1, D2, D3 (skipping the unfolded mode).
        long long col = 0;
        long long stride = 1;

        if (mode != 0) { col += i0 * stride; stride *= D0; }
        if (mode != 1) { col += i1 * stride; stride *= D1; }
        if (mode != 2) { col += i2 * stride; stride *= D2; }
        if (mode != 3) { col += i3 * stride; stride *= D3; }


        // Determine the total number of columns in the output matrix.
        long long num_cols = (long long)total_elements / ( (mode==0) ? D0 : ((mode==1) ? D1 : ((mode==2) ? D2 : D3)) );
        if (num_cols == 0) return; // Avoid division by zero or index calculation issues

        // Calculate the linear index for the output matrix M (row-major).
        long long output_matrix_idx = (long long)row * num_cols + col;

        // Perform the copy
        output_matrix[output_matrix_idx] = input_tensor[thread_id];
    }
}


// --- 3. Copy Columns Kernel Implementation --- //
// CUDA kernel to copy the first 'num_cols_to_copy' columns from a source matrix
// to a destination matrix. Uses a 2D grid for parallelization.
__global__ void CopyColumnsKernelImpl(
    float* __restrict__ dest_matrix,        // Destination matrix (M x K, device pointer)
    const float* __restrict__ src_matrix,   // Source matrix (M x N, device pointer)
    const int num_rows,                 // Number of rows (M)
    const int num_cols_to_copy,         // Number of columns to copy (K)
    const int src_ld,                   // Leading dimension of source (N for row-major)
    const int dest_ld                   // Leading dimension of destination (K for row-major)
) {
    // Use 2D grid/block indexing
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Index for columns (0..K-1)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Index for rows (0..M-1)

    // Check bounds
    if (row < num_rows && col < num_cols_to_copy) {
        // Calculate linear indices assuming row-major layout
        long long dest_idx = (long long)row * dest_ld + col; // Index in destination matrix
        long long src_idx = (long long)row * src_ld + col;  // Index in source matrix

        // Perform copy
        dest_matrix[dest_idx] = src_matrix[src_idx];
    }
}


// === Host Wrapper Functions === //

// (Removed helper function copy_vector_to_device as dims are passed directly now)
// template<typename T>
// T* copy_vector_to_device(const std::vector<T>& vec) { ... }

// --- 1. n-Mode Product Wrapper --- //
// Host function to configure and launch the nModeProductKernelImpl kernel.
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
    const int num_modes = input_dims.size();
    if (num_modes != 4 || output_dims.size() != 4 || factor_dims.size() != 2) {
         fprintf(stderr, "Error: nModeProductKernel requires 4D tensors and 2D factors.\n");
         exit(EXIT_FAILURE);
    }

    // Calculate total elements in the output tensor for grid size determination.
    long long output_total_elements = 1;
    for (int dim : output_dims) {
         if (dim <= 0) { output_total_elements = 0; break; }
        output_total_elements *= dim;
    }
    if (output_total_elements == 0) {
        // cudaMemsetAsync might be better if buffer needs zeroing
        // CHECK_CUDA(cudaMemsetAsync(d_output_tensor, 0, output_bytes, stream));
        return; // Nothing to compute
    }


    // Configure kernel launch parameters (block size and grid size).
    const int block_size = 512; // Typical block size
    dim3 grid_size((unsigned int)((output_total_elements + block_size - 1) / block_size));
    dim3 block_dim(block_size);

    // Launch the kernel, passing dimensions as individual arguments.
    nModeProductKernelImpl<<<grid_size, block_dim, 0, stream>>>(
        d_input_tensor,
        d_output_tensor,
        d_factor_matrix,
        input_dims[0], input_dims[1], input_dims[2], input_dims[3],
        output_dims[0], output_dims[1], output_dims[2], output_dims[3],
        factor_dims[0], // I_n
        factor_dims[1], // R_n
        mode,
        use_transpose,
        output_total_elements
    );

    // Check for errors immediately after kernel launch (asynchronous call).
    CHECK_CUDA(cudaGetLastError());
}


// --- 2. Matricization Wrapper --- //
// Host function to configure and launch the MatricizeKernelImpl kernel.
void launch_MatricizeKernel(
    const float* d_input_tensor,
    float* d_output_matrix,
    const std::vector<int>& input_dims,
    int mode,
    cudaStream_t stream)
{
     const int num_modes = input_dims.size();
     if (num_modes != 4) {
         fprintf(stderr, "Error: MatricizeKernel requires 4D tensors.\n");
         exit(EXIT_FAILURE);
    }

    // Calculate total elements for grid size determination.
    long long total_elements = 1;
    for (int dim : input_dims) {
         if (dim <= 0) { total_elements = 0; break; }
        total_elements *= dim;
    }
    if (total_elements == 0) return; // Nothing to do

    // Kernel launch configuration
    const int block_size = 512;
    dim3 grid_size((unsigned int)((total_elements + block_size - 1) / block_size));
    dim3 block_dim(block_size);

     // Launch Kernel
    MatricizeKernelImpl<<<grid_size, block_dim, 0, stream>>>(
        d_input_tensor,
        d_output_matrix,
        input_dims[0], input_dims[1], input_dims[2], input_dims[3],
        mode,
        total_elements
    );

    // Check for asynchronous kernel launch errors.
    CHECK_CUDA(cudaGetLastError());
}


// --- 3. Copy Columns Wrapper --- //
// Host function to configure and launch the CopyColumnsKernelImpl kernel.
void launch_CopyColumnsKernel(
    float* d_dest_matrix,
    const float* d_src_matrix,
    int num_rows,
    int num_cols_to_copy,
    int src_cols, // Total columns in source matrix (for source LD)
    cudaStream_t stream)
{
    if (num_rows <= 0 || num_cols_to_copy <= 0) return; // Nothing to do
    if (src_cols <= 0) {
        fprintf(stderr, "Error: Source columns must be positive in CopyColumnsKernel.\n");
        exit(EXIT_FAILURE);
    }
    if (num_cols_to_copy > src_cols) {
         fprintf(stderr, "Error: Cannot copy more columns (%d) than source has (%d) in CopyColumnsKernel.\n", num_cols_to_copy, src_cols);
         exit(EXIT_FAILURE);
    }

    // Determine leading dimensions (number of columns) for row-major matrices.
    int src_ld = src_cols;
    int dest_ld = num_cols_to_copy; // Dest matrix has only the copied columns

    // Kernel launch configuration (2D grid)
    dim3 block_dim(16, 16); // Use a 2D block (e.g., 16x16 = 256 threads)
    dim3 grid_dim(
        (unsigned int)((num_cols_to_copy + block_dim.x - 1) / block_dim.x),
        (unsigned int)((num_rows + block_dim.y - 1) / block_dim.y)
    );

    // Launch Kernel
    CopyColumnsKernelImpl<<<grid_dim, block_dim, 0, stream>>>(
        d_dest_matrix,
        d_src_matrix,
        num_rows,
        num_cols_to_copy,
        src_ld,
        dest_ld
    );

    // Check for asynchronous kernel launch errors.
    CHECK_CUDA(cudaGetLastError());
} 