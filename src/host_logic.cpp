#include "host_logic.h"
#include "tucker_kernels.h" // Kernels for tensor ops

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <cassert>
#include <iomanip>      // For printing progress
#include <algorithm>    // For std::max_element, std::min

// CUDA includes
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// Use the consistent data type from host_logic.h
// using real = float;

// --- CUDA Error Checking Macros --- //
// Redefine error checking macros locally for robustness.
#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(err));                         \
            throw std::runtime_error(cudaGetErrorString(err));                    \
        }                                                                       \
    } while (0)
#endif

#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = call;                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            char error_str[128];                                                \
            snprintf(error_str, sizeof(error_str), "cuBLAS error code %d",       \
                     static_cast<int>(status));                                 \
            fprintf(stderr, "cuBLAS Error in %s at line %d: %s\n", __FILE__,   \
                    __LINE__, error_str);                                       \
            throw std::runtime_error("cuBLAS error");                             \
        }                                                                       \
    } while (0)
#endif

#ifndef CHECK_CUSOLVER
#define CHECK_CUSOLVER(call)                                                    \
    do {                                                                        \
        cusolverStatus_t status = call;                                         \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                \
            char error_str[128];                                                \
            snprintf(error_str, sizeof(error_str), "cuSOLVER error code %d",    \
                     static_cast<int>(status));                                 \
            fprintf(stderr, "cuSOLVER Error in %s at line %d: %s\n", __FILE__, \
                    __LINE__, error_str);                                       \
            throw std::runtime_error("cuSOLVER error");                           \
        }                                                                       \
    } while (0)
#endif

// --- Helper Function --- //
long long product(const std::vector<int>& dims) {
    if (dims.empty()) return 0;
    long long p = 1;
    for (int d : dims) {
        if (d < 0) return -1; // Indicate invalid dimension
        p *= d;
    }
    return p;
}

// --- SVD-based Factor Initialization (Placeholder/Stub) ---
// NOTE: The current implementation is a non-functional stub.
//       Factor initialization relies on the data provided in h_A from the caller.
void initialize_factors_svd(
    cusolverDnHandle_t cusolverH,
    cublasHandle_t cublasH,
    cudaStream_t stream,
    real* d_X,
    const std::vector<int>& X_dims,
    std::vector<real*>& d_A,
    const std::vector<int>& R_dims_clamped,
    int num_modes,
    real* d_temp_unfold, size_t max_unfold_bytes,
    real* d_temp_VT, size_t max_VT_bytes,
    real* d_temp_S, size_t max_S_bytes,
    real* d_svd_work, int lwork,
    int* d_svd_info)
{
    std::cerr << "WARNING: initialize_factors_svd() is a STUB and does not perform SVD initialization!" << std::endl;
    std::cout << "         Factors initialized using data provided from host (h_A)." << std::endl;
}

// --- Main Host Logic for CUDA-accelerated Tucker Decomposition (HOOI) ---
void tucker_hooi_cuda(
    const std::vector<real>& h_X,
    const std::vector<int>& X_dims,
    std::vector<std::vector<real>>& h_A,
    std::vector<real>& h_G,
    const std::vector<int>& R_dims_in,
    real tolerance,
    int max_iterations
    )
{
    if (!(X_dims.size() == 4 && R_dims_in.size() == 4 && h_A.size() == 4)) {
         throw std::invalid_argument("Tucker HOOI requires 4D tensors, 4 ranks, and 4 factor matrix slots.");
    }

    const int num_modes = 4;
    std::vector<int> R_dims = R_dims_in;

    // Clamp ranks
    for (int n = 0; n < num_modes; ++n) {
        if (X_dims[n] <= 0) {
             throw std::invalid_argument("Tensor dimensions must be positive.");
        }
        if (R_dims[n] <= 0) {
             throw std::invalid_argument("Target ranks must be positive.");
        }
        R_dims[n] = std::min(R_dims[n], X_dims[n]);
    }
    std::cout << "Using clamped ranks: [" << R_dims[0] << ", " << R_dims[1] << ", " << R_dims[2] << ", " << R_dims[3] << "]" << std::endl;

    long long X_size = product(X_dims);
    long long G_size = product(R_dims);

    if (X_size < 0 || G_size < 0) {
        throw std::runtime_error("Invalid dimensions or ranks resulted in negative size.");
    }
    if (X_size == 0) {
        h_G.assign(G_size, 0.0f);
        for (int n = 0; n < num_modes; ++n) h_A[n].assign(X_dims[n] * R_dims[n], 0.0f);
        return;
    }

    // Validate input host vector sizes
    if (h_X.size() != static_cast<size_t>(X_size)) {
        throw std::runtime_error("Input tensor host vector size mismatch. Expected " + std::to_string(X_size) +
                                 ", got " + std::to_string(h_X.size()) + ".");
    }
    for(int n=0; n<num_modes; ++n) {
        size_t expected_A_size = (size_t)X_dims[n] * R_dims[n];
        if (h_A[n].size() != expected_A_size) {
             throw std::runtime_error("Host factor matrix size mismatch for mode " + std::to_string(n+1) +
                                      ". Expected " + std::to_string(expected_A_size) + ", got " +
                                      std::to_string(h_A[n].size()) + ".");
        }
    }
    if (h_G.size() != static_cast<size_t>(G_size)) {
        throw std::runtime_error("Host core tensor vector size mismatch. Expected " + std::to_string(G_size) +
                                 ", got " + std::to_string(h_G.size()) + ".");
    }

    // --- 1. Initialize CUDA, cuBLAS, cuSOLVER --- //
    cudaStream_t stream = 0;
    cublasHandle_t cublasH = nullptr;
    cusolverDnHandle_t cusolverH = nullptr;
    try {
        CHECK_CUBLAS(cublasCreate(&cublasH));
        CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
        CHECK_CUBLAS(cublasSetStream(cublasH, stream));
        CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, stream));
    } catch (...) {
        // Cleanup partially initialized handles on error
        if (cusolverH) cusolverDnDestroy(cusolverH);
        if (cublasH) cublasDestroy(cublasH);
        throw;
    }

    // --- 2. Allocate GPU Memory --- //
    real *d_X = nullptr, *d_G = nullptr;
    std::vector<real*> d_A(num_modes, nullptr);
    std::vector<real*> d_A_prev(num_modes, nullptr);
    real *d_Y_projected1 = nullptr;
    real *d_Y_projected2 = nullptr;
    real *d_Y_unfolded = nullptr;
    real *d_VT_svd = nullptr;
    real *d_S_svd = nullptr;
    real *d_svd_work = nullptr;
    int *d_svd_info = nullptr;

    // RAII class or unique_ptr would be safer for cleanup
    try {
        size_t X_bytes = X_size * sizeof(real);
        size_t G_bytes = G_size * sizeof(real);
        CHECK_CUDA(cudaMalloc(&d_X, X_bytes));
        if (G_size > 0) CHECK_CUDA(cudaMalloc(&d_G, G_bytes));

        std::vector<size_t> A_sizes_bytes(num_modes);
        std::vector<size_t> A_element_counts(num_modes);
        for (int n = 0; n < num_modes; ++n) {
            A_element_counts[n] = (size_t)X_dims[n] * R_dims[n];
            A_sizes_bytes[n] = A_element_counts[n] * sizeof(real);
            if (A_element_counts[n] > 0) {
                CHECK_CUDA(cudaMalloc(&d_A[n], A_sizes_bytes[n]));
                CHECK_CUDA(cudaMalloc(&d_A_prev[n], A_sizes_bytes[n]));
            }
        }

        // Determine max sizes needed for intermediate buffers
        size_t max_proj_bytes = 0;
        long long max_unfold_rows = 0, max_unfold_cols = 0;
        size_t max_unfold_bytes = 0;
        size_t max_VT_bytes = 0;
        int max_svd_min_dim = 0;

        for(int n=0; n < num_modes; ++n) {
            std::vector<int> proj_dims = X_dims;
            long long current_unfold_rows = X_dims[n];
            long long current_unfold_cols = 1;
            for (int m=0; m < num_modes; ++m) {
                if (n != m) {
                    proj_dims[m] = R_dims[m];
                    current_unfold_cols *= R_dims[m];
                }
            }
            long long current_proj_size = product(proj_dims);
            max_proj_bytes = std::max(max_proj_bytes, (size_t)current_proj_size * sizeof(real));

            max_unfold_rows = std::max(max_unfold_rows, current_unfold_rows);
            max_unfold_cols = std::max(max_unfold_cols, current_unfold_cols);
            max_unfold_bytes = std::max(max_unfold_bytes, (size_t)(current_unfold_rows * current_unfold_cols) * sizeof(real));

            long long svd_m = current_unfold_cols;
            long long svd_n = current_unfold_rows;
            long long svd_k = std::min(svd_m, svd_n);
            max_VT_bytes = std::max(max_VT_bytes, (size_t)(svd_k * svd_n) * sizeof(real));
            max_svd_min_dim = std::max(max_svd_min_dim, (int)svd_k);
        }

        if (max_proj_bytes > 0) {
             CHECK_CUDA(cudaMalloc(&d_Y_projected1, max_proj_bytes));
             CHECK_CUDA(cudaMalloc(&d_Y_projected2, max_proj_bytes));
        }
        if (max_unfold_bytes > 0) CHECK_CUDA(cudaMalloc(&d_Y_unfolded, max_unfold_bytes));
        if (max_VT_bytes > 0) CHECK_CUDA(cudaMalloc(&d_VT_svd, max_VT_bytes));
        if (max_svd_min_dim > 0) CHECK_CUDA(cudaMalloc(&d_S_svd, max_svd_min_dim * sizeof(real)));
        CHECK_CUDA(cudaMalloc(&d_svd_info, sizeof(int)));

        int lwork = 0;
        // SVD is computed on Y_(n)^T (R_others x I_n)
        CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(
            cusolverH,
            max_unfold_cols,
            max_unfold_rows,
            &lwork
        ));
        if (lwork > 0) CHECK_CUDA(cudaMalloc(&d_svd_work, lwork * sizeof(real)));
        std::cout << "Allocated SVD workspace size: " << lwork << " elements (" << (lwork * sizeof(real)) / (1024.0 * 1024.0) << " MiB)" << std::endl;

        // --- 3. Copy Input Data to GPU --- //
        std::cout << "Copying input data to GPU..." << std::endl;
        CHECK_CUDA(cudaMemcpyAsync(d_X, h_X.data(), X_bytes, cudaMemcpyHostToDevice, stream));
        for (int n = 0; n < num_modes; ++n) {
            if(A_sizes_bytes[n] > 0) {
                CHECK_CUDA(cudaMemcpyAsync(d_A[n], h_A[n].data(), A_sizes_bytes[n], cudaMemcpyHostToDevice, stream));
            }
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::cout << "Input data copied to GPU." << std::endl;

        // --- 4. SVD Initialization (Placeholder) --- //
        // initialize_factors_svd(...); // Currently relies on user-provided h_A

        // --- 5. HOOI Iteration --- //
        std::cout << "Starting HOOI iterations..." << std::endl;
        real change = tolerance + 1.0f;
        int iter = 0;

        while (change > tolerance && iter < max_iterations) {
            change = 0.0f;

            for (int n = 0; n < num_modes; ++n) {
                if(A_sizes_bytes[n] > 0) {
                    CHECK_CUDA(cudaMemcpyAsync(d_A_prev[n], d_A[n], A_sizes_bytes[n], cudaMemcpyDeviceToDevice, stream));
                }
            }

            // Update factors A^(n)
            for (int n_update = 0; n_update < num_modes; ++n_update) {
                // a) Project X onto other factors: Y = X ×_{m!=n} A^(m)ᵀ
                real* d_current_in = d_X;
                std::vector<int> current_in_dims = X_dims;
                real* d_current_out = nullptr;
                int proj_count = 0;

                for (int n_proj = 0; n_proj < num_modes; ++n_proj) {
                    if (n_update == n_proj) continue;

                    std::vector<int> current_out_dims = current_in_dims;
                    current_out_dims[n_proj] = R_dims[n_proj];
                    std::vector<int> factor_dims_proj = {X_dims[n_proj], R_dims[n_proj]};

                    d_current_out = (proj_count % 2 == 0) ? d_Y_projected1 : d_Y_projected2;

                    launch_nModeProductKernel(
                        d_current_in,
                        d_current_out,
                        d_A[n_proj],
                        current_in_dims,
                        current_out_dims,
                        factor_dims_proj,
                        n_proj,
                        true, // Use Transpose
                        stream
                    );

                    d_current_in = d_current_out;
                    current_in_dims = current_out_dims;
                    proj_count++;
                }
                real* d_Y_projected_final = d_current_in;
                std::vector<int> Y_projected_dims = current_in_dims;

                // b) Matricize (Unfold) Y along mode n_update -> Y_(n)
                launch_MatricizeKernel(
                    d_Y_projected_final,
                    d_Y_unfolded,
                    Y_projected_dims,
                    n_update,
                    stream
                );

                // c) Compute SVD of Y_(n)
                // Treat row-major Y_(n) [I_n x R_others] as (Y_(n)^T)^T.
                // SVD input Z = Y_(n)^T [R_others x I_n].
                long long svd_m = 1;
                for(int i=0; i<num_modes; ++i) {
                    if (i != n_update) svd_m *= R_dims[i];
                }
                long long svd_n = X_dims[n_update];

                // Sanity check dimensions
                long long svd_m_check_proj = 1;
                if (svd_n != 0) {
                   for(int dim : Y_projected_dims) svd_m_check_proj *= dim;
                   svd_m_check_proj /= svd_n;
                   assert(svd_m == svd_m_check_proj);
                } else if (svd_m != product(Y_projected_dims)) {
                    assert(false);
                }


                if (svd_m > 0 && svd_n > 0) {
                    int lda_svd = svd_m;
                    // VT output is k x n = min(m,n) x I_n. LD is n.
                    int ldvt_svd = svd_n;

                    // Want U from SVD(Y_(n)). Compute SVD(Z=Y_(n)^T) = U_z S V_z^T.
                    // Y_(n) = Z^T = V_z S U_z^T. U of Y_(n) is V_z.
                    // Need VT output (V_z^T) from SVD(Z).
                    // The first R_n rows of VT are needed.
                    CHECK_CUSOLVER(cusolverDnSgesvd(
                        cusolverH,
                        'N',
                        'S',
                        svd_m,
                        svd_n,
                        d_Y_unfolded,
                        lda_svd,
                        d_S_svd,
                        nullptr,
                        svd_m,
                        d_VT_svd,
                        ldvt_svd,
                        d_svd_work,
                        lwork,
                        nullptr,
                        d_svd_info
                    ));

                    // Optional: Check SVD convergence info
                    // int h_svd_info = 0;
                    // CHECK_CUDA(cudaMemcpyAsync(&h_svd_info, d_svd_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
                    // CHECK_CUDA(cudaStreamSynchronize(stream));
                    // if (h_svd_info < 0) { throw std::runtime_error("SVD failed: invalid argument"); }
                    // else if (h_svd_info > 0) { std::cerr << "Warning: SVD did not converge..." << std::endl; }

                    // d) Update A^(n) = transpose(VT[0:R_n-1, :])
                    real alpha = 1.0f;
                    real beta = 0.0f;
                    // C(I_n, R_n) = alpha * op(A(k, I_n)) + beta * op(B)
                    // op(A) = T. A is first R_n rows of VT.
                    CHECK_CUBLAS(cublasSgeam(
                        cublasH,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        svd_n,              // Rows of C (I_n)
                        R_dims[n_update],   // Cols of C (R_n)
                        &alpha,
                        d_VT_svd,           // A = VT (k x I_n)
                        ldvt_svd,           // lda = I_n
                        &beta,
                        nullptr,
                        svd_n,
                        d_A[n_update],      // C = A^(n) (I_n x R_n)
                        svd_n               // ldc = I_n
                    ));
                } else { // Handle zero-sized SVD input
                     if (A_sizes_bytes[n_update] > 0) {
                         CHECK_CUDA(cudaMemsetAsync(d_A[n_update], 0, A_sizes_bytes[n_update], stream));
                     }
                 }
            }

            // --- 6. Check Convergence --- //
            // Calculate sum ||A^(n) - A_prev^(n)||^2_F
            change = 0.0f;
            for (int n = 0; n < num_modes; ++n) {
                if (A_element_counts[n] > 0) {
                    // A_prev = A_prev - A
                    real alpha_saxpy = -1.0f;
                    CHECK_CUBLAS(cublasSaxpy(cublasH, A_element_counts[n], &alpha_saxpy, d_A[n], 1, d_A_prev[n], 1));
                    // Calculate norm: ||A_prev - A||_F (stored in d_A_prev)
                    real norm_diff = 0.0f;
                    CHECK_CUBLAS(cublasSnrm2(cublasH, A_element_counts[n], d_A_prev[n], 1, &norm_diff));
                    change += norm_diff * norm_diff;
                }
            }
            if (change > 0.0f) {
                 change = sqrt(change);
            }

            iter++;

            if (iter % 10 == 0 || change <= tolerance || iter == max_iterations) {
                 std::cout << "Iter: " << std::setw(4) << iter
                           << ", Change: " << std::scientific << std::setprecision(6) << change
                           << std::endl;
            }
        }

        std::cout << "HOOI iterations finished after " << iter << " iterations." << std::endl;

        // --- 7. Compute Core Tensor G --- //
        // G = X ×₁ A⁽¹⁾ᵀ ×₂ A⁽²⁾ᵀ ×₃ A⁽³⁾ᵀ ×₄ A⁽⁴⁾ᵀ
        std::cout << "Computing core tensor G..." << std::endl;
        real* d_current_in = d_X;
        std::vector<int> current_in_dims = X_dims;
        real* d_current_out = nullptr;
        int proj_count = 0;

        for (int n = 0; n < num_modes; ++n) {
            std::vector<int> current_out_dims = current_in_dims;
            current_out_dims[n] = R_dims[n];
            std::vector<int> factor_dims_n = {X_dims[n], R_dims[n]};

            bool is_last_step = (n == num_modes - 1);
            d_current_out = is_last_step ? d_G : ((proj_count % 2 == 0) ? d_Y_projected1 : d_Y_projected2);

            // Check for null buffers before launch
            if (!is_last_step && !d_current_out) {
                 throw std::runtime_error("Intermediate buffer is null during core tensor computation.");
            }
            if (is_last_step && G_size > 0 && !d_G) {
                 throw std::runtime_error("Core tensor buffer d_G is null when needed.");
            }

            if (product(current_in_dims) > 0 && product(factor_dims_n) > 0) {
                 launch_nModeProductKernel(
                     d_current_in,
                     d_current_out,
                     d_A[n],
                     current_in_dims,
                     current_out_dims,
                     factor_dims_n,
                     n,
                     true, // Use Transpose
                     stream
                 );
             } else if (product(current_out_dims) > 0) {
                 CHECK_CUDA(cudaMemsetAsync(d_current_out, 0, product(current_out_dims) * sizeof(real), stream));
             }

            d_current_in = d_current_out;
            current_in_dims = current_out_dims;
            proj_count++;
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::cout << "Core tensor computed." << std::endl;

        // --- 8. Copy Results Back to Host --- //
        std::cout << "Copying results back to host..." << std::endl;
        if (G_size > 0) CHECK_CUDA(cudaMemcpyAsync(h_G.data(), d_G, G_bytes, cudaMemcpyDeviceToHost, stream));
        for (int n = 0; n < num_modes; ++n) {
            if(A_sizes_bytes[n] > 0) {
                CHECK_CUDA(cudaMemcpyAsync(h_A[n].data(), d_A[n], A_sizes_bytes[n], cudaMemcpyDeviceToHost, stream));
            }
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::cout << "Finished copying results." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during HOOI execution: " << e.what() << std::endl;
        if (cusolverH) cusolverDnDestroy(cusolverH);
        if (cublasH) cublasDestroy(cublasH);
        throw; // Re-throw standard exception
    } catch (...) {
        // --- Cleanup GPU Memory on Error (Unknown Exception) --- //
        std::cerr << "Unknown exception caught during HOOI execution, cleaning up GPU memory..." << std::endl;
        // Avoid CHECK macros here as they might throw again.
        if(d_svd_work) cudaFree(d_svd_work);
        if(d_svd_info) cudaFree(d_svd_info);
        if(d_S_svd) cudaFree(d_S_svd);
        if(d_VT_svd) cudaFree(d_VT_svd);
        if(d_Y_unfolded) cudaFree(d_Y_unfolded);
        if(d_Y_projected2) cudaFree(d_Y_projected2);
        if(d_Y_projected1) cudaFree(d_Y_projected1);
        for (int n = 0; n < num_modes; ++n) {
            if(d_A[n]) cudaFree(d_A[n]);
            if(d_A_prev[n]) cudaFree(d_A_prev[n]);
        }
        if(d_G) cudaFree(d_G);
        if(d_X) cudaFree(d_X);

        if (cusolverH) cusolverDnDestroy(cusolverH);
        if (cublasH) cublasDestroy(cublasH);

        std::cerr << "GPU memory cleanup attempted." << std::endl;
        throw;
    }

    // --- 9. Cleanup GPU Memory (Success Path / After Catch) --- //
    std::cout << "Cleaning up GPU resources..." << std::endl;
    if(d_svd_work) CHECK_CUDA(cudaFree(d_svd_work));
    if(d_svd_info) CHECK_CUDA(cudaFree(d_svd_info));
    if(d_S_svd) CHECK_CUDA(cudaFree(d_S_svd));
    if(d_VT_svd) CHECK_CUDA(cudaFree(d_VT_svd));
    if(d_Y_unfolded) CHECK_CUDA(cudaFree(d_Y_unfolded));
    if(d_Y_projected2) CHECK_CUDA(cudaFree(d_Y_projected2));
    if(d_Y_projected1) CHECK_CUDA(cudaFree(d_Y_projected1));
    for (int n = 0; n < num_modes; ++n) {
        if(d_A[n]) CHECK_CUDA(cudaFree(d_A[n]));
        if(d_A_prev[n]) CHECK_CUDA(cudaFree(d_A_prev[n]));
    }
    if(d_G) CHECK_CUDA(cudaFree(d_G));
    if(d_X) CHECK_CUDA(cudaFree(d_X));

    if (cusolverH) CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    if (cublasH) CHECK_CUBLAS(cublasDestroy(cublasH));

    std::cout << "Tucker decomposition host logic finished." << std::endl;
} 