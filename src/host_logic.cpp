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
#include <limits>       // For numeric_limits

// CUDA includes
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// Use the consistent data type from host_logic.h
using real = float;

// --- CUDA Error Checking Macros ---
// Throw runtime_error on CUDA API errors.
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
// Throw runtime_error on cuBLAS API errors.
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
// Throw runtime_error on cuSOLVER API errors.
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

// --- HOSVD-based Factor Initialization ---
// Replaces the previous stub implementation.
void initialize_factors_svd( // Renamed from pseudocode 'initialize_factors_hosvd' to match existing stub name
    const real* d_X,                  // Input tensor (GPU)
    const std::vector<int>& X_dims,   // Input tensor dimensions {I0, I1, I2, I3}
    const std::vector<int>& R_dims,   // Target ranks (clamped) {R0, R1, R2, R3}
    std::vector<real*>& d_A,          // Output Factor matrices (GPU pointers) {d_A0, d_A1, d_A2, d_A3}
    cusolverDnHandle_t cusolverH,
    cublasHandle_t cublasH,
    cudaStream_t stream
) {
    //printf("--- Starting HOSVD Initialization ---\n"); // Commented out
    const int num_modes = X_dims.size(); // Should be 4 based on caller checks

    if (num_modes != 4 || R_dims.size() != num_modes || d_A.size() != num_modes) {
         throw std::runtime_error("HOSVD requires 4D tensors, 4 ranks, and 4 factor matrix pointers.");
    }

    // --- 1. Determine Max Buffer Sizes ---
    long long max_unfold_rows = 0;
    long long max_unfold_cols = 0;
    long long max_unfold_elements = 0;
    int max_svd_m = 0; // Max 'm' dimension fed to SVD (rows of input to SVD)
    int max_svd_n = 0; // Max 'n' dimension fed to SVD (cols of input to SVD)
    int max_S_elements = 0; // Max size of singular value vector

    for (int n = 0; n < num_modes; ++n) {
        long long unfold_rows_n = X_dims[n]; // I_n
        long long unfold_cols_n = 1;
        for (int m = 0; m < num_modes; ++m) {
            if (m != n) {
                if (X_dims[m] == 0) { unfold_cols_n = 0; break; } // Avoid overflow if dim is 0
                unfold_cols_n *= X_dims[m];
            }
        }

        max_unfold_rows = std::max(max_unfold_rows, unfold_rows_n);
        max_unfold_cols = std::max(max_unfold_cols, unfold_cols_n);
        max_unfold_elements = std::max(max_unfold_elements, unfold_rows_n * unfold_cols_n);

        // Dimensions for SVD (computing SVD(X_(n)^T))
        // Input to SVD is treated as column-major: (Product Others) x I_n
        int svd_m_n = static_cast<int>(unfold_cols_n); // Rows for SVD = original columns
        int svd_n_n = static_cast<int>(unfold_rows_n); // Cols for SVD = original rows (I_n)
        max_svd_m = std::max(max_svd_m, svd_m_n);
        max_svd_n = std::max(max_svd_n, svd_n_n);
        max_S_elements = std::max(max_S_elements, std::min(svd_m_n, svd_n_n));
    }

    if (max_unfold_elements == 0) {
         //printf("Input tensor is empty, skipping HOSVD.\n"); // Commented out
         // Factors d_A should already be allocated (possibly zero-size) by caller.
         return;
    }


    // --- 2. Allocate Temporary GPU Buffers ---
    real* d_X_unfold = nullptr;
    real* d_S = nullptr;
    real* d_VT = nullptr;
    int* d_info = nullptr;
    real* d_svd_work = nullptr;
    int lwork = 0;
    real* d_dummy_B = nullptr; // Allocate a dummy buffer for cublasSgeam
    real* d_Tmp_ColMajor = nullptr; // Intermediate buffer for transpose

    try {
        CHECK_CUDA(cudaMalloc(&d_X_unfold, max_unfold_elements * sizeof(real)));
        CHECK_CUDA(cudaMalloc(&d_S, max_S_elements * sizeof(real)));

        // VT size: min(m, n) x n = max_S_elements x max_svd_n
        size_t VT_elements = (size_t)max_S_elements * max_svd_n;
        CHECK_CUDA(cudaMalloc(&d_VT, VT_elements * sizeof(real)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

        // Allocate workspace for SVD
        CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(cusolverH, max_svd_m, max_svd_n, &lwork));
        if (lwork < 0) { // Check for potential negative lwork
            throw std::runtime_error("cusolverDnSgesvd_bufferSize returned negative lwork: " + std::to_string(lwork));
        }
        CHECK_CUDA(cudaMalloc(&d_svd_work, (size_t)lwork * sizeof(real)));

        // Allocate dummy buffer (size based on max In * Rn)
        int max_In = 0; // Avoid potential issue if X_dims is empty
        if (!X_dims.empty()) max_In = *std::max_element(X_dims.begin(), X_dims.end());
        int max_Rn = 0;
        if (!R_dims.empty()) max_Rn = *std::max_element(R_dims.begin(), R_dims.end());

        if (max_In > 0 && max_Rn > 0) {
             CHECK_CUDA(cudaMalloc(&d_dummy_B, (size_t)max_In * max_Rn * sizeof(real)));
             // Allocate temp buffer for col-major In x Rn matrix
             CHECK_CUDA(cudaMalloc(&d_Tmp_ColMajor, (size_t)max_In * max_Rn * sizeof(real)));
        }

    } catch (...) {
        // Cleanup partially allocated buffers on error
        if(d_Tmp_ColMajor) CHECK_CUDA(cudaFree(d_Tmp_ColMajor));
        if(d_dummy_B) CHECK_CUDA(cudaFree(d_dummy_B));
        if(d_svd_work) CHECK_CUDA(cudaFree(d_svd_work));
        if(d_info) CHECK_CUDA(cudaFree(d_info));
        if(d_VT) CHECK_CUDA(cudaFree(d_VT));
        if(d_S) CHECK_CUDA(cudaFree(d_S));
        if(d_X_unfold) CHECK_CUDA(cudaFree(d_X_unfold));
        //printf("Error during HOSVD temporary buffer allocation.\n"); // Commented out
        throw; // Re-throw the exception
    }


    // --- 3. Loop Through Modes ---
    for (int n = 0; n < num_modes; ++n) {
        //printf("Initializing factor for mode %d...\n", n); // Commented out

        long long current_unfold_rows = X_dims[n]; // I_n
        long long current_unfold_cols = 1;
         for (int m = 0; m < num_modes; ++m) {
            if (m != n) {
                 if (X_dims[m] == 0) { current_unfold_cols = 0; break; }
                 current_unfold_cols *= X_dims[m];
             }
        }
        long long current_unfold_elements = current_unfold_rows * current_unfold_cols;

        if (current_unfold_elements == 0 || R_dims[n] == 0) {
            //printf("  Skipping mode %d due to zero dimension or rank.\n", n); // Commented out
            // No need to initialize d_A[n], caller ensures it's allocated (possibly zero size)
            continue;
        }


        // --- a. Unfold X along mode n ---
        //printf("  Unfolding tensor (mode %d) -> (%lld x %lld)...\n", n, current_unfold_rows, current_unfold_cols); // Commented out
        // Assuming launch_MatricizeKernel unfolds into row-major d_X_unfold[I_n][Others]
        launch_MatricizeKernel(d_X, d_X_unfold, X_dims, n, stream);
        CHECK_CUDA(cudaGetLastError()); // Check kernel launch
        CHECK_CUDA(cudaDeviceSynchronize()); // Ensure kernel completes before SVD


        // --- b. Prepare for SVD ---
        // We compute SVD of X_(n)^T, where X_(n) is the matrix in d_X_unfold (I_n x Others, row-major).
        // When read by cuSOLVER as column-major, X_(n)^T is (Others x I_n).
        int svd_m = static_cast<int>(current_unfold_cols); // Rows for SVD = Others
        int svd_n_dim = static_cast<int>(current_unfold_rows); // Cols for SVD = I_n
        // Leading dimension of the matrix as interpreted by cuSOLVER (col-major):
        // Since input d_X_unfold is row-major I_n x Others, reading as col-major Others x I_n means ld = Others = svd_m
        int svd_lda = svd_m; // LDA for the input matrix X_(n)^T

        // SVD computes: A = U' * S * VT' (where A = X_(n)^T)
        // We want the first R_n columns of the left singular vectors of X_(n). Let X_(n) = U * S * V^T.
        // We compute SVD of A = X_(n)^T = (U*S*V^T)^T = V*S*U^T.
        // So, A = U'*S*VT' => U' = V and VT' = U^T.
        // The left singular vectors U of X_(n) are the rows of VT' = U^T.
        // We need VT' from cusolver, which computes A = U' S (V')^T.
        // So we need VT = (V')^T from cuSOLVER. Wait, the pseudocode was correct.
        // A = X_(n)^T. SVD(A) = U' S (V')^T.
        // We want the first R_n columns of U from SVD(X_(n)).
        // X_(n) = U S V^T.
        // A = X_(n)^T = V S U^T.
        // Comparing A = U' S (V')^T with A = V S U^T, we have U' = V and (V')^T = U^T.
        // So the desired U is V'. We need the columns of V'.
        // cuSolver computes SVD of A (our X_(n)^T) and gives us `VT = (V')^T`.
        // The columns of V' are the rows of VT.
        // So we need the first R_n rows of VT computed by cuSolver.

        signed char jobu = 'N';  // Don't compute U' (which is V)
        signed char jobvt = 'S'; // Compute first min(m, n) rows of VT = (V')^T = U^T. These rows contain the vectors we need.
                                 // The rows of U^T are the columns of U. Correct.

        //printf("  Computing SVD (%d x %d)...\n", svd_m, svd_n_dim); // Commented out
        CHECK_CUSOLVER(cusolverDnSgesvd(
            cusolverH,
            jobu,       // Don't compute U'
            jobvt,      // Compute first min(m,n) rows of VT = U^T
            svd_m,      // Rows of A = X_(n)^T = Others
            svd_n_dim,  // Cols of A = X_(n)^T = I_n
            d_X_unfold, // Input matrix A = X_(n)^T (interpreted col-major)
            svd_lda,    // Leading Dim = svd_m = Others
            d_S,        // Output Singular values
            nullptr,    // Placeholder for U' (not computed)
            svd_m,      // LDU placeholder
            d_VT,       // Output VT = U^T. Size min(m,n) x n_dim = min(Others, I_n) x I_n
            svd_n_dim,  // Leading Dimension of VT = I_n
            d_svd_work, // Workspace
            lwork,      // Workspace size
            nullptr,    // rwork (deprecated)
            d_info      // Convergence info
        ));
         CHECK_CUDA(cudaDeviceSynchronize()); // Ensure SVD completes


        // --- c. Check SVD Convergence ---
        int info_host = -1; // Initialize to invalid
        CHECK_CUDA(cudaMemcpy(&info_host, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info_host < 0) {
             std::string msg = "HOSVD Error: cusolverDnSgesvd argument " + std::to_string(-info_host) + " was illegal for mode " + std::to_string(n) + ".";
             //printf("HOSVD Error: %s\n", msg.c_str()); // Commented out
             throw std::runtime_error(msg);
        } else if (info_host > 0) {
             // Non-convergence might not be fatal for initialization, print warning.
             //printf("HOSVD Warning: cusolverDnSgesvd did not converge for mode %d. %d superdiagonals failed to converge.\n", n, info_host); // Commented out
             // Continue, the computed factors might still be usable.
        }


        // --- d. Extract Factor A^(n) --- Step 1: Transpose VT submatrix -> Temp ColMajor --- 
        //printf("  Extracting factor A[%d] (size %d x %d)... Step 1/2\n", n, In, Rn); // Commented out
        int Rn = R_dims[n]; // Target rank for this mode
        int In = X_dims[n]; // Dimension for this mode

        // First transpose: VT[0:Rn-1, 0:In-1] (Rn x In, ld=In) -> d_Tmp_ColMajor (In x Rn, ld=In)
        // C = alpha * op(A) + beta * op(B)
        // C = d_Tmp_ColMajor (m=In, n=Rn, ldc=In)
        // A = d_VT (first Rn rows) (conceptual Rn x In, lda=In)
        // op(A) = transpose(A) (In x Rn)
        // B = d_dummy_B (m=In, n=Rn, ldb=In)
        // op(B) = no-op(B)
        const real alpha = 1.0f;
        const real beta = 0.0f;
        CHECK_CUBLAS(cublasSgeam(cublasH,
                                 CUBLAS_OP_T,    // Transpose Source (VT submatrix)
                                 CUBLAS_OP_N,    // No-Op on Dummy B
                                 In,             // m: Rows of op(A) and C = In
                                 Rn,             // n: Cols of op(A) and C = Rn
                                 &alpha,
                                 d_VT,           // A: Source matrix (VT)
                                 svd_n_dim,      // lda: Leading dim of source d_VT = In
                                 &beta,
                                 d_dummy_B,      // B: Dummy matrix B pointer
                                 In,             // ldb: Use m=In
                                 d_Tmp_ColMajor, // C: Destination is Temp buffer
                                 In              // ldc: Leading dim of C (>= m=In), use In for ColMajor
                                ));
        CHECK_CUDA(cudaDeviceSynchronize()); // Sync after first transpose

        // --- d. Extract Factor A^(n) --- Step 2: Transpose Temp ColMajor -> Final RowMajor d_A[n] ---
        //printf("  Extracting factor A[%d] ... Step 2/2\n", n); // Commented out
        // Second transpose: d_Tmp_ColMajor (In x Rn, ld=In) -> d_A[n] (In x Rn, ld=Rn)
        // C = alpha * op(A) + beta * op(B)
        // C = d_A[n] (m=Rn, n=In, ldc=Rn)
        // A = d_Tmp_ColMajor (In x Rn, lda=In)
        // op(A) = transpose(A) (Rn x In)
        // B = d_dummy_B (m=Rn, n=In, ldb=Rn)
        // op(B) = no-op(B)
        CHECK_CUBLAS(cublasSgeam(cublasH,
                                 CUBLAS_OP_T,    // Transpose Source (Temp buffer)
                                 CUBLAS_OP_N,    // No-Op on Dummy B
                                 Rn,             // m: Rows of op(A) and C = Rn
                                 In,             // n: Cols of op(A) and C = In
                                 &alpha,
                                 d_Tmp_ColMajor, // A: Source is Temp buffer
                                 In,             // lda: Leading dim of source = In
                                 &beta,
                                 d_dummy_B,      // B: Dummy matrix B pointer
                                 Rn,             // ldb: Use m=Rn
                                 d_A[n],         // C: Destination is final d_A[n]
                                 Rn              // ldc: Leading dim of C (>=m=Rn), use Rn for RowMajor
                                ));
        CHECK_CUDA(cudaDeviceSynchronize()); // Sync after second transpose

        //printf("  Factor A[%d] initialized.\n", n); // Commented out

    } // End For loop n

    // --- 4. Cleanup Temporary Buffers ---
    //printf("Cleaning up temporary HOSVD buffers...\n"); // Commented out
    if (d_Tmp_ColMajor) CHECK_CUDA(cudaFree(d_Tmp_ColMajor)); // Free temp transpose buffer
    if (d_dummy_B) CHECK_CUDA(cudaFree(d_dummy_B));
    CHECK_CUDA(cudaFree(d_svd_work));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUDA(cudaFree(d_VT));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_X_unfold));

    //printf("--- HOSVD Initialization Complete ---\n"); // Commented out
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
    // ========================================================================
    // ALERT / TODO: Potential Issues in HOOI Algorithm Implementation
    // ------------------------------------------------------------------------
    // Testing (e.g., via test_tucker_cuda.py) indicates that the current
    // implementation produces results with:
    //   1. High Relative Reconstruction Error: The computed core and factors
    //      do not accurately reconstruct the original tensor.
    //   2. Lack of Factor Orthogonality: The computed factor matrices (A_n)
    //      are not sufficiently orthogonal (||A_n^T * A_n - I|| is large).
    //
    // This suggests potential problems within the main HOOI iteration loop below,
    // specifically concerning:
    //   - How the SVD results are extracted and used to update factors.
    //   - The tensor projection steps (n-mode product kernel calls).
    //   - The calculation of the convergence metric (`delta`).
    //   - General numerical stability or correctness of the iterative updates.
    //
    // The HOSVD initialization (`initialize_factors_svd`) appears to run,
    // but the subsequent iterative refinement fails to converge to an accurate
    // and orthogonal solution within the specified iterations/tolerance.
    // Further debugging and verification of the HOOI steps are required.
    // ========================================================================

    if (!(X_dims.size() == 4 && R_dims_in.size() == 4)) {
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
    //printf("Using clamped ranks: [%d, %d, %d, %d]\n", R_dims[0], R_dims[1], R_dims[2], R_dims[3]); // Commented out

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
        //printf("Allocated SVD workspace size: %d elements (%f MiB)\n", lwork, (lwork * sizeof(real)) / (1024.0 * 1024.0)); // Commented out

        // --- 3. Copy Input Data to GPU --- //
        //printf("Copying input data to GPU...\n"); // Commented out
        CHECK_CUDA(cudaMemcpyAsync(d_X, h_X.data(), X_bytes, cudaMemcpyHostToDevice, stream));
        for (int n = 0; n < num_modes; ++n) {
            if(A_sizes_bytes[n] > 0) {
                 CHECK_CUDA(cudaMemcpyAsync(d_A[n], h_A[n].data(), A_sizes_bytes[n], cudaMemcpyHostToDevice, stream));
            }
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        //printf("Input data copied to GPU.\n"); // Commented out

        // --- 4. Initialize Factors using HOSVD --- //
        try {
             initialize_factors_svd(d_X, X_dims, R_dims, d_A, cusolverH, cublasH, stream); // Use clamped R_dims
             // Initialize d_A_prev with the results from HOSVD
             //printf("Copying initial factors to d_A_prev...\n"); // Commented out
             for(int n=0; n<num_modes; ++n) {
                 if (d_A[n] != nullptr && d_A_prev[n] != nullptr) { // Check for null pointers if size is 0
                     size_t A_bytes = (size_t)X_dims[n] * R_dims[n] * sizeof(real);
                     CHECK_CUDA(cudaMemcpy(d_A_prev[n], d_A[n], A_bytes, cudaMemcpyDeviceToDevice));
                 }
             }
              CHECK_CUDA(cudaDeviceSynchronize()); // Sync after init and copy
        } catch (...) {
             // ... (add full cleanup from allocation block) ...
             //printf("Error during HOSVD initialization.\n"); // Commented out
             throw;
        }

        // --- 5. HOOI Iteration --- //
        //printf("Starting HOOI iterations...\n"); // Commented out
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
                 //printf("Iter: %4d, Change: %10.6e\n", iter, change); // Commented out
            }
        }

        //printf("HOOI iterations finished after %d iterations.\n", iter); // Commented out

        // --- 7. Compute Core Tensor G --- //
        //printf("Computing core tensor G...\n"); // Commented out
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
        //printf("Core tensor computed.\n"); // Commented out
        std::cout << "Core tensor computed." << std::endl;

        // --- 8. Copy Results Back to Host --- //
        //printf("Copying results back to host...\n"); // Commented out
        std::cout << "Copying results back to host..." << std::endl;
        if (G_size > 0) CHECK_CUDA(cudaMemcpyAsync(h_G.data(), d_G, G_size * sizeof(real), cudaMemcpyDeviceToHost, stream));
        for (int n = 0; n < num_modes; ++n) {
            if(A_sizes_bytes[n] > 0) {
                CHECK_CUDA(cudaMemcpyAsync(h_A[n].data(), d_A[n], A_sizes_bytes[n], cudaMemcpyDeviceToHost, stream));
            }
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        //printf("Finished copying results.\n"); // Commented out
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
    //printf("Cleaning up GPU resources...\n"); // Commented out
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

    //printf("Tucker decomposition host logic finished.\n"); // Commented out
    std::cout << "Tucker decomposition host logic finished." << std::endl;
} 