#include "host_logic.h" // Declares tucker_hooi_cuda, product
#include "tucker_kernels.h" // Declares launch_nModeProductKernel

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip> // Needed for setprecision

#include <cuda_runtime.h>
#include <cublas_v2.h>

// --- Add necessary error checking macros ---
#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\\n", __FILE__,       \
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
            fprintf(stderr, "cuBLAS Error in %s at line %d: %s\\n", __FILE__,   \
                    __LINE__, error_str);                                       \
            throw std::runtime_error("cuBLAS error");                             \
        }                                                                       \
    } while (0)
#endif
// --- Add CHECK_CUSOLVER if needed by host_logic internals ---

using real = float;

// --- Helper: Calculate Frobenius norm squared on device ---
real frobenius_norm_sq_device(cublasHandle_t handle, const real* d_vec, size_t size) {
    if (size == 0) return 0.0f;
    real norm_val = 0.0f; // cublasSnrm2 returns the norm, not norm^2
    CHECK_CUBLAS(cublasSnrm2(handle, static_cast<int>(size), d_vec, 1, &norm_val));
    return norm_val * norm_val; // Square the result
}

// --- Main Entry Point: Reconstruction Error Test ---
int main(int argc, char* argv[]) {

    // --- Argument Parsing ---
    const int num_required_args = 1 + 1 + 1 + 4;
    if (argc != num_required_args) {
        // Output nothing on incorrect usage for scripting
        // std::cerr << "Usage: " << argv[0] << " <max_iterations> <tolerance> <I1> <I2> <I3> <I4>" << std::endl;
        return 1;
    }
    int max_iterations;
    real tolerance;
    std::vector<int> X_dims(4);
    std::vector<int> R_dims_in = {8, 8, 8, 8}; // Fixed ranks for this executable
    try {
        max_iterations = std::stoi(argv[1]);
        tolerance = std::stof(argv[2]);
        for (int i = 0; i < 4; ++i) X_dims[i] = std::stoi(argv[3 + i]);
        if (max_iterations <=0 || tolerance <= 0 || std::any_of(X_dims.begin(), X_dims.end(), [](int d){return d <=0;})) {
             throw std::invalid_argument("Iterations, tolerance, dimensions must be positive.");
        }
    } catch (const std::exception& e) {
        // Output nothing on error
        // std::cerr << "Error parsing args: " << e.what() << std::endl;
        return 1;
    }

    // --- Clamp Ranks ---
    std::vector<int> R_dims_clamped = R_dims_in;
    for(int n=0; n<4; ++n) {
        R_dims_clamped[n] = std::min(R_dims_in[n], X_dims[n]);
         if (R_dims_clamped[n] <= 0) { /* Error */ return 1;}
    }

    // --- Generate Random Input Data ---
    long long X_size = product(X_dims);
    if (X_size <= 0) { /* Error */ return 1;}
    std::vector<real> h_X(X_size);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<real> distrib(0.0, 1.0);
    for(long long i = 0; i < X_size; ++i) h_X[i] = distrib(gen);

    // --- Allocate Host Memory for Results ---
    std::vector<std::vector<real>> h_A(4);
    std::vector<real> h_G;
    long long G_size_clamped = product(R_dims_clamped);
    try {
        for(int n=0; n<4; ++n) {
             size_t a_size = (size_t)X_dims[n] * R_dims_clamped[n];
             if (a_size > 0) h_A[n].resize(a_size);
        }
        if (G_size_clamped > 0) h_G.resize(G_size_clamped);
    } catch (const std::bad_alloc& e) {
       /* Error */ return 1;
    }

    // --- Call HOOI Logic ---
    try {
        // Suppress HOOI output for cleaner testing
        // std::cout << "\n--- Starting HOOI Computation ---" << std::endl;
        tucker_hooi_cuda(h_X, X_dims, h_A, h_G, R_dims_in, tolerance, max_iterations);
        // std::cout << "--- HOOI Computation Finished Successfully ---" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "HOOI computation error: " << e.what() << std::endl;
        return 1;
    }

    // --- Reconstruction & Error Calculation ---
    real relative_error = -1.0f;
    cudaStream_t stream = 0;
    cublasHandle_t cublasH = nullptr;
    real *d_G = nullptr, *d_recon_step1 = nullptr, *d_recon_step2 = nullptr, *d_X_orig = nullptr;
    std::vector<real*> d_A(4, nullptr);

    try {
        CHECK_CUBLAS(cublasCreate(&cublasH));
        CHECK_CUBLAS(cublasSetStream(cublasH, stream));

        // Allocate GPU memory
        if (G_size_clamped > 0) CHECK_CUDA(cudaMalloc(&d_G, G_size_clamped * sizeof(real)));
        size_t max_intermediate_size = 0;
        std::vector<int> current_dims = R_dims_clamped;
        long long recon_step_size = G_size_clamped; // Size starts at G
        for(int n=0; n<4; ++n) {
            long long dim_n_size = X_dims[n];
            recon_step_size = (recon_step_size / R_dims_clamped[n]) * dim_n_size; // Replace R_n with I_n
            max_intermediate_size = std::max(max_intermediate_size, (size_t)recon_step_size);
        }
        max_intermediate_size = std::max(max_intermediate_size, (size_t)X_size); // Ensure it's at least X_size
        max_intermediate_size *= sizeof(real); // Convert elements to bytes

        if (max_intermediate_size > 0) {
            CHECK_CUDA(cudaMalloc(&d_recon_step1, max_intermediate_size));
            CHECK_CUDA(cudaMalloc(&d_recon_step2, max_intermediate_size));
        }
        CHECK_CUDA(cudaMalloc(&d_X_orig, X_size * sizeof(real)));
        for(int n=0; n<4; ++n) if (!h_A[n].empty()) CHECK_CUDA(cudaMalloc(&d_A[n], h_A[n].size() * sizeof(real)));

        // Copy data H->D
        CHECK_CUDA(cudaMemcpyAsync(d_X_orig, h_X.data(), X_size*sizeof(real), cudaMemcpyHostToDevice, stream));
        if (d_G) CHECK_CUDA(cudaMemcpyAsync(d_G, h_G.data(), h_G.size()*sizeof(real), cudaMemcpyHostToDevice, stream));
        for(int n=0; n<4; ++n) if (d_A[n]) CHECK_CUDA(cudaMemcpyAsync(d_A[n], h_A[n].data(), h_A[n].size()*sizeof(real), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // Reconstruct: X_rec = G x_1 A1^T ... x_4 A4^T
        real* d_current_in = d_G;
        std::vector<int> current_recon_dims = R_dims_clamped;
        real* d_current_out = nullptr;
        for (int n = 0; n < 4; ++n) {
            std::vector<int> next_recon_dims = current_recon_dims;
            next_recon_dims[n] = X_dims[n];
            std::vector<int> factor_dims_n = {X_dims[n], R_dims_clamped[n]};
            bool is_last_step = (n == 3);
            d_current_out = is_last_step ? d_recon_step1 : ((n % 2 == 0) ? d_recon_step1 : d_recon_step2);

            if (!d_current_out && product(next_recon_dims) > 0) throw std::runtime_error("Null intermediate buffer.");

            long long current_in_size = product(current_recon_dims);
            long long factor_size = product(factor_dims_n);
            long long current_out_size = product(next_recon_dims);

            if (current_in_size > 0 && factor_size > 0) {
                 launch_nModeProductKernel(
                     d_current_in, d_current_out, d_A[n],
                     current_recon_dims, next_recon_dims, factor_dims_n,
                     n, true, stream);
            } else if (current_out_size > 0) {
                 CHECK_CUDA(cudaMemsetAsync(d_current_out, 0, current_out_size * sizeof(real), stream));
            }
            d_current_in = d_current_out;
            current_recon_dims = next_recon_dims;
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        real* d_X_rec = d_current_in;

        // Calculate error norm: || X - X_rec ||^2
        real alpha_axpy = -1.0f; // d_X_rec = d_X_rec - d_X_orig
        CHECK_CUBLAS(cublasSaxpy(cublasH, X_size, &alpha_axpy, d_X_orig, 1, d_X_rec, 1));
        real error_norm_sq = frobenius_norm_sq_device(cublasH, d_X_rec, X_size);
        real orig_norm_sq = frobenius_norm_sq_device(cublasH, d_X_orig, X_size);

        if (orig_norm_sq > 1e-12f) {
            relative_error = std::sqrt(error_norm_sq / orig_norm_sq);
        } else {
            relative_error = (error_norm_sq > 1e-12f) ? 1.0f : 0.0f;
        }

    } catch (const std::exception& e) {
        std::cerr << "Reconstruction/Error calculation failed: " << e.what() << std::endl;
        relative_error = -2.0f; // Indicate error occurred
    }

    // Cleanup
    if(d_X_orig) cudaFree(d_X_orig);
    if(d_recon_step1) cudaFree(d_recon_step1);
    if(d_recon_step2) cudaFree(d_recon_step2);
    if(d_G) cudaFree(d_G);
    for(auto p : d_A) if(p) cudaFree(p);
    if (cublasH) cublasDestroy(cublasH);

    // Final Output: Only the error value
    std::cout << std::scientific << std::setprecision(6) << relative_error << std::endl;

    return (relative_error >= 0 && relative_error < 1e-2) ? 0 : 1; // Return success/fail code
} 