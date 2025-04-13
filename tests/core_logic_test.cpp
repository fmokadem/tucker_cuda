#include "host_logic.h" // Includes product(), tucker_hooi_cuda(), initialize_factors_svd()
#include "tucker_kernels.h" // Includes launch_nModeProductKernel()

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <limits>   // Required for numeric_limits
#include <tuple>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// --- Add necessary error checking macros ---
// (Copy definitions from host_logic.cpp or include a common header if one exists)
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
#ifndef CHECK_CUSOLVER
#define CHECK_CUSOLVER(call)                                                    \
    do {                                                                        \
        cusolverStatus_t status = call;                                         \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                \
            char error_str[128];                                                \
            snprintf(error_str, sizeof(error_str), "cuSOLVER error code %d",    \
                     static_cast<int>(status));                                 \
            fprintf(stderr, "cuSOLVER Error in %s at line %d: %s\\n", __FILE__, \
                    __LINE__, error_str);                                       \
            throw std::runtime_error("cuSOLVER error");                           \
        }                                                                       \
    } while (0)
#endif

using real = float; // Match the project's type

// --- Test Parameters ---
struct TestParams {
    std::string test_id;
    std::vector<int> dims;
    std::vector<int> ranks;
    enum DataType { RANDOM_UNIFORM, WITH_ZEROS, HIGH_LOW_VALS } data_type = DataType::RANDOM_UNIFORM;
};

// Define test cases
std::vector<TestParams> test_cases = {
    {"Small",           {6, 7, 8, 9},     {2, 3, 4, 5}},
    {"Medium",          {20, 22, 24, 26}, {8, 8, 8, 8}},
    {"LargeDimSmallRank",{50, 55, 60, 65}, {5, 6, 7, 8}},
    {"Clamp1",          {10, 11, 12, 13}, {12, 4, 5, 6}},
    {"Clamp2",          {10, 11, 12, 13}, {3, 15, 5, 6}},
    {"ClampAll",        {8, 9, 10, 11},   {10, 10, 10, 10}},
    {"RankEqDim",       {5, 6, 7, 8},     {5, 6, 7, 8}},
    {"Rank1Mode0",      {10, 11, 12, 13}, {1, 4, 5, 6}},
    {"Rank1Mode3",      {10, 11, 12, 13}, {3, 4, 5, 1}},
    {"Rank1All",        {10, 11, 12, 13}, {1, 1, 1, 1}},
    {"ThinDim1",        {2, 20, 22, 24},  {2, 5, 6, 7}},
    {"ThinDim3",        {20, 22, 24, 2},  {5, 6, 7, 2}},
    {"CubeSmall",       {10, 10, 10, 10}, {4, 4, 4, 4}},
    {"CubeMedium",      {30, 30, 30, 30}, {10, 10, 10, 10}},
    {"Mixed1",          {15, 30, 10, 40}, {5, 10, 5, 10}},
    {"Mixed2",          {40, 10, 30, 15}, {10, 5, 10, 5}},
    {"HighRankRatio",   {10, 12, 14, 16}, {8, 10, 12, 14}},
    {"LowRankRatio",    {50, 50, 50, 50}, {5, 5, 5, 5}},
    {"PrimeDims",       {17, 19, 23, 29}, {5, 6, 7, 8}},
    {"NearSquare",      {15, 16, 17, 18}, {7, 8, 9, 10}},
    // Additional Cases
    {"WithZeros",       {20, 20, 20, 20}, {5, 5, 5, 5}, TestParams::DataType::WITH_ZEROS},
    {"HighLowVals",     {15, 15, 15, 15}, {6, 6, 6, 6}, TestParams::DataType::HIGH_LOW_VALS},
    {"SmallZeros",      {6, 7, 8, 9},     {2, 3, 4, 5}, TestParams::DataType::WITH_ZEROS},
    {"MediumHighLow",   {20, 22, 24, 26}, {8, 8, 8, 8}, TestParams::DataType::HIGH_LOW_VALS},
    {"ClampZeros",      {8, 9, 10, 11},   {10, 10, 10, 10}, TestParams::DataType::WITH_ZEROS},
};


// --- Helper Functions ---

// Generate tensor data based on type
void generate_tensor_data(std::vector<real>& vec, const TestParams::DataType type, unsigned int seed) {
    std::mt19937 gen(seed);
    if (type == TestParams::DataType::WITH_ZEROS) {
        std::uniform_real_distribution<real> distrib(0.0, 1.0);
        std::bernoulli_distribution bernoulli(0.7); // ~70% non-zero
        for (real& val : vec) {
            val = bernoulli(gen) ? distrib(gen) : 0.0f;
        }
    } else if (type == TestParams::DataType::HIGH_LOW_VALS) {
         std::uniform_real_distribution<real> distrib(-1000.0, 1000.0); // Wider range
         for (real& val : vec) {
            val = distrib(gen);
        }
    }
    else { // Default: RANDOM_UNIFORM
        std::uniform_real_distribution<real> distrib(0.0, 1.0);
        for (real& val : vec) {
            val = distrib(gen);
        }
    }
}

// Check orthogonality of a host factor matrix
double check_orthogonality(const std::vector<real>& h_factor, int rows, int cols) {
    if (rows < cols || cols <= 0) return 0.0; // Not applicable or empty

    std::vector<real> identity(cols * cols, 0.0f);
    for(int i = 0; i < cols; ++i) identity[i * cols + i] = 1.0f;

    std::vector<real> factorTfactor(cols * cols);

    // Simple CPU Matrix Multiply: C = A^T * A
    for(int i = 0; i < cols; ++i) { // Row of C (col of A)
        for(int j = 0; j < cols; ++j) { // Col of C (col of A)
            double sum = 0.0;
            for(int k = 0; k < rows; ++k) { // Row of A
                // Factor is row-major: rows x cols. Access A(k, i) and A(k, j)
                 sum += static_cast<double>(h_factor[k * cols + i]) * static_cast<double>(h_factor[k * cols + j]);
            }
             factorTfactor[i * cols + j] = static_cast<real>(sum);
        }
    }

    double max_abs_diff = 0.0;
    for(size_t i = 0; i < factorTfactor.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, (double)std::abs(factorTfactor[i] - identity[i]));
    }
    return max_abs_diff;
}

// Calculate Frobenius norm squared on device
real frobenius_norm_sq_device(cublasHandle_t handle, const real* d_vec, size_t size) {
    if (size == 0) return 0.0f;
    real norm_val = 0.0f;
    CHECK_CUBLAS(cublasSnrm2(handle, static_cast<int>(size), d_vec, 1, &norm_val));
    return norm_val * norm_val;
}

// Reconstruct tensor on device: X_rec = G x_1 A1^T ... x_4 A4^T
// Places result in d_output (must be pre-allocated to X_size)
void reconstruct_tensor_device(
    const real* d_G, const std::vector<int>& R_dims,
    const std::vector<real*>& d_A, const std::vector<int>& X_dims,
    real* d_output, real* d_intermediate1, real* d_intermediate2,
    cudaStream_t stream)
{
    real* d_current_in = const_cast<real*>(d_G); // Start with core tensor
    std::vector<int> current_recon_dims = R_dims;
    real* d_current_out = nullptr;

    long long current_size = product(R_dims);
    if (current_size <= 0) { // Handle empty core tensor case
         long long out_size = product(X_dims);
         if (out_size > 0) CHECK_CUDA(cudaMemsetAsync(d_output, 0, out_size * sizeof(real), stream));
         return;
    }


    for (int n = 0; n < 4; ++n) {
        std::vector<int> next_recon_dims = current_recon_dims;
        next_recon_dims[n] = X_dims[n];
        std::vector<int> factor_dims_n = {X_dims[n], R_dims[n]}; // A_n is I_n x R_n

        bool is_last_step = (n == 3);
        // Use output buffer directly on last step, alternate intermediates otherwise
        d_current_out = is_last_step ? d_output : ((n % 2 == 0) ? d_intermediate1 : d_intermediate2);

        if (!d_current_out && product(next_recon_dims) > 0) {
            throw std::runtime_error("Null intermediate buffer during reconstruction.");
        }

        long long current_in_size = product(current_recon_dims);
        long long factor_size = product(factor_dims_n);
        long long current_out_size = product(next_recon_dims);

        if (current_in_size > 0 && factor_size > 0 && d_A[n] != nullptr) {
             launch_nModeProductKernel(
                 d_current_in, d_current_out, d_A[n],
                 current_recon_dims, next_recon_dims, factor_dims_n,
                 n, true, stream); // Use Transpose
        } else if (current_out_size > 0) {
            // If input or factor is zero-sized, output is zero
             CHECK_CUDA(cudaMemsetAsync(d_current_out, 0, current_out_size * sizeof(real), stream));
        }

        d_current_in = d_current_out;
        current_recon_dims = next_recon_dims;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream)); // Ensure reconstruction is complete
}


// --- Main Test Runner ---
int main() {
    cublasHandle_t cublasH = nullptr;
    cusolverDnHandle_t cusolverH = nullptr;
    cudaStream_t stream = 0;

    std::cout << "Initializing CUDA/CUBLAS/CUSOLVER..." << std::endl;
    try {
        CHECK_CUBLAS(cublasCreate(&cublasH));
        CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
        // Optional: set stream on handles if needed by host_logic
        // CHECK_CUBLAS(cublasSetStream(cublasH, stream));
        // CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, stream));
    } catch (const std::exception& e) {
        std::cerr << "FATAL: Failed to initialize CUDA handles: " << e.what() << std::endl;
        return 1;
    }
     std::cout << "Initialization complete.\n" << std::endl;

    bool overall_success = true;

    // Set output precision
    std::cout << std::scientific << std::setprecision(6);

    for (const auto& tc : test_cases) {
        std::cout << "--- Test Case: " << tc.test_id << " ---" << std::endl;
        std::cout << "    Dims: [";
        for(size_t i=0; i<tc.dims.size(); ++i) std::cout << tc.dims[i] << (i==tc.dims.size()-1 ? "" : ",");
        std::cout << "], Ranks: [";
        for(size_t i=0; i<tc.ranks.size(); ++i) std::cout << tc.ranks[i] << (i==tc.ranks.size()-1 ? "" : ",");
        std::cout << "], Data: " << (int)tc.data_type << std::endl;

        // Test variables
        double hosvd_ortho_err = std::numeric_limits<double>::quiet_NaN();
        double hooi_ortho_err = std::numeric_limits<double>::quiet_NaN();
        real hooi_recon_err = std::numeric_limits<real>::quiet_NaN();

        // Host memory
        std::vector<real> h_X;
        std::vector<std::vector<real>> h_A_hosvd(4);
        std::vector<std::vector<real>> h_A_hooi(4);
        std::vector<real> h_G_hooi;

        // Device memory
        real* d_X = nullptr;
        std::vector<real*> d_A(4, nullptr);
        real* d_G = nullptr;
        real* d_recon_step1 = nullptr;
        real* d_recon_step2 = nullptr;

        try {
            // --- Setup ---
            std::vector<int> X_dims = tc.dims;
            std::vector<int> R_dims_in = tc.ranks;
            std::vector<int> R_dims_clamped = R_dims_in;
            for(int n=0; n<4; ++n) R_dims_clamped[n] = std::min(R_dims_in[n], X_dims[n]);

            long long X_size = product(X_dims);
            if (X_size <= 0) { std::cout << "    Skipping: Zero input size." << std::endl; continue; }
            h_X.resize(X_size);
            generate_tensor_data(h_X, tc.data_type, 789 + (int)tc.data_type); // Different seed per data type

            long long G_size_clamped = product(R_dims_clamped);
            std::vector<size_t> A_sizes(4);

            // Allocate device memory
            CHECK_CUDA(cudaMalloc(&d_X, X_size * sizeof(real)));
            for(int n=0; n<4; ++n) {
                A_sizes[n] = (size_t)X_dims[n] * R_dims_clamped[n];
                if (A_sizes[n] > 0) CHECK_CUDA(cudaMalloc(&d_A[n], A_sizes[n] * sizeof(real)));
            }
             // Allocate intermediates needed for reconstruction (potentially large)
             size_t max_intermediate_size_bytes = 0;
             long long recon_step_size = G_size_clamped > 0 ? G_size_clamped : 1; // Avoid 0 start if G is empty
             for(int n=0; n<4; ++n) {
                 long long dim_n_size = X_dims[n];
                 long long rank_n_size = R_dims_clamped[n] > 0 ? R_dims_clamped[n] : 1;
                 recon_step_size = (recon_step_size / rank_n_size) * dim_n_size;
                 max_intermediate_size_bytes = std::max(max_intermediate_size_bytes, (size_t)recon_step_size);
             }
             max_intermediate_size_bytes = std::max(max_intermediate_size_bytes, (size_t)X_size);
             max_intermediate_size_bytes *= sizeof(real);

             if (max_intermediate_size_bytes > 0) {
                 CHECK_CUDA(cudaMalloc(&d_recon_step1, max_intermediate_size_bytes));
                 CHECK_CUDA(cudaMalloc(&d_recon_step2, max_intermediate_size_bytes));
             }


            // --- Test HOSVD Initialization Orthogonality ---
            /*
            std::cout << "    Running HOSVD init..." << std::flush;
            CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), X_size * sizeof(real), cudaMemcpyHostToDevice));
            tucker_hooi_cuda(h_X, X_dims, h_A_hooi, h_G_hooi, R_dims_in, 1e-6f, 100);
            std::cout << " done." << std::endl;

            hosvd_ortho_err = 0.0;
            for(int n=0; n<4; ++n) {
                if (A_sizes[n] > 0) {
                    h_A_hosvd[n].resize(A_sizes[n]);
                    CHECK_CUDA(cudaMemcpy(h_A_hosvd[n].data(), d_A[n], A_sizes[n] * sizeof(real), cudaMemcpyDeviceToHost));
                    if (std::any_of(h_A_hosvd[n].begin(), h_A_hosvd[n].end(), [](real v){ return std::isnan(v) || std::isinf(v); })) {
                         hosvd_ortho_err = std::numeric_limits<double>::infinity(); // Mark as error
                         break;
                    }
                    hosvd_ortho_err = std::max(hosvd_ortho_err,
                        check_orthogonality(h_A_hosvd[n], X_dims[n], R_dims_clamped[n]));
                }
            }
             std::cout << "    HOSVD Ortho Error (Max): " << hosvd_ortho_err << std::endl;
            */
            // --- Test HOOI ---
             std::cout << "    Running HOOI..." << std::flush;
             // Prepare host outputs for HOOI function
             for(int n=0; n<4; ++n) if(A_sizes[n] > 0) h_A_hooi[n].resize(A_sizes[n]); else h_A_hooi[n].clear();
             if(G_size_clamped > 0) h_G_hooi.resize(G_size_clamped); else h_G_hooi.clear();

             // Call HOOI (uses HOSVD internally again, overwrites d_A, returns results in h_A_hooi, h_G_hooi)
             tucker_hooi_cuda(h_X, X_dims, h_A_hooi, h_G_hooi, R_dims_in, 1e-6f, 100);
             std::cout << " done." << std::endl;

             // Check HOOI Orthogonality
             hooi_ortho_err = 0.0;
             for(int n=0; n<4; ++n) {
                 if (A_sizes[n] > 0) {
                     if (std::any_of(h_A_hooi[n].begin(), h_A_hooi[n].end(), [](real v){ return std::isnan(v) || std::isinf(v); })) {
                         hooi_ortho_err = std::numeric_limits<double>::infinity(); break;
                     }
                     hooi_ortho_err = std::max(hooi_ortho_err,
                         check_orthogonality(h_A_hooi[n], X_dims[n], R_dims_clamped[n]));
                 }
             }
             std::cout << "    HOOI Ortho Error (Max):  " << hooi_ortho_err << std::endl;

             // Check HOOI Reconstruction Error
             if (G_size_clamped > 0) {
                 CHECK_CUDA(cudaMalloc(&d_G, G_size_clamped * sizeof(real)));
             } else {
                 d_G = nullptr;
             }
             // Copy final HOOI results to device for reconstruction check
             if (d_G) CHECK_CUDA(cudaMemcpy(d_G, h_G_hooi.data(), G_size_clamped*sizeof(real), cudaMemcpyHostToDevice));
             for(int n=0; n<4; ++n) if(A_sizes[n]>0) CHECK_CUDA(cudaMemcpy(d_A[n], h_A_hooi[n].data(), A_sizes[n]*sizeof(real), cudaMemcpyHostToDevice));

             // Perform reconstruction G x A^T ...
             reconstruct_tensor_device(d_G, R_dims_clamped, d_A, X_dims,
                                       d_recon_step1, d_recon_step1, d_recon_step2, // Use step1 as output
                                       stream);
             real* d_X_rec = d_recon_step1; // Result is now in step1

             // Calculate error ||X - X_rec|| / ||X||
             real alpha_axpy = -1.0f; // d_X_rec = d_X_rec - d_X (original X still on device)
             CHECK_CUBLAS(cublasSaxpy(cublasH, X_size, &alpha_axpy, d_X, 1, d_X_rec, 1));
             real error_norm_sq = frobenius_norm_sq_device(cublasH, d_X_rec, X_size);
             real orig_norm_sq = frobenius_norm_sq_device(cublasH, d_X, X_size);

             if (orig_norm_sq > 1e-12f) {
                 hooi_recon_err = std::sqrt(error_norm_sq / orig_norm_sq);
             } else {
                 hooi_recon_err = (error_norm_sq > 1e-12f) ? 1.0f : 0.0f;
             }
             std::cout << "    HOOI Recon Error (Rel):  " << hooi_recon_err << std::endl;


        } catch (const std::exception& e) {
            std::cerr << "    ERROR encountered: " << e.what() << std::endl;
            overall_success = false; // Mark failure if any test case throws
        }

        // Cleanup device memory for this test case
        if(d_recon_step1) cudaFree(d_recon_step1); d_recon_step1 = nullptr;
        if(d_recon_step2) cudaFree(d_recon_step2); d_recon_step2 = nullptr;
        if(d_G) cudaFree(d_G); d_G = nullptr;
        for(auto& p : d_A) if(p) cudaFree(p); d_A = std::vector<real*>(4, nullptr);
        if(d_X) cudaFree(d_X); d_X = nullptr;

        std::cout << "------------------------------------" << std::endl;
    } // End loop over test cases


    // Cleanup global handles
    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (cublasH) cublasDestroy(cublasH);

    std::cout << "\nTest suite finished." << std::endl;
    return overall_success ? 0 : 1; // Indicate failure if any exception occurred
} 