#include "host_logic.h" // Declares tucker_hooi_cuda

#include <iostream>
#include <vector>
#include <string>
// #include <fstream> // No longer needed
#include <stdexcept>
#include <numeric>   // For std::accumulate
#include <random>    // For random initialization
#include <cmath>     // For std::sqrt
#include <algorithm> // For std::min, std::any_of
#include <iomanip>   // For std::setprecision

// Use the consistent data type (matches host_logic.cpp)
using real = float;

// --- REMOVED File I/O Helper Functions ---

// Helper function to calculate product of dimensions
// REMOVED: product function definition moved to host_logic or utility file

// --- Main Entry Point ---
int main(int argc, char* argv[]) {
    std::cout << R"(--- CUDA Tucker Decomposition (4D HOOI) - Test Program ---)" << std::endl;

    // --- Argument Parsing ---
    const int num_required_args = 1 + 1 + 1 + 4; // Exec name + iter + tol + 4 dims
    if (argc != num_required_args) {
        std::cerr << "Usage: " << argv[0] << " <max_iterations> <tolerance> <I1> <I2> <I3> <I4>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 100 1e-6 20 22 24 26" << std::endl;
        return 1;
    }

    int max_iterations = 100;
    real tolerance = 1e-6f;
    std::vector<int> X_dims(4);
    std::vector<int> R_dims_in = {8, 8, 8, 8}; // Fixed target ranks for simplicity

    try {
        max_iterations = std::stoi(argv[1]);
        tolerance = std::stof(argv[2]);
        for (int i = 0; i < 4; ++i) X_dims[i] = std::stoi(argv[3 + i]);

        if (max_iterations <= 0) {
             throw std::invalid_argument("Max iterations must be positive.");
        }
        if (tolerance <= 0) {
             throw std::invalid_argument("Tolerance must be positive.");
        }
        if (std::any_of(X_dims.begin(), X_dims.end(), [](int d){ return d <= 0; })) {
            throw std::invalid_argument("Dimensions (I1-I4) must be positive.");
        }

    } catch (const std::exception& e) {
        std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
        return 1;
    }

    // --- Clamp Ranks ---
    std::vector<int> R_dims_clamped = R_dims_in;
    for(int n=0; n<4; ++n) {
        R_dims_clamped[n] = std::min(R_dims_clamped[n], X_dims[n]);
        if (R_dims_clamped[n] <= 0) {
             std::cerr << "Error: Clamped rank for mode " << n << " is non-positive (" << R_dims_clamped[n]
                       << "). Check input dimensions and fixed ranks." << std::endl;
             return 1;
        }
    }

    // Print the configuration being used.
    std::cout << "\n--- Configuration ---" << std::endl;
    std::cout << "Input Dimensions    : [" << X_dims[0] << ", " << X_dims[1] << ", " << X_dims[2] << ", " << X_dims[3] << "]" << std::endl;
    std::cout << "Target Ranks (Fixed): [" << R_dims_in[0] << ", " << R_dims_in[1] << ", " << R_dims_in[2] << ", " << R_dims_in[3] << "]" << std::endl;
    std::cout << "Clamped Ranks       : [" << R_dims_clamped[0] << ", " << R_dims_clamped[1] << ", " << R_dims_clamped[2] << ", " << R_dims_clamped[3] << "]" << std::endl;
    std::cout << "Tolerance           : " << std::scientific << std::setprecision(1) << tolerance << std::defaultfloat << std::endl;
    std::cout << "Max Iterations      : " << max_iterations << std::endl;
    std::cout << "Data Type           : " << (std::is_same<real, float>::value ? "float" : "double") << std::endl;
    std::cout << "---------------------\n" << std::endl;


    // --- Generate Random Input Data ---
    std::vector<real> h_X;
    long long X_size = product(X_dims);
    if (X_size <= 0) {
        std::cerr << "Error: Calculated input tensor size is zero or negative." << std::endl;
        return 1;
    }
    try {
        std::cout << "Generating random input tensor (size " << X_size << ")..." << std::endl;
        h_X.resize(X_size);

        // Use a random number generator
        std::mt19937 gen(12345); // Mersenne Twister with fixed seed
        std::uniform_real_distribution<real> distrib(0.0, 1.0);

        for(long long i = 0; i < X_size; ++i) {
            h_X[i] = distrib(gen);
        }
         std::cout << "Input tensor generated." << std::endl;
    } catch (const std::bad_alloc& e) {
         std::cerr << "Error allocating memory for input tensor: " << e.what() << std::endl;
         return 1;
    } catch (const std::exception& e) {
         std::cerr << "Error generating random data: " << e.what() << std::endl;
         return 1;
    }


    // --- Allocate Host Memory for Results ---
    std::vector<std::vector<real>> h_A(4);
    std::vector<real> h_G;
    try {
        for(int n=0; n<4; ++n) {
            size_t A_size_clamped = (size_t)X_dims[n] * R_dims_clamped[n];
             if (A_size_clamped > 0) // Avoid zero-size allocation if rank or dim is 0 (already checked)
                 h_A[n].resize(A_size_clamped);
        }

        long long G_size_clamped = product(R_dims_clamped);
        if (G_size_clamped < 0) { // Should not happen due to earlier checks
            std::cerr << "Error: Clamped ranks result in negative core tensor size." << std::endl;
            return 1;
        }
        if (G_size_clamped > 0)
             h_G.resize(G_size_clamped);

    } catch (const std::bad_alloc& e) {
         std::cerr << "Error allocating host memory for results: " << e.what() << std::endl;
         return 1;
    }

    // --- Call HOOI Logic ---
    try {
        std::cout << "\n--- Starting HOOI Computation ---" << std::endl;
        // Pass the *original fixed* ranks R_dims_in, the function clamps internally
        tucker_hooi_cuda(h_X, X_dims, h_A, h_G, R_dims_in, tolerance, max_iterations);
        std::cout << "--- HOOI Computation Finished Successfully ---" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError during Tucker HOOI computation: " << e.what() << std::endl;
        return 1;
    }

    // --- Simple Output ---
    std::cout << "\n--- Results Summary ---" << std::endl;
    std::cout << "Core Tensor size computed: " << h_G.size() << " elements." << std::endl;
    for(int n=0; n<4; ++n) {
        std::cout << "Factor Matrix A" << n+1 << " size computed: " << h_A[n].size() << " elements ("
                  << X_dims[n] << " x " << R_dims_clamped[n] << ")." << std::endl;
    }
    std::cout << "\n--- Test Program Finished ---" << std::endl;

    return 0;
} 