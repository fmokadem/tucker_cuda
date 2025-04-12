#include "host_logic.h" // Declares tucker_hooi_cuda

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <numeric> // std::iota for dummy init
#include <random>  // For random initialization
#include <cmath>   // For std::sqrt
#include <algorithm> // For std::min

// Use the consistent data type
using real = float;

// --- File I/O Helper Functions --- //

// Reads tensor data (expected to be raw contiguous `real` values)
// from a binary file into a std::vector.
// Performs size validation based on expected dimensions.
void read_tensor_from_file(const std::string& filename, const std::vector<int>& expected_shape, std::vector<real>& data) {
    long long expected_elements = product(expected_shape);
    if (expected_elements < 0) {
        throw std::runtime_error("Invalid expected shape (negative dimension?) for reading file: " + filename);
    }
    if (expected_elements == 0) {
        data.clear();
        return;
    }
    size_t expected_bytes = expected_elements * sizeof(real);

    std::ifstream infile(filename, std::ios::binary | std::ios::ate);
    if (!infile) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    std::streamsize size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    if (size != static_cast<std::streamsize>(expected_bytes)) {
         infile.close();
         throw std::runtime_error("File size mismatch for " + filename +\
                                  ". Expected " + std::to_string(expected_bytes) +\
                                  " bytes, but got " + std::to_string(size) + " bytes.");
    }

    data.resize(expected_elements);
    if (!infile.read(reinterpret_cast<char*>(data.data()), size)) {
        infile.close();
        throw std::runtime_error("Failed to read data from file: " + filename);
    }

    infile.close();
    std::cout << "Successfully read " << expected_elements << " elements (" << size << " bytes) from " << filename << std::endl;
}

// Writes tensor data from a std::vector to a binary file
// as raw contiguous `real` values.
// Performs size validation against the provided shape.
void write_tensor_to_file(const std::string& filename, const std::vector<int>& shape, const std::vector<real>& data) {
    long long expected_elements = product(shape);
     if (expected_elements < 0) {
          throw std::runtime_error("Invalid shape (negative dimension?) for writing file " + filename);
     }
     if (data.size() != static_cast<size_t>(expected_elements)) {
          throw std::runtime_error("Data size mismatch when writing file " + filename +\
                                   ". Expected " + std::to_string(expected_elements) +\
                                   " elements based on shape, but got " + std::to_string(data.size()) + " elements in vector.");
     }
     if (expected_elements == 0) {
         std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);
         if (!outfile) {
            throw std::runtime_error("Failed to open/truncate zero-byte file: " + filename);
         }
         outfile.close();
         return;
     }

    std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);
    if (!outfile) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    size_t bytes_to_write = data.size() * sizeof(real);
    outfile.write(reinterpret_cast<const char*>(data.data()), bytes_to_write);

    if (!outfile) {
         outfile.close();
         throw std::runtime_error("Failed to write data to file: " + filename);
    }

    outfile.close();
    std::cout << "Successfully wrote " << data.size() << " elements (" << bytes_to_write << " bytes) to " << filename << std::endl;
}


// --- Orthonormal Initialization Helper (CPU Version) --- //
// Initializes factor matrices (stored in `factors`) with random orthonormal columns
// using a basic Classical Gram-Schmidt process on the CPU.
// NOTE: This is primarily for providing an initial guess if SVD-based init isn't used.
//       Modified Gram-Schmidt or Householder QR would offer better numerical stability.
void initialize_orthogonal_factors(std::vector<std::vector<real>>& factors, const std::vector<int>& X_dims, const std::vector<int>& R_dims_clamped) {
    std::mt19937 gen(1234);
    std::normal_distribution<real> distrib(0.0, 1.0);

    for (size_t n = 0; n < factors.size(); ++n) {
        int I = X_dims[n];
        int R = R_dims_clamped[n];
        size_t expected_size = (size_t)I * R;

        if (expected_size == 0) {
            factors[n].clear();
            continue;
        }

        if (factors[n].size() != expected_size) {
             factors[n].resize(expected_size);
        }

        // Use temporary column-major representation for easier Gram-Schmidt.
        std::vector<std::vector<real>> matrix(R, std::vector<real>(I));
        for (int j = 0; j < R; ++j) {
            for (int i = 0; i < I; ++i) {
                matrix[j][i] = distrib(gen);
            }
        }

        // Basic Gram-Schmidt
        for (int j = 0; j < R; ++j) {
            for (int k = 0; k < j; ++k) {
                real dot_prod = 0.0;
                for (int i = 0; i < I; ++i) {
                    dot_prod += matrix[j][i] * matrix[k][i];
                }
                for (int i = 0; i < I; ++i) {
                    matrix[j][i] -= dot_prod * matrix[k][i];
                }
            }
            real norm_sq = 0.0;
            for (int i = 0; i < I; ++i) {
                norm_sq += matrix[j][i] * matrix[j][i];
            }
            real norm = std::sqrt(norm_sq);
            if (norm > 1e-10) {
                real inv_norm = 1.0f / norm;
                for (int i = 0; i < I; ++i) {
                    matrix[j][i] *= inv_norm;
                }
            } else {
                // Handle linearly dependent columns (rare with random normal init).
                // For simplicity here, fill with zeros, although this breaks orthogonality.
                std::cerr << "Warning: Linear dependency encountered during Gram-Schmidt for factor " << n+1 << ", column " << j << ". Column set to zero." << std::endl;
                for (int i = 0; i < I; ++i) {
                    matrix[j][i] = 0.0f;
                }
            }
        }

        // Copy orthonormal columns back to the flat host factor vector (h_A[n]),
        // converting to row-major format expected by CUDA kernels.
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < R; ++j) {
                factors[n][(size_t)i * R + j] = matrix[j][i];
            }
        }
         std::cout << "Initialized factor A" << n+1 << " with random orthogonal matrix (size " << I << "x" << R << ").";
    }
     std::cout << std::endl; // Add newline after initialization messages
}


// --- Main Entry Point --- //
int main(int argc, char* argv[]) {
    std::cout << R"(--- CUDA Tucker Decomposition (4D HOOI) ---)" << std::endl;

    // --- Argument Parsing --- //
    const int num_required_args = 1 + 6 + 4 + 4; // Exec name + files + dims + ranks
    const int num_optional_args = 2; // tolerance, max_iter
    if (argc < num_required_args) {
        std::cerr << "Usage: " << argv[0] << " <input.bin> <A1.bin> <A2.bin> <A3.bin> <A4.bin> <G.bin> \n"
                  << "       <I1> <I2> <I3> <I4> <R1> <R2> <R3> <R4> [tolerance] [max_iter]" << std::endl;
        return 1;
    }

    // Parse command line arguments for file paths, dimensions, ranks, and optional parameters.
    std::string input_tensor_file = argv[1];
    std::vector<std::string> output_A_files = {argv[2], argv[3], argv[4], argv[5]};
    std::string output_G_file = argv[6];

    std::vector<int> X_dims(4);
    std::vector<int> R_dims_in(4);
    real tolerance = 1e-5f; // Default tolerance
    int max_iterations = 100; // Default max iterations

    try {
        for (int i = 0; i < 4; ++i) X_dims[i] = std::stoi(argv[7 + i]);
        for (int i = 0; i < 4; ++i) R_dims_in[i] = std::stoi(argv[11 + i]);

        if (argc > num_required_args) {
            tolerance = std::stof(argv[num_required_args]);
        }
        if (argc > num_required_args + 1) {
            max_iterations = std::stoi(argv[num_required_args + 1]);
        }
        if (argc > num_required_args + num_optional_args) {
             std::cerr << "Warning: Extra command line arguments ignored." << std::endl;
        }

        if (any_of(X_dims.begin(), X_dims.end(), [](int d){ return d <= 0; }) ||
            any_of(R_dims_in.begin(), R_dims_in.end(), [](int r){ return r <= 0; })) {
            throw std::invalid_argument("Dimensions and ranks must be positive.");
        }

    } catch (const std::exception& e) {
        std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
        return 1;
    }

    // Print the configuration being used.
    std::cout << "--- Configuration ---" << std::endl;
    std::cout << "Input Tensor File : " << input_tensor_file << std::endl;
    for(int i=0; i<4; ++i) std::cout << "Output Factor A" << i+1 << " File: " << output_A_files[i] << std::endl;
    std::cout << "Output Core G File  : " << output_G_file << std::endl;
    std::cout << "Input Dimensions    : [" << X_dims[0] << ", " << X_dims[1] << ", " << X_dims[2] << ", " << X_dims[3] << "]" << std::endl;
    std::cout << "Requested Ranks     : [" << R_dims_in[0] << ", " << R_dims_in[1] << ", " << R_dims_in[2] << ", " << R_dims_in[3] << "]" << std::endl;
    std::cout << "Tolerance           : " << tolerance << std::endl;
    std::cout << "Max Iterations      : " << max_iterations << std::endl;
    std::cout << "---------------------" << std::endl;

    // --- Clamp Ranks (as done inside tucker_hooi_cuda) for allocation --- //
    // Clamp ranks here primarily to correctly size the host allocation for results.
    // The main `tucker_hooi_cuda` function also performs internal clamping.
    std::vector<int> R_dims_clamped = R_dims_in;
    for(int n=0; n<4; ++n) {
        R_dims_clamped[n] = std::min(R_dims_clamped[n], X_dims[n]);
    }

    // --- Load Input Data --- //
    std::vector<real> h_X;
    try {
        std::cout << "Loading input tensor..." << std::endl;
        read_tensor_from_file(input_tensor_file, X_dims, h_X);
    } catch (const std::exception& e) {
        std::cerr << "Error reading input tensor: " << e.what() << std::endl;
        return 1;
    }

    // --- Allocate Host Memory for Results --- //
    std::vector<std::vector<real>> h_A(4);
    std::vector<real> h_G;
    try {
        for(int n=0; n<4; ++n) {
            size_t factor_size = (size_t)X_dims[n] * R_dims_clamped[n];
            h_A[n].resize(factor_size); // Resize even if 0
        }

        long long core_size_ll = product(R_dims_clamped);
        if (core_size_ll < 0) throw std::runtime_error("Invalid clamped ranks resulted in negative core size.");
        size_t core_size = static_cast<size_t>(core_size_ll);
        h_G.resize(core_size); // Resize even if 0
    } catch (const std::bad_alloc& e) {
         std::cerr << "Error allocating host memory for results: " << e.what() << std::endl;
         return 1;
    }

    // Initialize with random orthonormal matrices using the CPU helper function.
    std::cout << "Initializing host factors randomly (using Gram-Schmidt)..." << std::endl;
    try {
        initialize_orthogonal_factors(h_A, X_dims, R_dims_clamped);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing orthogonal factors: " << e.what() << std::endl;
        return 1;
    }


    // --- Call the HOOI function --- //
    int exit_code = 0;
    try {
        std::cout << "\nStarting HOOI Computation..." << std::endl;
        tucker_hooi_cuda(h_X, X_dims, h_A, h_G, R_dims_in, tolerance, max_iterations); // Pass original ranks
        std::cout << "HOOI Computation Finished." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError during Tucker decomposition: " << e.what() << std::endl;
        // Cleanup might have occurred in tucker_hooi_cuda's catch block
        exit_code = 1;
    }

    // --- Write Output Data (only if computation succeeded) --- //
    if (exit_code == 0) {
        try {
            // Write the computed factor matrices and core tensor to binary files.
            std::cout << "\nWriting output factors and core tensor..." << std::endl;
            for(int n=0; n<4; ++n) {
                // Use clamped ranks for writing shapes
                std::vector<int> factor_shape = {X_dims[n], R_dims_clamped[n]};
                write_tensor_to_file(output_A_files[n], factor_shape, h_A[n]);
            }
            write_tensor_to_file(output_G_file, R_dims_clamped, h_G);
            std::cout << "Finished writing output files." << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error writing output files: " << e.what() << std::endl;
            exit_code = 1;
        }
    }

    if (exit_code == 0) {
        std::cout << "\n--- Tucker Decomposition Completed Successfully ---" << std::endl;
    } else {
         std::cout << "\n--- Tucker Decomposition Failed ---" << std::endl;
    }

    return exit_code;
} 