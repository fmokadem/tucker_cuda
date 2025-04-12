import numpy as np
import time
import os

# Attempt to import the compiled C++/CUDA module
try:
    import tucker_cuda
except ImportError:
    print("Error: Failed to import the 'tucker_cuda' module.")
    print("Ensure project is compiled and module is in Python path.")
    exit(1)

# --- Configuration --- #
# Example tensor dimensions and ranks
I1, I2, I3, I4 = 20, 22, 24, 26 # Input dimensions
R1, R2, R3, R4 = 5, 6, 7, 8     # Target ranks

X_dims = [I1, I2, I3, I4]
R_dims = [R1, R2, R3, R4]

# HOOI Parameters
tolerance = 1e-6
max_iterations = 100

# Data type (must match 'real' defined in C++ code, usually float or double)
DTYPE = np.float32

# --- Helper Functions --- #

def generate_random_orthogonal_matrix(rows, cols, dtype=np.float32):
    """Generates a random matrix with orthonormal columns via QR."""
    if cols > rows:
        raise ValueError(f"Cannot generate {cols} orthonormal columns for {rows} rows.")
    A = np.random.rand(rows, cols).astype(dtype)
    Q, _ = np.linalg.qr(A)
    return np.ascontiguousarray(Q[:, :cols])

# --- Main Example --- #

if __name__ == "__main__":
    print("--- Running Python Example for CUDA Tucker HOOI ---")
    print(f"Input Dimensions (X_dims): {X_dims}")
    print(f"Target Ranks (R_dims): {R_dims}")
    print(f"Tolerance: {tolerance:.1e}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Data Type: {DTYPE}")

    # Calculate clamped ranks (for validating input factor shapes)
    R_dims_clamped = [min(I, R) for I, R in zip(X_dims, R_dims)]
    print(f"Clamped Ranks: {R_dims_clamped}")

    # 1. Generate Random Input Tensor
    print("\nGenerating random input tensor...")
    input_tensor_np = np.random.rand(*X_dims).astype(DTYPE)
    print(f"Input tensor shape: {input_tensor_np.shape}, dtype: {input_tensor_np.dtype}")

    # 2. Generate Initial Factor Guesses (Random Orthogonal)
    # The C++ code expects a list of NumPy arrays as initial guesses.
    print("Generating initial random orthogonal factor matrices...")
    initial_factors_np = []
    for n in range(4):
        rows = X_dims[n]
        cols = R_dims_clamped[n]
        try:
            factor = generate_random_orthogonal_matrix(rows, cols, dtype=DTYPE)
            initial_factors_np.append(factor)
            print(f"  Initial Factor A{n+1} shape: {factor.shape}, dtype: {factor.dtype}")
        except ValueError as e:
            print(f"Error generating orthogonal factor {n+1}: {e}")
            exit(1)

    # 3. Call the C++/CUDA HOOI function via pybind11
    print(f"\nCalling tucker_cuda.hooi(..., max_iterations={max_iterations}, tolerance={tolerance:.1e})...")
    start_time = time.time()
    try:
        # Call the bound function
        core_tensor, computed_factors = tucker_cuda.hooi(
            h_X_np=input_tensor_np,
            X_dims=X_dims,
            h_A_np_list=initial_factors_np,
            R_dims=R_dims,
            tolerance=tolerance,
            max_iterations=max_iterations
        )
        end_time = time.time()
        print(f"CUDA HOOI computation finished in {end_time - start_time:.4f} seconds.")

    except RuntimeError as e:
        print(f"\nError during CUDA HOOI execution: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn unexpected Python error occurred: {e}")
        exit(1)

    # 4. Print Results Information
    print("\n--- Results ---")
    print(f"Computed Core Tensor shape: {core_tensor.shape}, dtype: {core_tensor.dtype}")
    print("Computed Factor Matrix shapes and types:")
    for i, factor in enumerate(computed_factors):
        print(f"  Factor A{i+1}: shape={factor.shape}, dtype={factor.dtype}")

    # 5. (Optional) Reconstruction and Error Check
    # Requires a way to perform tucker_to_tensor (e.g., using tensorly or custom code)
    try:
        import tensorly as tl
        from tensorly.tenalg import multi_mode_dot

        print("\nReconstructing tensor from computed factors...")
        # Ensure factors are in the correct format if needed by tensorly
        reconstructed_tensor = multi_mode_dot(core_tensor, computed_factors, modes=[0,1,2,3])

        print(f"Reconstructed tensor shape: {reconstructed_tensor.shape}")

        # Calculate relative reconstruction error
        input_norm = np.linalg.norm(input_tensor_np)
        if input_norm > 1e-12:
            recon_error = np.linalg.norm(input_tensor_np - reconstructed_tensor) / input_norm
            print(f"Relative Reconstruction Error: {recon_error:.6e}")
        else:
            print("Input tensor norm is near zero, cannot calculate relative error reliably.")

    except ImportError:
        print("\n(Skipping reconstruction check: tensorly library not found.)")
    except Exception as e:
        print(f"\nError during reconstruction check: {e}")

    print("\n--- Example Finished ---") 