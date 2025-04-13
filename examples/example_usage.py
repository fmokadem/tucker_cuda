import numpy as np
import time
import os
import sys

# Add build directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.abspath(os.path.join(script_dir, '../build'))
print(f"Adding build directory to sys.path: {build_dir}")
sys.path.insert(0, build_dir)

try:
    import tucker_cuda
except ImportError:
    print("Error: Failed to import the 'tucker_cuda' module.")
    print("Ensure project is compiled and module is in Python path.")
    exit(1)

# Configuration
I1, I2, I3, I4 = 20, 22, 24, 26
R1, R2, R3, R4 = 5, 6, 7, 8

X_dims = [I1, I2, I3, I4]
R_dims = [R1, R2, R3, R4]
tolerance = 1e-6
max_iterations = 100
DTYPE = np.float32 # Match 'real' in C++

if __name__ == "__main__":
    print("--- Running Python Example for CUDA Tucker HOOI ---")
    print(f"Input Dimensions (X_dims): {X_dims}")
    print(f"Target Ranks (R_dims): {R_dims}")
    print(f"Tolerance: {tolerance:.1e}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Data Type: {DTYPE}")

    print("\nGenerating random input tensor...")
    input_tensor_np = np.random.rand(*X_dims).astype(DTYPE)
    print(f"Input tensor shape: {input_tensor_np.shape}, dtype: {input_tensor_np.dtype}")

    print(f"\nCalling tucker_cuda.hooi(..., max_iterations={max_iterations}, tolerance={tolerance:.1e})...")
    start_time = time.time()
    try:
        core_tensor, computed_factors = tucker_cuda.hooi(
            h_X_np=input_tensor_np,
            X_dims=X_dims,
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

    print("\n--- Results ---")
    print(f"Computed Core Tensor shape: {core_tensor.shape}, dtype: {core_tensor.dtype}")
    print("Computed Factor Matrix shapes and types:")
    for i, factor in enumerate(computed_factors):
        print(f"  Factor A{i+1}: shape={factor.shape}, dtype={factor.dtype}")

    # Optional: Reconstruction check using tensorly
    try:
        import tensorly as tl
        from tensorly.tenalg import multi_mode_dot

        print("\nReconstructing tensor from computed factors...")
        reconstructed_tensor = multi_mode_dot(core_tensor, computed_factors, modes=[0,1,2,3])
        print(f"Reconstructed tensor shape: {reconstructed_tensor.shape}")

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