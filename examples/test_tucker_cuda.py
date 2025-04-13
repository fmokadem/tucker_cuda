# /pub1/frank/tdcu/test_tucker_cuda.py
import numpy as np
import time
import sys
import os
import warnings

# Add build directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.abspath(os.path.join(script_dir, '../build'))
print(f"Adding build directory to sys.path: {build_dir}")
sys.path.insert(0, build_dir)

try:
    import tucker_cuda
except ImportError:
    print("ERROR: Failed to import the 'tucker_cuda' module.")
    print("Ensure project is compiled and module is in Python path.")
    exit(1)

# Try importing TensorLy for comparison only
try:
    import tensorly as tl
    from tensorly.decomposition import tucker as tl_tucker
    from tensorly.tenalg import multi_mode_dot
    _TENSORLY_AVAILABLE = True
except ImportError:
    print("WARNING: TensorLy library not found. Some checks will be skipped.")
    print("         Install it (`pip install tensorly`) for full testing.")
    _TENSORLY_AVAILABLE = False
    tl = None
    tl_tucker = None
    multi_mode_dot = None


# Configuration
DIMS_DEFAULT = [10, 11, 12, 13]
RANKS_DEFAULT = [3, 4, 5, 6]
TOLERANCE = 1e-5
MAX_ITER = 100
DTYPE = np.float32 # Match C++ 'real' type

RECONSTRUCTION_ERROR_DIFF_THRESHOLD = 1e-4
RECONSTRUCTION_ERROR_ABS_THRESHOLD = 1e-3
ORTHOGONALITY_THRESHOLD = 1e-3

if __name__ == "__main__":
    print("--- Starting Tucker Decomposition Test (using Python bindings) ---")

    dims = DIMS_DEFAULT
    ranks_req = RANKS_DEFAULT

    if len(sys.argv) == 1 + 4 + 4:
        try:
             print("Using dimensions and ranks from command line arguments.")
             dims = [int(x) for x in sys.argv[1:5]]
             ranks_req = [int(x) for x in sys.argv[5:9]]
             if any(d <= 0 for d in dims) or any(r <= 0 for r in ranks_req):
                  raise ValueError("Dimensions and ranks must be positive.")
        except ValueError as e:
             print(f"ERROR: Invalid command line arguments for dimensions/ranks: {e}", file=sys.stderr)
             print("Usage: python test_tucker_cuda.py [I1 I2 I3 I4 R1 R2 R3 R4]", file=sys.stderr)
             sys.exit(1)
    else:
        print("Using default dimensions and ranks.")

    ranks_clamped = [min(I, R) for I, R in zip(dims, ranks_req)]
    print(f"Testing with Dimensions: {dims}, Requested Ranks: {ranks_req}, Clamped Ranks: {ranks_clamped}")

    overall_passed = True
    rng = np.random.RandomState(1234) # Reproducible random data

    try:
        print(f"\nGenerating random tensor with shape {dims} and dtype {DTYPE}...")
        host_tensor_np = rng.standard_normal(dims).astype(DTYPE)
        tensor_norm = np.linalg.norm(host_tensor_np)
        if tensor_norm < 1e-12: tensor_norm = 1.0
        print(f"Input tensor norm: {tensor_norm:.6f}")

        print(f"\nRunning tucker_cuda.hooi (tol={TOLERANCE:.1e}, max_iter={MAX_ITER})...")
        start_time_cuda = time.time()
        try:
            cuda_core, cuda_factors = tucker_cuda.hooi(
                h_X_np=host_tensor_np,
                X_dims=dims,
                R_dims=ranks_req,
                tolerance=TOLERANCE,
                max_iterations=MAX_ITER
            )
            cuda_time = time.time() - start_time_cuda
            print(f"CUDA execution finished in {cuda_time:.4f} seconds.")
        except Exception as e:
            print(f"ERROR during tucker_cuda.hooi execution: {e}", file=sys.stderr)
            raise

        tl_core, tl_factors = None, None
        tl_time = -1.0
        if _TENSORLY_AVAILABLE:
            print(f"\nRunning tensorly.decomposition.tucker (init='svd', tol={TOLERANCE:.1e}, max_iter={MAX_ITER})...")
            start_time_tl = time.time()
            try:
                tl_core, tl_factors = tl_tucker(
                    host_tensor_np,
                    rank=ranks_req,
                    init='svd',
                    tol=TOLERANCE,
                    n_iter_max=MAX_ITER,
                    random_state=rng
                )
                tl_time = time.time() - start_time_tl
                print(f"TensorLy execution finished in {tl_time:.4f} seconds.")
            except Exception as e:
                 print(f"WARNING: TensorLy execution failed: {e}. Cannot compare results.", file=sys.stderr)
                 _TENSORLY_AVAILABLE = False
        else:
            print("\nSkipping TensorLy execution (library not found).")

        print("\n--- Verification ---")

        print("Checking output shapes...")
        correct_shapes = True
        expected_core_shape = tuple(ranks_clamped)
        if cuda_core.shape != expected_core_shape:
            print(f"FAIL: CUDA core shape mismatch. Expected {expected_core_shape}, Got {cuda_core.shape}", file=sys.stderr)
            correct_shapes = False
        if len(cuda_factors) != len(dims):
            print(f"FAIL: CUDA returned incorrect number of factors. Expected {len(dims)}, Got {len(cuda_factors)}", file=sys.stderr)
            correct_shapes = False
        else:
            for i, factor in enumerate(cuda_factors):
                expected_factor_shape = (dims[i], ranks_clamped[i])
                if factor.shape != expected_factor_shape:
                     print(f"FAIL: CUDA factor A{i+1} shape mismatch. Expected {expected_factor_shape}, Got {factor.shape}", file=sys.stderr)
                     correct_shapes = False
        if correct_shapes: print("PASS: Output shapes are correct.")
        overall_passed &= correct_shapes

        if multi_mode_dot is not None:
            print("Checking reconstruction error...")
            cuda_reconstruction = multi_mode_dot(cuda_core, cuda_factors, modes=list(range(len(dims))))
            cuda_recon_error = np.linalg.norm(host_tensor_np - cuda_reconstruction) / tensor_norm
            print(f"  Relative Reconstruction Error (CUDA): {cuda_recon_error:.6e}")

            if cuda_recon_error > RECONSTRUCTION_ERROR_ABS_THRESHOLD:
                 print(f"FAIL: CUDA reconstruction error ({cuda_recon_error:.6e}) exceeds threshold ({RECONSTRUCTION_ERROR_ABS_THRESHOLD:.1e})", file=sys.stderr)
                 overall_passed = False
            else:
                 print(f"PASS: CUDA reconstruction error within threshold {RECONSTRUCTION_ERROR_ABS_THRESHOLD:.1e}.")

            if _TENSORLY_AVAILABLE and tl_core is not None:
                tl_reconstruction = multi_mode_dot(tl_core, tl_factors, modes=list(range(len(dims))))
                tl_recon_error = np.linalg.norm(host_tensor_np - tl_reconstruction) / tensor_norm
                print(f"  Relative Reconstruction Error (TensorLy): {tl_recon_error:.6e}")

                error_diff = abs(cuda_recon_error - tl_recon_error)
                print(f"  Difference in Reconstruction Error: {error_diff:.6e}")
                if error_diff > RECONSTRUCTION_ERROR_DIFF_THRESHOLD:
                     print(f"WARNING: Difference vs TensorLy ({error_diff:.6e}) exceeds threshold {RECONSTRUCTION_ERROR_DIFF_THRESHOLD:.1e}.")
                     # Algorithm differences could account for this
                else:
                     print(f"PASS: Reconstruction error difference vs TensorLy within threshold {RECONSTRUCTION_ERROR_DIFF_THRESHOLD:.1e}.")
        else:
            print("Skipping reconstruction check (TensorLy not available).")

        print("Checking CUDA Factor Orthogonality...")
        ortho_passed = True
        for i, factor in enumerate(cuda_factors):
            I, R = factor.shape
            if I >= R:
                identity = np.eye(R, dtype=DTYPE)
                try:
                    factor_T_factor = factor.T @ factor
                    ortho_error = np.max(np.abs(factor_T_factor - identity))
                    if ortho_error > ORTHOGONALITY_THRESHOLD:
                        print(f"FAIL: Orthogonality check A{i+1}. Max deviation {ortho_error:.6e} > {ORTHOGONALITY_THRESHOLD:.1e}", file=sys.stderr)
                        ortho_passed = False
                    else:
                        print(f"  Orthogonality check A{i+1}: PASS (Max dev: {ortho_error:.6e})")
                except Exception as e:
                     print(f"FAIL: Error during orthogonality check for A{i+1}: {e}", file=sys.stderr)
                     ortho_passed = False
            else:
                print(f"  Orthogonality check A{i+1}: Skipped (Dimension {I} < Rank {R})")
        if ortho_passed: print("PASS: Orthogonality checks passed (where applicable).")
        overall_passed &= ortho_passed

    except Exception as e:
        print(f"\n--- TEST FAILED DUE TO UNEXPECTED ERROR ---", file=sys.stderr)
        print(f"Error encountered: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        overall_passed = False

    print("\n--- Test Summary ---")
    if overall_passed:
        print("RESULT: PASSED")
        sys.exit(0)
    else:
        print("RESULT: FAILED")
        sys.exit(1) 