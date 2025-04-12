# /pub1/frank/tdcu/test_tucker_cuda.py
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import multi_mode_dot, mode_dot
from tensorly.base import unfold
import subprocess
import os
import struct
import time
import sys

# --- Configuration ---
CUDA_EXECUTABLE_PATH = "../build/tucker_app"
I1, I2, I3, I4 = 10, 11, 12, 13
R1, R2, R3, R4 = 3, 4, 5, 6

INPUT_TENSOR_FILE = "temp_input_tensor.bin"
OUTPUT_A1_FILE = "temp_output_A1.bin"
OUTPUT_A2_FILE = "temp_output_A2.bin"
OUTPUT_A3_FILE = "temp_output_A3.bin"
OUTPUT_A4_FILE = "temp_output_A4.bin"
OUTPUT_G_FILE = "temp_output_G.bin"

RECONSTRUCTION_ERROR_THRESHOLD = 1e-4
DTYPE = np.float32

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def get_abs_path(filename):
    return os.path.join(get_script_dir(), filename)

def save_tensor_to_file(tensor, filename):
    """Saves a NumPy tensor to a raw binary file (row-major)."""
    abs_filename = get_abs_path(filename)
    print(f"Saving tensor of shape {tensor.shape} to {abs_filename}...")
    tensor_contiguous = np.ascontiguousarray(tensor, dtype=DTYPE)
    try:
        with open(abs_filename, 'wb') as f:
            f.write(tensor_contiguous.tobytes())
        print(f"Saved {tensor_contiguous.nbytes} bytes.")
    except IOError as e:
        print(f"ERROR saving file {abs_filename}: {e}")
        raise

def load_tensor_from_file(filename, shape, dtype=DTYPE):
    """Loads a tensor from a raw binary file into a NumPy array (row-major)."""
    abs_filename = get_abs_path(filename)
    print(f"Loading tensor for expected shape {shape} from {abs_filename}...")
    expected_elements = np.prod(shape)
    if expected_elements < 0:
         raise ValueError(f"Invalid shape {shape} results in negative element count.")

    expected_bytes = expected_elements * np.dtype(dtype).itemsize
    if not os.path.exists(abs_filename):
         raise FileNotFoundError(f"Output file not found: {abs_filename}")

    try:
        actual_bytes = os.path.getsize(abs_filename)
        if actual_bytes != expected_bytes:
            raise ValueError(f"File size mismatch for {abs_filename}. Expected {expected_bytes} bytes, got {actual_bytes}.")

        if expected_bytes == 0:
             return np.zeros(shape, dtype=dtype)

        with open(abs_filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=dtype)

        if data.size != expected_elements:
             raise ValueError(f"Element count mismatch for {abs_filename}. Expected {expected_elements}, got {data.size}.")

        return data.reshape(shape, order='C')

    except IOError as e:
        print(f"ERROR loading file {abs_filename}: {e}")
        raise
    except ValueError as e:
        print(f"ERROR processing file {abs_filename}: {e}")
        raise

def cleanup_files(filenames):
    """Removes temporary files."""
    print("Cleaning up temporary files...")
    script_dir = get_script_dir()
    for f in filenames:
        abs_f = os.path.join(script_dir, f)
        if os.path.exists(abs_f):
            try:
                os.remove(abs_f)
            except OSError as e:
                print(f"Warning: Could not remove temporary file {abs_f}: {e}")

# --- Main Test Logic ---
if __name__ == "__main__":
    print("--- Starting Tucker Decomposition Test ---")
    script_dir = get_script_dir()
    print(f"Running test from directory: {script_dir}")

    try:
        import tensorly as tl
        print(f"Using TensorLy version: {tl.__version__}")
    except ImportError:
        print("ERROR: TensorLy library not found. Please install it (`pip install tensorly`).")
        sys.exit(1)

    # Override dims/ranks from command line if provided
    if len(sys.argv) == 1 + 4 + 4:
        try:
             print("Using dimensions and ranks from command line arguments.")
             I1, I2, I3, I4 = [int(x) for x in sys.argv[1:5]]
             R1, R2, R3, R4 = [int(x) for x in sys.argv[5:9]]
             if any(d <= 0 for d in [I1, I2, I3, I4]) or any(r <= 0 for r in [R1, R2, R3, R4]):
                  raise ValueError("Dimensions and ranks must be positive.")
             R1 = min(R1, I1); R2 = min(R2, I2); R3 = min(R3, I3); R4 = min(R4, I4)
        except ValueError as e:
             print(f"ERROR: Invalid command line arguments for dimensions/ranks: {e}")
             print("Usage: python test_tucker_cuda.py [I1 I2 I3 I4 R1 R2 R3 R4]")
             sys.exit(1)
    else:
        print("Using default dimensions and ranks.")

    dims = [I1, I2, I3, I4]
    ranks = [R1, R2, R3, R4]
    print(f"Testing with Dimensions: {dims}, Ranks: {ranks}")

    temp_files = [
        INPUT_TENSOR_FILE, OUTPUT_A1_FILE, OUTPUT_A2_FILE,
        OUTPUT_A3_FILE, OUTPUT_A4_FILE, OUTPUT_G_FILE
    ]
    overall_passed = True

    try:
        abs_executable_path = get_abs_path(CUDA_EXECUTABLE_PATH)
        if not os.path.exists(abs_executable_path):
            print(f"ERROR: CUDA executable not found at '{abs_executable_path}'")
            sys.exit(1)
        if not os.access(abs_executable_path, os.X_OK):
             print(f"ERROR: CUDA executable at '{abs_executable_path}' is not executable.")
             sys.exit(1)

        print(f"Generating random tensor with shape {dims} and dtype {DTYPE}...")
        np.random.seed(1234)
        host_tensor_np = np.random.random(dims).astype(DTYPE)
        tensor_norm = np.linalg.norm(host_tensor_np)
        if tensor_norm < 1e-12: tensor_norm = 1.0
        print(f"Input tensor norm: {tensor_norm:.6f}")

        save_tensor_to_file(host_tensor_np, INPUT_TENSOR_FILE)

        print(f"Running CUDA executable: {abs_executable_path}...")
        command = [
            abs_executable_path,
            INPUT_TENSOR_FILE,
            OUTPUT_A1_FILE, OUTPUT_A2_FILE, OUTPUT_A3_FILE, OUTPUT_A4_FILE,
            OUTPUT_G_FILE,
            str(I1), str(I2), str(I3), str(I4),
            str(R1), str(R2), str(R3), str(R4),
            str(1e-5), # Tolerance
            str(100)   # Max iterations
        ]
        print(f"Executing: {' '.join(command)}")
        start_time = time.time()
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        cuda_time = time.time() - start_time

        print("--- CUDA stdout ---")
        print(process.stdout.strip())
        print("-------------------")
        if process.returncode != 0:
            print(f"ERROR: CUDA executable failed with return code {process.returncode}")
            print("--- CUDA stderr ---")
            print(process.stderr.strip())
            print("-------------------")
            overall_passed = False
            raise RuntimeError("CUDA executable failed.")
        else:
            print("CUDA Execution successful.")

        print(f"CUDA execution time: {cuda_time:.4f} seconds")

        print("Loading results from CUDA executable...")
        cuda_A1 = load_tensor_from_file(OUTPUT_A1_FILE, (I1, R1))
        cuda_A2 = load_tensor_from_file(OUTPUT_A2_FILE, (I2, R2))
        cuda_A3 = load_tensor_from_file(OUTPUT_A3_FILE, (I3, R3))
        cuda_A4 = load_tensor_from_file(OUTPUT_A4_FILE, (I4, R4))
        cuda_G = load_tensor_from_file(OUTPUT_G_FILE, ranks)
        cuda_factors = [cuda_A1, cuda_A2, cuda_A3, cuda_A4]
        print("Successfully loaded CUDA results.")

        print("Running TensorLy Tucker decomposition...")
        start_time_tl = time.time()
        tl_core, tl_factors = tucker(
            host_tensor_np,
            rank=ranks,
            init='random', # Use random init for fair comparison with C++ random init
            random_state=1234, # Use same seed if possible
            tol=1e-5,
            n_iter_max=100
        )
        tl_time = time.time() - start_time_tl
        print("TensorLy execution finished.")
        print(f"TensorLy execution time: {tl_time:.4f} seconds")

        print("Comparing Reconstruction Errors...")
        cuda_reconstruction = multi_mode_dot(cuda_G, cuda_factors, modes=[0, 1, 2, 3])
        tl_reconstruction = multi_mode_dot(tl_core, tl_factors, modes=[0, 1, 2, 3])

        cuda_recon_error = np.linalg.norm(host_tensor_np - cuda_reconstruction) / tensor_norm
        tl_recon_error = np.linalg.norm(host_tensor_np - tl_reconstruction) / tensor_norm

        print(f"Relative Reconstruction Error (CUDA)  : {cuda_recon_error:.6e}")
        print(f"Relative Reconstruction Error (TensorLy): {tl_recon_error:.6e}")

        print("--- Verification ---")
        if cuda_recon_error > RECONSTRUCTION_ERROR_THRESHOLD:
             print(f"FAIL: CUDA reconstruction error ({cuda_recon_error:.6e}) exceeds threshold ({RECONSTRUCTION_ERROR_THRESHOLD:.6e})")
             overall_passed = False
        else:
             print(f"PASS: CUDA reconstruction error ({cuda_recon_error:.6e}) is within threshold.")

        # Orthogonality check
        print("Checking CUDA Factor Orthogonality:")
        ortho_failed = False
        for i, factor in enumerate(cuda_factors):
            I, R = factor.shape
            if I >= R:
                identity = np.eye(R, dtype=DTYPE)
                factor_T_factor = factor.T @ factor
                ortho_error = np.max(np.abs(factor_T_factor - identity))
                if ortho_error > 1e-3: # Allow some tolerance for numerical errors
                    print(f"  Orthogonality check (CUDA) A{i+1}.T @ A{i+1} vs Identity: False")
                    print(f"    Max deviation from Identity: {ortho_error:.6e}")
                    ortho_failed = True
                else:
                    print(f"  Orthogonality check (CUDA) A{i+1}.T @ A{i+1} vs Identity: True (Max dev: {ortho_error:.6e})")
            else:
                # If I < R, factor columns cannot be fully orthogonal.
                print(f"  Orthogonality check (CUDA) A{i+1}: Skipped (Dimension {I} < Rank {R})")

        if ortho_failed:
            print("WARNING: One or more CUDA factors failed the orthogonality check.")
            # Decide if this constitutes a failure for your criteria
            # overall_passed = False

    except Exception as e:
        print(f"\n--- TEST FAILED DUE TO ERROR ---")
        print(f"Error encountered: {e}")
        import traceback
        traceback.print_exc()
        overall_passed = False
    finally:
        # Cleanup temporary files
        cleanup_files(temp_files)

    print("--- Test Finished ---")
    if overall_passed:
        print("Overall Result: PASS")
        sys.exit(0)
    else:
        print("Overall Result: FAIL")
        sys.exit(1) 