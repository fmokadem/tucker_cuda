import numpy as np
import time
import warnings
from typing import List, Optional, Sequence, Tuple, Union, Literal

# Import the compiled C++/CUDA binding
try:
    import tucker_cuda
except ImportError as e:
    raise ImportError(
        "Could not import the compiled 'tucker_cuda' module. "
        "Ensure the project is built correctly using CMake and the resulting module "
        "(e.g., tucker_cuda.so) is in the Python path."
    ) from e

# Define the data type consistently (must match 'real' in C++)
DTYPE = np.float32

# Try importing tensorly for HOSVD initialization if requested
try:
    import tensorly as tl
    from tensorly.decomposition import tucker as tl_tucker # For SVD init
    _TENSORLY_AVAILABLE = True
except ImportError:
    _TENSORLY_AVAILABLE = False
    tl = None
    tl_tucker = None


def tucker(
    tensor: np.ndarray,
    rank: Union[int, Sequence[int]],
    n_iter_max: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Tucker decomposition via HOOI algorithm accelerated with CUDA.

    Backend uses internal HOSVD initialization.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor (must be 4D and float32).
    rank : int or Sequence[int]
        Desired target ranks for each mode.
    n_iter_max : int, optional
        Maximum number of HOOI iterations, by default 100.
    tol : float, optional
        Convergence tolerance, by default 1e-6.
    verbose : bool, optional
        Whether to print progress messages, by default False.

    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        (core_tensor, factors)
        Core tensor has shape (R1_c, ..., R4_c) where R_c is clamped rank.
        Factors is a list [A1, ..., A4] where An has shape (In, Rn_c).

    Raises
    ------
    TypeError, ValueError, NotImplementedError, RuntimeError
        If input validation fails or backend encounters an error.
    """
    if verbose:
        print("--- Running CUDA Tucker Decomposition ---")
        start_time = time.time()

    try:
        # --- Input Validation (moved from _validate_tucker_inputs) ---
        if not isinstance(tensor, np.ndarray):
            raise TypeError("Input tensor must be a NumPy ndarray.")

        if tensor.ndim != 4:
            raise NotImplementedError(
                f"The 'tucker_cuda' backend currently only supports 4-dimensional tensors, "
                f"but the input tensor has {tensor.ndim} dimensions."
            )
        ndim = 4
        X_dims = list(tensor.shape)

        if isinstance(rank, int):
            warnings.warn(f"Single rank passed: {rank}. Using this rank for all {ndim} modes.", UserWarning)
            R_dims = [rank] * ndim
        elif len(rank) == ndim:
            R_dims = list(rank)
        else:
            raise ValueError(
                f"Invalid 'rank': expected an integer or a sequence of length {ndim} "
                f"(tensor dimensions), but got {rank}."
            )

        if not all(isinstance(r, int) and r > 0 for r in R_dims):
            raise ValueError(f"Target ranks must be positive integers, but got {R_dims}")

        R_dims_in = list(R_dims) # Keep original request

        if tensor.dtype != DTYPE:
            warnings.warn(f"Input tensor dtype ({tensor.dtype}) differs from backend expectation ({DTYPE}). Casting to {DTYPE}.", UserWarning)
            tensor = tensor.astype(DTYPE)
        if not tensor.flags.c_contiguous:
            warnings.warn("Input tensor is not C-contiguous. Creating a C-contiguous copy.", UserWarning)
            tensor = np.ascontiguousarray(tensor)
        # --- End Validation ---

        if verbose:
             # Clamped ranks are computed internally but not needed here
             print(f"Input Shape: {X_dims}, Target Ranks: {R_dims_in}")
             print(f"Calling tucker_cuda.hooi with tolerance={tol:.1e}, max_iterations={n_iter_max}")

        # Call the backend C++/CUDA function (no initial factors passed)
        core_tensor, computed_factors = tucker_cuda.hooi(
            h_X_np=tensor,
            X_dims=X_dims,
            R_dims=R_dims_in,
            tolerance=tol,
            max_iterations=n_iter_max
        )

        if verbose:
            end_time = time.time()
            print(f"Tucker decomposition finished in {end_time - start_time:.4f} seconds.")
            print(f"Output Core Shape: {core_tensor.shape}")
            for i, f in enumerate(computed_factors):
                print(f"Output Factor {i} Shape: {f.shape}")

        return core_tensor, computed_factors

    except (TypeError, ValueError, NotImplementedError, RuntimeError) as e:
        raise e # Re-raise exceptions from validation or backend
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during Tucker decomposition: {e}") from e

# --- Placeholder for Partial Tucker ---
# The current CUDA backend only supports full Tucker decomposition (all modes).
# If partial Tucker support were added to the backend, this function could call it.
def partial_tucker(
    tensor: np.ndarray,
    rank: Union[int, Sequence[int]],
    modes: Sequence[int],
    n_iter_max: int = 100,
    init: Union[Literal['svd', 'random'], Tuple[np.ndarray, List[np.ndarray]], List[np.ndarray]] = 'random',
    tol: float = 1e-6,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Computes partial Tucker decomposition for specified modes.

    NOTE: This function is currently a placeholder as the 'tucker_cuda' backend
          only supports full (4D) decomposition.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor (must be 4D and float32).
    rank : int or Sequence[int]
        Desired target ranks for the modes specified in `modes`.
        If int, used for all modes in `modes`.
        If sequence, length must match `modes`.
    modes : Sequence[int]
        Modes along which to perform the decomposition.
    n_iter_max : int, optional
        Maximum number of HOOI iterations, by default 100.
    init : {'svd', 'random'} or List[np.ndarray] or Tuple[np.ndarray, List[np.ndarray]], optional
        Initialization method or initial factors for the specified modes, by default 'random'.
    tol : float, optional
        Convergence tolerance, by default 1e-6.
    random_state : int or np.random.RandomState, optional
        Seed for 'random' initialization, by default None.
    verbose : bool, optional
        Whether to print progress messages, by default False.

    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        (core_tensor, factors)
        Core tensor has ranks `rank` along `modes` and original size otherwise.
        Factors is a list corresponding to the `modes`.

    Raises
    ------
    NotImplementedError
        Always raised, as the backend doesn't support partial decomposition.
    """
    raise NotImplementedError(
        "Partial Tucker decomposition is not supported by the 'tucker_cuda' backend. "
        "Only full 4D decomposition is available via tucker()."
    )

# TODO: Add reconstruction function `tucker_to_tensor` if needed?
#       Requires n-mode product, could potentially call backend if exposed,
#       or use tensorly.tenalg.multi_mode_dot if tensorly is available.

# Example basic usage (can be run if module is imported)
if __name__ == '__main__':
    print("--- Testing tucker.py Module (Basic) ---")

    # Define small problem
    I1, I2, I3, I4 = 8, 9, 7, 10
    R1, R2, R3, R4 = 3, 4, 3, 5
    X_dims = [I1, I2, I3, I4]
    R_dims = [R1, R2, R3, R4]
    dtype = np.float32

    print(f"Dims: {X_dims}, Ranks: {R_dims}")

    # Create random tensor
    X = np.random.rand(*X_dims).astype(dtype)

    # Run Tucker decomposition using the CUDA backend via the Python wrapper
    print("\nRunning tucker() with init='random'...")
    try:
        core, factors = tucker(X, rank=R_dims, init='random', verbose=True, tol=1e-7, n_iter_max=50)
        print("Tucker decomposition successful!")
        print(f"Core shape: {core.shape}")
        print(f"Factor shapes: {[f.shape for f in factors]}")

        # Optional: Check reconstruction if tensorly is available
        if _TENSORLY_AVAILABLE:
             from tensorly.tenalg import multi_mode_dot
             X_rec = multi_mode_dot(core, factors, modes=[0,1,2,3])
             error = np.linalg.norm(X - X_rec) / np.linalg.norm(X)
             print(f"Relative reconstruction error: {error:.4e}")
        else:
            print("(Skipping reconstruction check, tensorly not installed)")

    except (RuntimeError, NotImplementedError, ValueError, TypeError) as e:
        print(f"Tucker decomposition failed: {e}")

    # Example of trying HOSVD init (requires tensorly)
    if _TENSORLY_AVAILABLE:
        print("\nRunning tucker() with init='svd'...")
        try:
            core_svd, factors_svd = tucker(X, rank=R_dims, init='svd', verbose=True, tol=1e-7, n_iter_max=50)
            print("Tucker decomposition (SVD init) successful!")
            print(f"Core shape: {core_svd.shape}")
            print(f"Factor shapes: {[f.shape for f in factors_svd]}")
            X_rec_svd = multi_mode_dot(core_svd, factors_svd, modes=[0,1,2,3])
            error_svd = np.linalg.norm(X - X_rec_svd) / np.linalg.norm(X)
            print(f"Relative reconstruction error (SVD init): {error_svd:.4e}")
        except (RuntimeError, NotImplementedError, ValueError, TypeError, ImportError) as e:
            print(f"Tucker decomposition (SVD init) failed: {e}")
    else:
        print("\n(Skipping tucker() test with init='svd', tensorly not installed)")

    # Example of calling partial_tucker (expected to fail)
    print("\nAttempting to call partial_tucker() (expected to fail)...")
    try:
        partial_tucker(X, rank=[R1,R2], modes=[0,1])
    except NotImplementedError as e:
        print(f"Successfully caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error from partial_tucker: {e}") 