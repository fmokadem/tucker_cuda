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

# Try importing tensorly for HOSVD initialization, but don't fail if not present
# We only need it if the user explicitly requests init='svd'
try:
    import tensorly as tl
    from tensorly.decomposition import tucker as tl_tucker # For SVD init
    _TENSORLY_AVAILABLE = True
except ImportError:
    _TENSORLY_AVAILABLE = False
    tl = None
    tl_tucker = None


def _validate_tucker_inputs(
    tensor: np.ndarray,
    rank: Union[int, Sequence[int]],
    modes: Optional[Sequence[int]] = None,
    n_iter_max: int = 100,
    init: Union[Literal['svd', 'random'], Tuple[np.ndarray, List[np.ndarray]], List[np.ndarray]] = 'random',
    tol: float = 1e-5,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    verbose: bool = False,
    return_errors: bool = False
) -> Tuple[np.ndarray, List[int], List[int], List[int], List[np.ndarray]]:
    """Internal helper to validate inputs and prepare initial factors.

    Checks backend limitations (4D only).
    Returns: (tensor_contig, X_dims, R_dims_in, R_dims_clamped, initial_factors)
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError("Input tensor must be a NumPy ndarray.")

    # --- Backend Limitation Check ---
    if tensor.ndim != 4:
        raise NotImplementedError(
            f"The 'tucker_cuda' backend currently only supports 4-dimensional tensors, "
            f"but the input tensor has {tensor.ndim} dimensions."
        )
    ndim = 4
    X_dims = list(tensor.shape)

    # --- Rank Handling ---
    if isinstance(rank, int):
        # If single int, repeat for all modes
        message = f"Single rank passed: {rank}. Using this rank for all {ndim} modes."
        warnings.warn(message, UserWarning)
        R_dims = [rank] * ndim
    elif len(rank) == ndim:
        R_dims = list(rank)
    else:
        raise ValueError(
            f"Invalid 'rank': expected an integer or a sequence of length {ndim} "
            f"(tensor dimensions), but got {rank}."
        )

    # Check ranks are positive integers
    if not all(isinstance(r, int) and r > 0 for r in R_dims):
        raise ValueError(f"Target ranks must be positive integers, but got {R_dims}")

    # Keep original requested ranks
    R_dims_in = list(R_dims) # Store original request

    # Clamp ranks (required for initial factor generation shape)
    R_dims_clamped = [min(I, R) for I, R in zip(X_dims, R_dims)]

    # --- Parameter Validation for CUDA backend --- 
    if return_errors:
        raise NotImplementedError("'return_errors=True' is not supported by the tucker_cuda backend.")
    if modes is not None:
        # This validation is primarily for partial_tucker, but check here too
        if sorted(list(set(modes))) != sorted(modes) or any(m < 0 or m >= ndim for m in modes):
             raise ValueError(f"Invalid 'modes': {modes}. Must be unique modes within [0, {ndim-1}] ")
        # For full tucker, rank must be sequence of length ndim or single int
        # Validation already done above
        # if len(R_dims) != len(modes):
        #     raise ValueError("Number of ranks must match number of modes for partial Tucker.")

    # Ensure tensor dtype matches backend expectation
    if tensor.dtype != DTYPE:
        warnings.warn(f"Input tensor dtype ({tensor.dtype}) differs from backend expectation ({DTYPE}). Casting to {DTYPE}.", UserWarning)
        tensor = tensor.astype(DTYPE)
    if not tensor.flags.c_contiguous:
        warnings.warn("Input tensor is not C-contiguous. Creating a C-contiguous copy.", UserWarning)
        tensor = np.ascontiguousarray(tensor)

    # --- Initialization --- #
    initial_factors: List[np.ndarray] = [] # Ensure it's defined
    if isinstance(init, str):
        init_method = init.lower()
        if init_method == 'random':
            if verbose:
                print("Initializing factors randomly (orthogonalized via QR)")
            rng = np.random.RandomState(random_state) if not isinstance(random_state, np.random.RandomState) else random_state
            # initial_factors = [] # Already defined
            for n in range(ndim):
                rows = X_dims[n]
                cols = R_dims_clamped[n]
                # Generate random matrix
                A = rng.standard_normal((rows, cols)).astype(DTYPE)
                # QR decomposition for orthogonalization
                if rows >= cols:
                    Q, _ = np.linalg.qr(A)
                    # Q has shape (rows, rows), take first 'cols' columns
                    factor = np.ascontiguousarray(Q[:, :cols])
                else:
                    # More columns than rows - cannot generate full orthonormal matrix
                    # Generate random and normalize columns (less ideal but provides guess)
                    # Note: HOOI itself ensures orthogonality later via SVD
                    warnings.warn(f"Mode {n}: Rank ({cols}) > Dimension ({rows}). Generating random columns, not strictly orthogonal.", UserWarning)
                    factor = A / np.linalg.norm(A, axis=0)[np.newaxis, :]
                    factor = np.ascontiguousarray(factor)

                initial_factors.append(factor)

        elif init_method == 'svd':
            if verbose:
                print("Initializing factors via HOSVD (using TensorLy if available)")
            if not _TENSORLY_AVAILABLE:
                raise ImportError("Initialization method 'svd' requires the TensorLy library, which was not found.")
            # Use TensorLy's tucker with HOSVD init (0 iterations) just for factors
            try:
                _, hosvd_factors = tl_tucker(
                    tensor, rank=R_dims_in, init='svd', n_iter_max=0, # Get HOSVD factors
                    # modes=modes if modes else list(range(ndim)), # Use clamped ranks for HOSVD?
                    normalize_factors=True # Standard HOSVD
                )
                # Ensure factors are correct dtype and contiguous, and correct shape
                initial_factors = []
                for n in range(ndim):
                     # HOSVD factors might not match clamped rank if I < R
                     factor_temp = np.ascontiguousarray(hosvd_factors[n], dtype=DTYPE)
                     if factor_temp.shape != (X_dims[n], R_dims_clamped[n]):
                         # This can happen if R_dims_in[n] > X_dims[n]
                         # Take appropriate slice
                         factor = factor_temp[:, :R_dims_clamped[n]]
                     else:
                         factor = factor_temp
                     initial_factors.append(np.ascontiguousarray(factor))

            except Exception as e:
                raise RuntimeError(f"TensorLy HOSVD initialization failed: {e}") from e

        else:
            raise ValueError(f"Initialization method '{init}' not recognized. Use 'random', 'svd', or provide initial factors.")

    elif isinstance(init, (list, tuple)):
        temp_factors: List[np.ndarray] = []
        # Check if it's a list of factors or a (core, factors) tuple
        if len(init) == ndim and isinstance(init[0], np.ndarray):
            if verbose:
                print("Using provided list of factors for initialization.")
            temp_factors = init
        elif len(init) == 2 and isinstance(init[0], np.ndarray) and isinstance(init[1], (list, tuple)):
             if verbose:
                 print("Using provided (core, factors) tuple for initialization.")
             temp_factors = init[1]
        else:
            raise TypeError("Invalid 'init': If list/tuple, must be list of factors or (core, factors) tuple.")

        if len(temp_factors) != ndim:
             raise ValueError(f"Invalid 'init': Expected {ndim} factor matrices, got {len(temp_factors)}.")

        initial_factors = []
        for n in range(ndim):
            factor_temp = np.ascontiguousarray(temp_factors[n], dtype=DTYPE)
            if factor_temp.shape != (X_dims[n], R_dims_clamped[n]):
                 raise ValueError(f"Provided factor matrix for mode {n} has incorrect shape. Expected {(X_dims[n], R_dims_clamped[n])}, got {factor_temp.shape}.")
            initial_factors.append(factor_temp)

    else:
        raise TypeError(f"Invalid 'init' type: {type(init)}. Use 'random', 'svd', or provide initial factors.")

    # Return validated tensor, dimensions, original ranks, clamped ranks, and initial factors
    return tensor, X_dims, R_dims_in, R_dims_clamped, initial_factors


def tucker(
    tensor: np.ndarray,
    rank: Union[int, Sequence[int]],
    n_iter_max: int = 100,
    init: Union[Literal['svd', 'random'], Tuple[np.ndarray, List[np.ndarray]], List[np.ndarray]] = 'random',
    tol: float = 1e-6,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Tucker decomposition via HOOI algorithm accelerated with CUDA.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor (must be 4D and float32).
    rank : int or Sequence[int]
        Desired target ranks for each mode.
    n_iter_max : int, optional
        Maximum number of HOOI iterations, by default 100.
    init : {'svd', 'random'} or List[np.ndarray] or Tuple[np.ndarray, List[np.ndarray]], optional
        Initialization method or initial factors, by default 'random'.
        'svd' requires TensorLy.
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
        Core tensor has shape (R1_c, ..., R4_c) where R_c is clamped rank.
        Factors is a list [A1, ..., A4] where An has shape (In, Rn_c).

    Raises
    ------
    TypeError
        If inputs are invalid types.
    ValueError
        If inputs have invalid values (e.g., ranks, shapes).
    NotImplementedError
        If input tensor is not 4D (backend limitation).
    ImportError
        If init='svd' is used and TensorLy is not installed.
    RuntimeError
        If the underlying CUDA computation fails.
    """
    if verbose:
        print("--- Running CUDA Tucker Decomposition ---")
        start_time = time.time()

    # Validate inputs and generate initial factors
    try:
        tensor_contig, X_dims, R_dims_in, R_dims_clamped, initial_factors = _validate_tucker_inputs(
            tensor=tensor, rank=rank, modes=None, n_iter_max=n_iter_max, init=init,
            tol=tol, random_state=random_state, verbose=verbose, return_errors=False
        )
    except (ValueError, TypeError, NotImplementedError, ImportError) as e:
        raise e # Re-raise validation errors

    if verbose:
        print("Input validation and initialization complete.")
        print(f"Input shape: {X_dims}, Target Ranks: {R_dims_in}, Clamped Ranks: {R_dims_clamped}")
        print("Calling CUDA backend...")

    # Call the pybind11 wrapper for the CUDA function
    try:
        cuda_start_time = time.time()
        core_tensor, computed_factors = tucker_cuda.hooi(
            h_X_np=tensor_contig,
            X_dims=X_dims,
            h_A_np_list=initial_factors,
            R_dims=R_dims_in, # Pass original target ranks
            tolerance=tol,
            max_iterations=n_iter_max
        )
        cuda_end_time = time.time()
        if verbose:
            print(f"CUDA backend execution time: {cuda_end_time - cuda_start_time:.4f} seconds")
            total_time = time.time() - start_time
            print(f"Total function time: {total_time:.4f} seconds")

    except RuntimeError as e:
        raise RuntimeError(f"CUDA backend execution failed: {e}") from e

    # Return the results
    return core_tensor, computed_factors


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
    """Partial Tucker decomposition (not implemented in this backend).

    Raises:
        NotImplementedError: Always raises, as this backend only supports full Tucker.
    """
    raise NotImplementedError("partial_tucker is not supported by the 'tucker_cuda' backend. Use the full tucker decomposition.")

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