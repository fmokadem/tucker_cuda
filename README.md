# CUDA Tucker Decomposition (4D HOOI)

This project implements the Tucker decomposition for a 4D tensor using the Higher-Order Orthogonal Iteration (HOOI) algorithm, accelerated with CUDA.

## Mathematical Background: Tucker Decomposition

Given a 4D tensor \( \mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times I_3 \times I_4} \), Tucker decomposition approximates it as the n-mode product of a smaller *core tensor* \( \mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3 \times R_4} \) with *factor matrices* \( A^{(n)} \in \mathbb{R}^{I_n \times R_n} \) for each mode \( n=1, 2, 3, 4 \):

\[
\mathcal{X} \approx \mathcal{G} \times_1 A^{(1)} \times_2 A^{(2)} \times_3 A^{(3)} \times_4 A^{(4)}
\]

Where:
- \( R_1, R_2, R_3, R_4 \) are the target ranks (typically \( R_n \le I_n \)).
- The factor matrices \( A^{(n)} \) usually have orthonormal columns (\( (A^{(n)})^T A^{(n)} = I \)).
- \( \times_n \) denotes the n-mode product.

The goal is to find the core tensor \( \mathcal{G} \) and the orthonormal factor matrices \( A^{(n)} \) that minimize the reconstruction error:
\[
\min_{\mathcal{G}, A^{(1)}, ..., A^{(4)}} || \mathcal{X} - \mathcal{G} \times_1 A^{(1)} \times_2 A^{(2)} \times_3 A^{(3)} \times_4 A^{(4)} ||_F^2 \quad \text{s.t. } (A^{(n)})^T A^{(n)} = I
\]
where \( || \cdot ||_F \) is the Frobenius norm.

## Numerical Algorithm: Higher-Order Orthogonal Iteration (HOOI)

HOOI is an Alternating Least Squares (ALS) algorithm that iteratively updates each factor matrix while keeping the others fixed.

**Algorithm Steps:**

1.  **Initialization:**
    *   Initialize factor matrices \( A^{(1)}, A^{(2)}, A^{(3)}, A^{(4)} \) with orthonormal columns.
        *   Common methods: Random orthogonal matrices (via QR or Gram-Schmidt) or HOSVD (using SVD on tensor unfoldings).
        *   This implementation uses random orthogonal initialization in `src/main.cpp` and provides a *stub* for HOSVD in `src/host_logic.cpp`.

2.  **Iteration:** Repeat until convergence (change in factors < `tol` or `max_iter` reached):
    *   For each mode \( n=1 \) to \( 4 \):
        *   Project \( \mathcal{X} \) onto the subspace of other *current* factors:
            \( \mathcal{Y}^{(n)} = \mathcal{X} \times_{m \neq n} (A^{(m)})^T \)
        *   Unfold (matricize) \( \mathcal{Y}^{(n)} \) along mode \( n \) to get matrix \( Y_{(n)} \).
        *   Compute SVD of \( Y_{(n)} = U S V^T \).
        *   Update \( A^{(n)} \) with the first \( R_n \) columns of \( U \).
    *   Check convergence (e.g., \( \sum_n ||A^{(n)}_k - A^{(n)}_{k-1}||_F^2 \)).

3.  **Compute Core Tensor:**
    *   Project \( \mathcal{X} \) onto the final factor subspaces:
        \[
        \mathcal{G} = \mathcal{X} \times_1 (A^{(1)})^T \times_2 (A^{(2)})^T \times_3 (A^{(3)})^T \times_4 (A^{(4)})^T
        \]

## CUDA Implementation Details

Key computationally intensive parts parallelized using CUDA:

1.  **n-Mode Product:** (`launch_nModeProductKernel` in `src/tucker_kernels.cu`)
    *   Implements \( \mathcal{Y} = \mathcal{T} \times_n A^T \) (used in HOOI updates and core computation).
    *   Each thread computes one element of \( \mathcal{Y} \).

2.  **Matricization (Unfolding):** (`launch_MatricizeKernel` in `src/tucker_kernels.cu`)
    *   Rearranges a 4D tensor \( \mathcal{T} \) into a 2D matrix \( T_{(n)} \).

3.  **Singular Value Decomposition (SVD):** (Called within `tucker_hooi_cuda` in `src/host_logic.cpp`)
    *   Uses cuSOLVER library (`cusolverDnSgesvd`).
    *   Computes SVD of the matricized tensor \( Y_{(n)} \).
    *   **Note on Layout:** Assumes row-major tensor storage. Kernels produce row-major unfoldings. cuSOLVER expects column-major; the code handles this by passing appropriate dimensions for the implicit transpose.

4.  **Factor Matrix Update:** (Called within `tucker_hooi_cuda` in `src/host_logic.cpp`)
    *   Uses cuBLAS (`cublasSgeam`) to extract and transpose the required singular vectors (first \( R_n \) rows of \( V^T \) from SVD) to form the updated \( A^{(n)} \).

5.  **Convergence Check:** (Performed within `tucker_hooi_cuda` in `src/host_logic.cpp`)
    *   Uses cuBLAS (`cublasSaxpy` for difference, `cublasSnrm2` for norm) to calculate \( || A^{(n)}_k - A^{(n)}_{k-1} ||_F \).

6.  **Host Logic:** (`src/host_logic.cpp`, `src/main.cpp`)
    *   Manages workflow, memory, library calls, iteration loop.
    *   Uses ping-pong buffering for intermediate projection results.

7.  **Python Bindings:** (`src/bindings.cpp`)
    *   Uses `pybind11` to expose `tucker_hooi_cuda` to Python.
    *   Handles NumPy array <-> `std::vector` conversions.

## File Structure

```
/
|-- CMakeLists.txt           # CMake build configuration
|-- compile.sh             # Build script
|-- README.md              # This file
|-- extern/
|   `-- pybind11/          # Required pybind11 source (submodule/clone)
|-- src/
|   |-- host_logic.cpp/h     # C++ host implementation of HOOI
|   |-- tucker_kernels.cu/h  # CUDA kernel implementations
|   |-- main.cpp             # Standalone C++ executable entry point
|   `-- bindings.cpp         # Python bindings (pybind11)
|-- tuckercuda/              # Python package
|   |-- __init__.py
|   `-- tucker.py          # High-level Python wrapper function
`-- examples/
    |-- example_usage.py     # Python usage example
    `-- test_tucker_cuda.py  # Test script comparing C++ exe to TensorLy
```

## Dependencies

-   CUDA Toolkit (>= 11.x)
-   Compatible NVIDIA Driver
-   C++ Compiler (C++11)
-   CMake (>= 3.18)
-   Build tool (Make, Ninja, etc.)
-   `pybind11` library source (in `extern/pybind11`)
-   Python 3.x (Interpreter & Development libraries)
-   **Python Packages:**
    -   NumPy (`pip install numpy`)
    -   (Optional) TensorLy (`pip install tensorly`) for `init='svd'` and `test_tucker_cuda.py`.

## Compilation

Uses CMake.

1.  **Get pybind11:** Place source in `extern/pybind11`.
    ```bash
    # Preferred method
    git submodule add https://github.com/pybind/pybind11.git extern/pybind11
    git submodule update --init --recursive
    ```

2.  **Run Build Script:**
    ```bash
    bash compile.sh
    ```
    Creates `build/` directory and compiles:
    *   Python module (`build/tucker_cuda*.so`)
    *   Standalone executable (`build/tucker_app`)

3.  **Manual CMake:**
    ```bash
    mkdir build && cd build
    cmake .. # Add options like -DCMAKE_CUDA_ARCHITECTURES="80"
    cmake --build . -j $(nproc)
    ```

## Usage

### 1. Python Module (`tuckercuda`)

Ensure `build/` is in `PYTHONPATH` or the module is installed.

Use the high-level wrapper `tuckercuda.tucker` (mimics TensorLy):

```python
import numpy as np
from tuckercuda import tucker

I_dims = (10, 12, 11, 13)
R_dims = (4, 5, 4, 6)
X = np.random.rand(*I_dims).astype(np.float32)

core, factors = tucker(X, rank=R_dims, init='random', tol=1e-7, n_iter_max=100)

print(f"Core shape: {core.shape}")
print(f"Factor shapes: {[f.shape for f in factors]}")
```
See `examples/example_usage.py`.

**Notes:**
*   Backend supports only **4D float32** tensors.
*   `partial_tucker` is **not implemented**.
*   `init='svd'` requires TensorLy.

### 2. Standalone Executable (`tucker_app`)

Located at `build/tucker_app`.
Usage (from build dir): `./tucker_app <in.bin> <A1.bin> ... <G.bin> <I1> ... <R4> [tol] [max_iter]`
Requires input tensor and outputs factors/core as raw binary files.

## Testing

-   `examples/example_usage.py`: Basic Python usage.
-   `examples/test_tucker_cuda.py`: Compares `build/tucker_app` output against TensorLy via file I/O (requires TensorLy). 