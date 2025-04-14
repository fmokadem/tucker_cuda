# CUDA Tucker Decomposition (4D HOOI)

This project implements the Tucker decomposition for a 4D tensor using the Higher-Order Orthogonal Iteration (HOOI) algorithm, accelerated with CUDA.

## Mathematical Background: Tucker Decomposition

Given a 4D tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times I_3 \times I_4}$, Tucker decomposition approximates it as the n-mode product of a smaller *core tensor* $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3 \times R_4}$ with *factor matrices* $A^{(n)} \in \mathbb{R}^{I_n \times R_n}$ for each mode $n=1, 2, 3, 4$:

$$\mathcal{X} \approx \mathcal{G} \times_1 A^{(1)} \times_2 A^{(2)} \times_3 A^{(3)} \times_4 A^{(4)}$$

Where:
- $R_1, R_2, R_3, R_4$ are the target ranks (typically $R_n \le I_n$).
- The factor matrices $A^{(n)}$ usually have orthonormal columns ($(A^{(n)})^T A^{(n)} = I$).
- $\times_n$ denotes the n-mode product.

The goal is to find the core tensor $\mathcal{G}$ and the orthonormal factor matrices $A^{(n)}$ that minimize the reconstruction error:

$$\min_{\mathcal{G}, A^{(1)}, ..., A^{(4)}} \| \mathcal{X} - \mathcal{G} \times_1 A^{(1)} \times_2 A^{(2)} \times_3 A^{(3)} \times_4 A^{(4)} \|_F^2 \quad \text{s.t. } (A^{(n)})^T A^{(n)} = I$$

where $\| \cdot \|_F$ is the Frobenius norm.

## Numerical Algorithm: Higher-Order Orthogonal Iteration (HOOI)

HOOI is an Alternating Least Squares (ALS) algorithm that iteratively updates each factor matrix while keeping the others fixed.

**Algorithm Steps Implemented (`src/host_logic.cpp`):**

1.  **Initialization:**
    *   Initialize factor matrices $A^{(1)}, A^{(2)}, A^{(3)}, A^{(4)}$ with orthonormal columns using Higher-Order Singular Value Decomposition (HOSVD).
        *   The function `initialize_factors_svd` computes the SVD of the mode-$n$ unfolding $X_{(n)} = U S V^T$ and initializes $A^{(n)}$ with the first $R_n$ columns of $U$.

2.  **Iteration:** Repeat until convergence ($\sum_n \|A^{(n)}_k - A^{(n)}_{k-1}\|_F^2 < \text{tol}$ or `max_iter` reached):
    *   For each mode $n=1$ to $4$:
        *   Project the *original* tensor $\mathcal{X}$ onto the subspace of other *current* factors:
            $$\mathcal{Y}^{(n)} = \mathcal{X} \times_{m \neq n} (A^{(m)})^T$$
        *   Unfold (matricize) $\mathcal{Y}^{(n)}$ along mode $n$ to get matrix $Y_{(n)}$.
        *   Compute SVD of $Y_{(n)} = U S V^T$.
        *   Update $A^{(n)}$ with the first $R_n$ columns of $U$.
    *   **Core Tensor Update:** Recompute the core tensor by projecting the *original* tensor $\mathcal{X}$ onto the *updated* factor subspaces:
        $$\mathcal{G} = \mathcal{X} \times_1 (A^{(1)})^T \times_2 (A^{(2)})^T \times_3 (A^{(3)})^T \times_4 (A^{(4)})^T$$
    *   Check convergence based on the change in factor matrices.

3.  **Final Output:** The final core tensor $\mathcal{G}$ and factor matrices $A^{(n)}$.

## CUDA Implementation Details

Key computationally intensive parts parallelized using CUDA:

1.  **n-Mode Product:** (`launch_nModeProductKernel` in `src/tucker_kernels.cu`)
    *   Implements $\mathcal{Y} = \mathcal{T} \times_n A^T$ (used in HOOI factor updates and core computation).

2.  **Matricization (Unfolding):** (`launch_MatricizeKernel` in `src/tucker_kernels.cu`)
    *   Rearranges a 4D tensor $\mathcal{T}$ into a 2D matrix $T_{(n)}$.

3.  **Singular Value Decomposition (SVD):** (Called within `initialize_factors_svd` and `tucker_hooi_cuda` in `src/host_logic.cpp`)
    *   Uses cuSOLVER library (`cusolverDnSgesvd`).
    *   Computes SVD of the matricized tensors $X_{(n)}$ or $Y_{(n)}$.
    *   **Note on Layout:** Assumes row-major tensor storage. Kernels produce row-major unfoldings. cuSOLVER expects column-major; the SVD implementation in `host_logic.cpp` computes the SVD of the *transpose* of the unfolded matrix ($X_{(n)}^T$ or $Y_{(n)}^T$) and extracts the correct singular vectors.

4.  **Factor Matrix Update:** (Called within `initialize_factors_svd` and `tucker_hooi_cuda` in `src/host_logic.cpp`)
    *   Uses cuBLAS (`cublasSgeam`) to transpose and extract the required singular vectors (first $R_n$ rows of $U^T$ from SVD, where $U$ corresponds to the left singular vectors of the original unfolded matrix) to form the updated $A^{(n)}$.

5.  **Convergence Check:** (Performed within `tucker_hooi_cuda` in `src/host_logic.cpp`)
    *   Uses cuBLAS (`cublasSaxpy` for difference, `cublasSnrm2` for norm) to calculate $\| A^{(n)}_k - A^{(n)}_{k-1} \|_F$.

6.  **Host Logic:** (`src/host_logic.cpp`)
    *   Orchestrates the HOOI algorithm, HOSVD initialization, memory management, and library calls.

7.  **Python Bindings:** (`src/bindings.cpp`)
    *   Uses `pybind11` to expose `tucker_hooi_cuda` to Python.
    *   Handles NumPy array <-> `std::vector` conversions.

8.  **Executables:**
    *   `src/main.cpp`: Standalone executable for basic reconstruction error testing.
    *   `tests/core_logic_test.cpp`: Test suite evaluating HOSVD orthogonality and HOOI reconstruction error across various scenarios.

## File Structure

```
./tdcu/
|-- CMakeLists.txt           # Top-level CMake build configuration
|-- compile.sh             # Build script
|-- README.md              # This file
|-- extern/
|   `-- pybind11/          # Required pybind11 source (submodule/clone)
|-- src/
|   |-- CMakeLists.txt     # CMake config for libraries and main app
|   |-- host_logic.cpp     # C++ host implementation of HOOI & HOSVD
|   |-- host_logic.h       # Header for host logic
|   |-- tucker_kernels.cu  # CUDA kernel implementations
|   |-- tucker_kernels.h   # Header for CUDA kernels
|   |-- main.cpp           # Standalone C++ app entry point (reconstruction test)
|   `-- bindings.cpp       # Python bindings (pybind11)
|-- tests/
|   |-- CMakeLists.txt     # CMake config for test executable
|   `-- core_logic_test.cpp # C++ test suite for core algorithms
|-- tuckercuda/              # (Optional) Python package structure
|   |-- __init__.py
|   `-- tucker.py          # High-level Python wrapper (if used)
`-- examples/
    |-- example_usage.py     # Python usage example
    `-- test_tucker_cuda.py  # Old test script (may be outdated)
```

## Dependencies

-   CUDA Toolkit (>= 11.x)
-   Compatible NVIDIA Driver
-   C++ Compiler (C++11)
-   CMake (>= 3.18)
-   Build tool (Make, Ninja, etc.)
-   `pybind11` library source (in `extern/pybind11`)
-   Python 3.x (Interpreter & Development libraries) for Python module
-   **Python Packages:**
    -   NumPy (`pip install numpy`)

## Compilation

Uses CMake.

1.  **Get pybind11:** Place source in `extern/pybind11`.
    ```bash
    # In ./tdcu/ directory
    git submodule add https://github.com/pybind/pybind11.git extern/pybind11
    git submodule update --init --recursive
    ```

2.  **Run Build Script:**
    ```bash
    # In ./tdcu/ directory
    bash compile.sh
    ```
    Creates `build/` directory and compiles:
    *   Static libraries (`libtucker_kernels.a`, `libtucker_host_logic.a`)
    *   Python module (`build/tucker_cuda*.so`)
    *   Standalone reconstruction test executable (`build/tucker_app`)
    *   Core logic test suite executable (`build/tests/core_logic_test`)

3.  **Manual CMake (Alternative):**
    ```bash
    # In ./tdcu/ directory
    mkdir build && cd build
    cmake .. # Add options like -DCMAKE_CUDA_ARCHITECTURES="80"
    cmake --build . -j $(nproc)
    ```

## Usage

### 1. Core Logic Tests (`core_logic_test`)

This is the primary way to test the numerical correctness of the HOSVD and HOOI implementations.
Located at `build/tests/core_logic_test` after compilation.

**Usage:**

```bash
# In ./tdcu/ directory
./build/tests/core_logic_test
```
This runs a series of predefined test cases with varying dimensions, ranks, and data types (random, with zeros, high/low values). For each case, it reports:
*   Maximum Orthogonality Error of HOSVD factors: $\max_n \| (A^{(n)}_{HOSVD})^T A^{(n)}_{HOSVD} - I \|_{\infty}$
*   Maximum Orthogonality Error of final HOOI factors: $\max_n \| (A^{(n)}_{HOOI})^T A^{(n)}_{HOOI} - I \|_{\infty}$
*   Final Relative Reconstruction Error of HOOI: $\| \mathcal{X} - \mathcal{G} \times_1 A^{(1)}_{HOOI} \dots \times_4 A^{(4)}_{HOOI} \|_F / \| \mathcal{X} \|_F$

### 2. Python Module (`tucker_cuda`)

Ensure `build/` is in `PYTHONPATH` or the module is installed/copied.

```python
import numpy as np
import tucker_cuda # Assuming build/*.so is accessible

I_dims = [10, 12, 11, 13]
R_dims = [4, 5, 4, 6]
X_np = np.random.rand(*I_dims).astype(np.float32)

# Call the C++/CUDA backend
core_np, factors_list_np = tucker_cuda.hooi(X_np, X_dims, R_dims, tolerance=1e-6, max_iterations=50)

print(f"Core shape: {core_np.shape}")
print(f"Factor shapes: {[f.shape for f in factors_list_np]}")
```
See `examples/example_usage.py`.

**Notes:**
*   The Python binding calls the `tucker_hooi_cuda` function which performs HOSVD initialization internally.
*   Input must be a 4D `float32` NumPy array.

### 3. Standalone Reconstruction Test (`tucker_app`)

Located at `build/tucker_app`.
Generates a random tensor based on command-line dimensions, runs HOOI, and prints **only the final relative reconstruction error**.

**Usage:**

```bash
# In ./tdcu/ directory
./build/tucker_app <max_iterations> <tolerance> <I1> <I2> <I3> <I4>
# Example:
./build/tucker_app 100 1e-6 20 22 24 26
```
Output is a single float value (the error).

## Project Status & Known Issues

-   **HOOI Logic:** The core HOOI algorithm logic in `src/host_logic.cpp` has been updated to compute the core tensor within each iteration, aligning better with standard ALS/HOOI procedures.
-   **Testing:** The `tests/core_logic_test.cpp` suite provides detailed testing of HOSVD orthogonality and HOOI convergence/accuracy.
-   **Potential Issues:** Numerical stability or convergence issues might still exist, especially for challenging test cases (e.g., ill-conditioned tensors, specific rank/dimension combinations). Further debugging might be needed if `core_logic_test` shows high errors or NaN results.
