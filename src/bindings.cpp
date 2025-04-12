#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // For std::vector conversion
#include <pybind11/numpy.h>    // For numpy array conversion
#include <stdexcept>
#include <vector>
#include <numeric>   // std::accumulate
#include <algorithm> // std::min
#include <cstring>   // std::memcpy

#include "host_logic.h" // Includes tucker_hooi_cuda and typedef for 'real'

namespace py = pybind11;

// Wrapper to handle Python/NumPy <-> C++ std::vector conversion
py::tuple tucker_hooi_cuda_py(
    py::array_t<real, py::array::c_style | py::array::forcecast> h_X_np,
    const std::vector<int>& X_dims,
    const std::vector<py::array_t<real, py::array::c_style | py::array::forcecast>>& h_A_np_list,
    const std::vector<int>& R_dims_in,
    real tolerance = 1e-5,
    int max_iterations = 100)
{
    // --- Input Validation and Conversion ---
    py::buffer_info x_buf = h_X_np.request();
    if (x_buf.ndim != 4) {
        throw std::runtime_error("Input tensor NumPy array must be 4-dimensional.");
    }
    if (X_dims.size() != 4) {
         throw std::runtime_error("Input tensor dimension list X_dims must have 4 elements.");
    }
    long long expected_x_size = product(X_dims);
     if (expected_x_size < 0 || x_buf.size != static_cast<size_t>(expected_x_size) ) {
         throw std::runtime_error("Input tensor NumPy array size mismatch with X_dims.");
     }

    const real* x_ptr = static_cast<real*>(x_buf.ptr);
    std::vector<real> h_X(x_ptr, x_ptr + x_buf.size);

    // --- Prepare C++ outputs (Factors and Core) ---
    std::vector<int> R_dims_clamped = R_dims_in;
    if (R_dims_clamped.size() != 4) {
        throw std::runtime_error("Target rank list R_dims must have 4 elements.");
    }
    for(int n=0; n<4; ++n) {
       if (X_dims[n] <= 0 || R_dims_clamped[n] <= 0) {
           throw std::runtime_error("Input dimensions and target ranks must be positive.");
       }
       R_dims_clamped[n] = std::min(R_dims_clamped[n], X_dims[n]);
    }

    std::vector<std::vector<real>> h_A(4);
    if (h_A_np_list.size() != 4) {
        throw std::runtime_error("Input factor list must contain 4 NumPy arrays.");
    }

    for (int n = 0; n < 4; ++n) {
        py::buffer_info a_buf = h_A_np_list[n].request();
        long long expected_A_rows = X_dims[n];
        long long expected_A_cols = R_dims_clamped[n];
        size_t expected_A_size = (size_t)expected_A_rows * expected_A_cols;

        if (a_buf.ndim != 2 || a_buf.shape[0] != expected_A_rows || a_buf.shape[1] != expected_A_cols) {
             throw std::runtime_error("Input factor A" + std::to_string(n+1) + " has incorrect shape. Expected ("
                + std::to_string(expected_A_rows) + ", " + std::to_string(expected_A_cols) + "), got ("
                + std::to_string(a_buf.shape[0]) + ", " + std::to_string(a_buf.shape[1]) + ").");
        }
        if (a_buf.size != expected_A_size) {
             throw std::runtime_error("Input factor A" + std::to_string(n+1) + " has incorrect size based on dimensions.");
        }

        const real* a_ptr = static_cast<real*>(a_buf.ptr);
        h_A[n].assign(a_ptr, a_ptr + a_buf.size);
    }

    long long core_size_ll = product(R_dims_clamped);
    if (core_size_ll < 0) {
        throw std::runtime_error("Invalid clamped ranks resulted in negative core size.");
    }
    std::vector<real> h_G(static_cast<size_t>(core_size_ll));

    // --- Call C++/CUDA function ---
    tucker_hooi_cuda(h_X, X_dims, h_A, h_G, R_dims_in, tolerance, max_iterations);

    // --- Output Conversion (C++ -> NumPy) ---
    std::vector<py::array_t<real>> out_A_list;
    for(int n=0; n<4; ++n) {
         py::array_t<real> a_out_np({(size_t)X_dims[n], (size_t)R_dims_clamped[n]});
         py::buffer_info a_out_buf = a_out_np.request();
         std::memcpy(a_out_buf.ptr, h_A[n].data(), h_A[n].size() * sizeof(real));
         out_A_list.push_back(a_out_np);
    }

     py::array_t<real> g_out_np(R_dims_clamped);
     py::buffer_info g_out_buf = g_out_np.request();
     if (h_G.size() > 0 && g_out_buf.ptr != nullptr) {
        std::memcpy(g_out_buf.ptr, h_G.data(), h_G.size() * sizeof(real));
     } else if (h_G.size() != static_cast<size_t>(g_out_buf.size)) {
         throw std::runtime_error("Core tensor size mismatch during output conversion.");
     }

    return py::make_tuple(g_out_np, out_A_list);
}


PYBIND11_MODULE(tucker_cuda, m) {
    m.doc() = R"(Python bindings for CUDA-accelerated 4D Tucker decomposition (HOOI).)";

    m.def("hooi", &tucker_hooi_cuda_py,
          py::arg("h_X_np"),
          py::arg("X_dims"),
          py::arg("h_A_np_list"),
          py::arg("R_dims"),
          py::arg("tolerance") = 1e-5,
          py::arg("max_iterations") = 100,
          R"(Performs Tucker decomposition for a 4D tensor using the HOOI algorithm on the GPU.

Args:
    h_X_np (numpy.ndarray): Input 4D tensor (float32, C-contiguous).
    X_dims (list[int]): Input tensor dimensions [I1, I2, I3, I4].
    h_A_np_list (list[numpy.ndarray]): List of 4 initial factor matrices (float32, C-contiguous).
        Shape of A_n must be (X_dims[n], min(R_dims[n], X_dims[n])).
    R_dims (list[int]): Target ranks [R1, R2, R3, R4].
    tolerance (float, optional): Convergence tolerance. Defaults to 1e-5.
    max_iterations (int, optional): Maximum HOOI iterations. Defaults to 100.

Returns:
    tuple[numpy.ndarray, list[numpy.ndarray]]: (core_tensor, factors)
        core_tensor shape: (R1_c, R2_c, R3_c, R4_c).
        factors: List [A1, A2, A3, A4], where An has shape (In, Rn_c).
)");
} 