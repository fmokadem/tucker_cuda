#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // std::vector
#include <pybind11/numpy.h>    // numpy array
#include <stdexcept>
#include <vector>
#include <numeric>   // std::accumulate
#include <algorithm> // std::min
#include <cstring>   // std::memcpy

#include "host_logic.h" // tucker_hooi_cuda, real, product

namespace py = pybind11;

// Python wrapper for tucker_hooi_cuda (internal HOSVD init)
py::tuple tucker_hooi_cuda_py(
    py::array_t<real, py::array::c_style | py::array::forcecast> h_X_np,
    const std::vector<int>& X_dims,
    const std::vector<int>& R_dims_in,
    real tolerance = 1e-5,
    int max_iterations = 100)
{
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

    if (R_dims_in.size() != 4) {
        throw std::runtime_error("Target rank list R_dims must have 4 elements.");
    }

    const real* x_ptr = static_cast<real*>(x_buf.ptr);
    std::vector<real> h_X(x_ptr, x_ptr + x_buf.size);

    std::vector<int> R_dims_clamped = R_dims_in;
    for(int n=0; n<4; ++n) {
       if (X_dims[n] <= 0 || R_dims_clamped[n] <= 0) {
           throw std::runtime_error("Input dimensions and target ranks must be positive.");
       }
       R_dims_clamped[n] = std::min(R_dims_clamped[n], X_dims[n]);
       if (R_dims_clamped[n] <= 0) {
            throw std::runtime_error("Clamped rank became non-positive for mode " + std::to_string(n));
       }
    }

    std::vector<std::vector<real>> h_A(4);
    try {
        for (int n = 0; n < 4; ++n) {
            size_t expected_A_size = (size_t)X_dims[n] * R_dims_clamped[n];
            if (expected_A_size > 0)
                h_A[n].resize(expected_A_size);
        }
    } catch (const std::bad_alloc& e) {
         throw std::runtime_error("Failed to allocate memory for host factor matrices: " + std::string(e.what()));
    }

    long long core_size_ll = product(R_dims_clamped);
    if (core_size_ll < 0) {
        throw std::runtime_error("Invalid clamped ranks resulted in negative core size.");
    }
    std::vector<real> h_G(static_cast<size_t>(core_size_ll));

    tucker_hooi_cuda(h_X, X_dims, h_A, h_G, R_dims_in, tolerance, max_iterations);

    std::vector<py::array_t<real>> out_A_list;
    for(int n=0; n<4; ++n) {
         py::array_t<real> a_out_np({(size_t)X_dims[n], (size_t)R_dims_clamped[n]});
         py::buffer_info a_out_buf = a_out_np.request();
         if (h_A[n].size() > 0 && a_out_buf.ptr != nullptr) {
             std::memcpy(a_out_buf.ptr, h_A[n].data(), h_A[n].size() * sizeof(real));
         } else if (h_A[n].size() != static_cast<size_t>(a_out_buf.size)){
              throw std::runtime_error("Factor matrix A" + std::to_string(n+1) + " size mismatch during output conversion.");
         }
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
    m.doc() = R"(Python bindings for CUDA-accelerated 4D Tucker decomposition (HOOI with internal HOSVD init).)";

    m.def("hooi", &tucker_hooi_cuda_py,
          py::arg("h_X_np"),
          py::arg("X_dims"),
          py::arg("R_dims"),
          py::arg("tolerance") = 1e-5,
          py::arg("max_iterations") = 100,
          R"doc(Performs Tucker decomposition for a 4D tensor using the HOOI algorithm on the GPU.
Initialization is performed internally using HOSVD.

Args:
    h_X_np (numpy.ndarray): Input 4D tensor (float32, C-contiguous).
    X_dims (list[int]): Input tensor dimensions [I1, I2, I3, I4].
    R_dims (list[int]): Target ranks [R1, R2, R3, R4].
    tolerance (float, optional): Convergence tolerance. Defaults to 1e-5.
    max_iterations (int, optional): Maximum HOOI iterations. Defaults to 100.

Returns:
    tuple[numpy.ndarray, list[numpy.ndarray]]: (core_tensor, factors)
        core_tensor shape: (R1_c, R2_c, R3_c, R4_c).
        factors: List [A1, A2, A3, A4], where An has shape (In, Rn_c).
)doc");
} 