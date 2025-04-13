cmake_minimum_required(VERSION 3.18)
project(TuckerCuda LANGUAGES CXX CUDA)

# --- Options --- #
option(BUILD_PYTHON_MODULE "Build the Python module" ON)
option(BUILD_EXECUTABLE "Build the standalone C++ executable" ON)

# --- Find CUDA --- #
# Prefer modern CUDAToolkit package which defines imported targets
# like CUDA::cudart, CUDA::cublas, CUDA::cusolver
find_package(CUDAToolkit REQUIRED)

# Specify CUDA Architectures (Required for CUDA compilation)
# Example: Add architectures for Maxwell, Pascal, Volta, Turing, Ampere
# Adjust these based on your target GPUs
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 CACHE STRING "CUDA Architectures")
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# --- Build Libraries --- #

# Set Position Independent Code (PIC) for static libraries if building shared libs
if(BUILD_PYTHON_MODULE OR BUILD_SHARED_LIBS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# CUDA Kernels Library
add_library(tucker_kernels STATIC
    src/tucker_kernels.cu
    src/tucker_kernels.h
)
target_compile_features(tucker_kernels PRIVATE cxx_std_11)
# Link kernels library against CUDA runtime
target_link_libraries(tucker_kernels PRIVATE CUDA::cudart)

# Host Logic Library
add_library(tucker_host_logic STATIC
    src/host_logic.cpp
    src/host_logic.h
)
target_compile_features(tucker_host_logic PRIVATE cxx_std_11)
# Host logic depends on kernel wrappers and CUDA libs
target_link_libraries(tucker_host_logic PRIVATE
    tucker_kernels
    CUDA::cublas
    CUDA::cusolver
)

# --- Build Python Module (if enabled) --- #
if(BUILD_PYTHON_MODULE)
    # Find Python 3 interpreter and development libraries (required for pybind11)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    # Find pybind11
    # Option 1: Use find_package if pybind11 is installed system-wide or via CMake config
    # find_package(pybind11 REQUIRED)
    # Option 2: Add pybind11 as a subdirectory (common method)
    # Assuming pybind11 source is placed in extern/pybind11
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)
        add_subdirectory(extern/pybind11)
        message(STATUS "Using pybind11 from subdirectory: extern/pybind11")
    else()
        message(FATAL_ERROR "pybind11 subdirectory not found at 'extern/pybind11'. "
                            "Please clone pybind11 there or configure CMake to find it elsewhere.")
        # Consider adding FetchContent here as an alternative
    endif()

    message(STATUS "Building Python module...")

    # Create the Python module (shared library)
    pybind11_add_module(tucker_cuda # Python module name (import tucker_cuda)
        SHARED           # Must be SHARED for Python module
        src/bindings.cpp # Source file with PYBIND11_MODULE definition
    )
    target_compile_features(tucker_cuda PRIVATE cxx_std_11)

    # Link the Python module against the host logic (which includes kernels)
    # and necessary CUDA libraries.
    target_link_libraries(tucker_cuda PRIVATE
        tucker_host_logic # Links transitively to kernels, cublas, cusolver
        # Explicitly link runtime if not covered transitively or by CUDAToolkit targets
        # CUDA::cudart
    )

    # Ensure Python can find the module (optional, useful for testing)
    # Installs the module to Python's site-packages if `cmake --install .` is run
    # include(GNUInstallDirs)
    # install(TARGETS tucker_cuda DESTINATION ${CMAKE_INSTALL_PYTHON_LIBRARIES})

endif()

# --- Build Standalone Executable (if enabled) --- #
if(BUILD_EXECUTABLE)
    message(STATUS "Building standalone executable...")

    add_executable(tucker_app src/main.cpp)
    target_compile_features(tucker_app PRIVATE cxx_std_11)

    # Link executable against host logic (which includes kernels)
    target_link_libraries(tucker_app PRIVATE
        tucker_host_logic # Links transitively to kernels, cublas, cusolver
        CUDA::cudart # Explicitly link CUDA runtime
    )

endif()

# --- Print Status --- #
message(STATUS "CMake configuration finished.")

# Add the tests directory
add_subdirectory(tests) 