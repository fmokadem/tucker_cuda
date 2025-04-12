#!/bin/bash

# This script provides a single point of execution for compiling the
# CUDA Tucker decomposition project using CMake.

# --- Configuration --- #
# Build directory (will be created if it doesn't exist)
BUILD_DIR="build"

# CMake generator (optional, e.g., "Unix Makefiles", "Ninja")
# Leave empty for CMake default
CMAKE_GENERATOR=""

# CUDA Architectures (passed to CMake)
# Example: "70;75;80;86" (separate multiple values with semicolon)
# Match this with the GPUs you intend to run on.
TARGET_CUDA_ARCH="70;75;80;86"

# Build type (e.g., Debug, Release, RelWithDebInfo)
BUILD_TYPE="Release"

# Options to pass to CMake (ON/OFF)
BUILD_PYTHON="ON"  # Build the Python module
BUILD_EXEC="ON"    # Build the standalone C++ executable

# Number of parallel jobs for make/ninja (e.g., $(nproc))
# Use '1' for serial build
NUM_JOBS=$(nproc)

# --- Script Logic --- #
echo "--- Starting Build Process ---"

# 1. Create Build Directory
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Creating build directory: ${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}" || { echo "Failed to create build directory."; exit 1; }
else
    echo "Build directory already exists: ${BUILD_DIR}"
fi

# 2. Configure with CMake
echo "\n--- Configuring with CMake ---"
CMAKE_CMD="cmake -S . -B ${BUILD_DIR}"

if [ -n "${CMAKE_GENERATOR}" ]; then
    CMAKE_CMD="${CMAKE_CMD} -G \"${CMAKE_GENERATOR}\""
fi

CMAKE_CMD="${CMAKE_CMD} -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
CMAKE_CMD="${CMAKE_CMD} -DCMAKE_CUDA_ARCHITECTURES='${TARGET_CUDA_ARCH}'"
CMAKE_CMD="${CMAKE_CMD} -DBUILD_PYTHON_MODULE=${BUILD_PYTHON}"
CMAKE_CMD="${CMAKE_CMD} -DBUILD_EXECUTABLE=${BUILD_EXEC}"

# Add flags to potentially find Python/pybind11 if not standard
# CMAKE_CMD="${CMAKE_CMD} -DPython3_ROOT_DIR=/path/to/python" # Example
# CMAKE_CMD="${CMAKE_CMD} -Dpybind11_DIR=/path/to/pybind11/share/cmake/pybind11" # Example

echo "Running CMake: ${CMAKE_CMD}"
eval ${CMAKE_CMD}
if [ $? -ne 0 ]; then
    echo "CMake configuration failed."
    exit 1
fi

# 3. Build with make/ninja
echo "\n--- Building Project ---"
BUILD_CMD="cmake --build ${BUILD_DIR} --parallel ${NUM_JOBS}"

echo "Running Build: ${BUILD_CMD}"
eval ${BUILD_CMD}
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

# 4. Post-Build Information
echo "\n--- Build Summary ---"
if [ "${BUILD_PYTHON}" == "ON" ]; then
    echo "Python module should be located in: ${BUILD_DIR}/"
    # Try to find the exact name (platform dependent)
    MODULE_FILE=$(find "${BUILD_DIR}" -name "tucker_cuda*.so" -print -quit)
    if [ -n "${MODULE_FILE}" ]; then
        echo "  -> Found: ${MODULE_FILE}"
        echo "     To use it, ensure the build directory is in your PYTHONPATH,"
        echo "     or copy '${MODULE_FILE}' to your working directory or site-packages."
        echo "     Example Python usage script: examples/example_usage.py"
    else
        echo "  -> Python module file (tucker_cuda*.so) not found in build directory."
    fi
fi

if [ "${BUILD_EXEC}" == "ON" ]; then
    echo "Standalone executable should be located at: ${BUILD_DIR}/tucker_app"
    if [ -f "${BUILD_DIR}/tucker_app" ]; then
        echo "  -> Found: ${BUILD_DIR}/tucker_app"
    else
        echo "  -> Standalone executable (tucker_app) not found in build directory."
    fi
fi

echo "\n--- Build Process Finished Successfully ---"
exit 0 