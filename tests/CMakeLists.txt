# tdcu/tests/CMakeLists.txt

# Define the test executable
add_executable(core_logic_test core_logic_test.cpp) # Correct source file

# Link against necessary libraries
target_link_libraries(core_logic_test PRIVATE
    tucker_host_logic  # Your host logic static library target name
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver     # Needed by HOSVD and HOOI
)

# Add include directory for src headers
target_include_directories(core_logic_test PRIVATE ../src) 