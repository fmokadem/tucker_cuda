# ... (existing add_executable for tucker_app) ...

# Ensure tucker_app links to necessary libs for its reconstruction code
target_link_libraries(tucker_app PRIVATE
    tucker_host_logic
    CUDA::cudart
    CUDA::cublas # Add this if not already present
    # Add others if needed by helpers (cusolver?)
) 