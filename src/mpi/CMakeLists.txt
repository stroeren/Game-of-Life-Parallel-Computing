# Step 6: gol_mpi. Built only when MPI was found (src/CMakeLists.txt guards this);
# nothing else depends on it. It calls the byte and bit-packed kernels in gol_backends.
add_executable(gol_mpi gol_mpi.cpp)
target_link_libraries(gol_mpi PRIVATE gol_backends MPI::MPI_CXX)
