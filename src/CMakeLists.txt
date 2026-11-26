# Backends are compiled into one OBJECT library so every self-registering object file is
# linked into gol and gol_tests (a static library would let the linker drop them).
add_library(gol_backends OBJECT)
target_link_libraries(gol_backends PUBLIC gol_common OpenMP::OpenMP_CXX)

# gol_backend(<sources>...)       a CPU backend
# gol_cuda_backend(<sources>...)  a CUDA backend; ignored when no CUDA compiler was found
function(gol_backend)
  target_sources(gol_backends PRIVATE ${ARGN})
endfunction()

function(gol_cuda_backend)
  if(GOL_HAVE_CUDA)
    target_sources(gol_backends PRIVATE ${ARGN})
  endif()
endfunction()

if(GOL_HAVE_CUDA)
  target_link_libraries(gol_backends PUBLIC CUDA::cudart)
endif()

# ---------------------------------------------------------------------------------------
# Registry: one line per backend source file. Keep them in optimization-path order.
# ---------------------------------------------------------------------------------------
gol_backend(byte/byte.cpp)

# ---------------------------------------------------------------------------------------

add_executable(gol ${PROJECT_SOURCE_DIR}/common/driver_main.cpp)
target_link_libraries(gol PRIVATE gol_backends)

# Step 6: gol_mpi is its own binary and is built only when MPI was found.
if(GOL_HAVE_MPI AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mpi/CMakeLists.txt)
  add_subdirectory(mpi)
endif()
