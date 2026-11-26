cmake_minimum_required(VERSION 3.20)
project(gol LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# All executables (gol, gol_tests, gol_mpi) land directly in build/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# All optimization flags live here (CLAUDE.md). Release adds -O3 -march=native.
set(GOL_CXX_OPT_FLAGS -O3 -march=native)
add_compile_options(
  "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:${GOL_CXX_OPT_FLAGS}>"
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>"
)

# OpenMP is required.
find_package(OpenMP REQUIRED COMPONENTS CXX)

# CUDA is optional: nothing outside src/cuda may depend on it.
include(CheckLanguage)
check_language(CUDA)
set(GOL_HAVE_CUDA OFF)
if(CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_compile_options("$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;-Xcompiler=-march=native>")
  set(GOL_HAVE_CUDA ON)
endif()

# MPI is optional: only gol_mpi uses it.
find_package(MPI COMPONENTS CXX)
set(GOL_HAVE_MPI ${MPI_CXX_FOUND})

message(STATUS "gol: build type ${CMAKE_BUILD_TYPE}, CUDA ${GOL_HAVE_CUDA}, MPI ${GOL_HAVE_MPI}")

enable_testing()
add_subdirectory(common)
add_subdirectory(src)
add_subdirectory(tests)
