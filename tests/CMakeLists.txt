# Every test loops over all available registered backends, so a new backend is covered
# as soon as its registry line is in src/CMakeLists.txt.
add_executable(gol_tests test_main.cpp)
target_link_libraries(gol_tests PRIVATE gol_backends)

add_test(NAME common COMMAND gol_tests common)
add_test(NAME blinker COMMAND gol_tests blinker)
add_test(NAME glider COMMAND gol_tests glider)
add_test(NAME checksum_n256 COMMAND gol_tests checksum 256)
add_test(NAME checksum_n1024 COMMAND gol_tests checksum 1024)
add_test(NAME threads COMMAND gol_tests threads)

# The driver itself: CSV shape and --check.
add_test(NAME driver_byte_check
         COMMAND gol --impl byte --n 256 --gens 10 --seed 42 --check)
set_tests_properties(driver_byte_check PROPERTIES
  PASS_REGULAR_EXPRESSION "byte,256,10,1,1,[0-9.]+,[0-9.e+]+,65536,[0-9.]+,0x[0-9a-f]+")

# ---------------------------------------------------------------------------------------
# Step 6: MPI. gol_mpi --check gathers the final grid on rank 0, checksums it and compares
# with the serial byte backend (exit code 3 on a mismatch), so each test below is "the MPI
# checksum equals the serial byte checksum" for one (decomposition, mode, kernel, p, N).
#
# mpirun refuses to start more ranks than it has slots. Only for these tests we then add
# --oversubscribe (Open MPI). It is probed at configure time, never applied to benchmarks
# (bench/mpi_sweep.sh never passes it). Force with -DGOL_MPI_OVERSUBSCRIBE=ON|OFF|AUTO.
# ---------------------------------------------------------------------------------------
if(TARGET gol_mpi AND MPIEXEC_EXECUTABLE)
  set(GOL_MPI_OVERSUBSCRIBE AUTO CACHE STRING "Pass --oversubscribe to mpirun in the MPI tests (AUTO probes)")
  set(_gol_mpi_max_p 6)
  set(_gol_mpi_over "")
  if(GOL_MPI_OVERSUBSCRIBE STREQUAL "ON")
    set(_gol_mpi_over --oversubscribe)
  elseif(GOL_MPI_OVERSUBSCRIBE STREQUAL "AUTO")
    execute_process(COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_gol_mpi_max_p}
                            ${MPIEXEC_PREFLAGS} ${CMAKE_COMMAND} -E echo probe
                    RESULT_VARIABLE _gol_probe_plain OUTPUT_QUIET ERROR_QUIET)
    if(NOT _gol_probe_plain EQUAL 0)
      execute_process(COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_gol_mpi_max_p}
                              --oversubscribe ${MPIEXEC_PREFLAGS} ${CMAKE_COMMAND} -E echo probe
                      RESULT_VARIABLE _gol_probe_over OUTPUT_QUIET ERROR_QUIET)
      if(_gol_probe_over EQUAL 0)
        set(_gol_mpi_over --oversubscribe)
      endif()
    endif()
  endif()
  message(STATUS "gol: MPI tests use mpirun flags '${_gol_mpi_over}' (oversubscribe only for tests)")

  function(gol_mpi_test name np)
    add_test(NAME ${name}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_gol_mpi_over}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:gol_mpi> ${ARGN} ${MPIEXEC_POSTFLAGS})
    set_tests_properties(${name} PROPERTIES LABELS mpi TIMEOUT 120)
  endfunction()

  foreach(kernel byte bitpack)
    foreach(decomp strip block)
      foreach(mode blocking nonblocking)
        # Required matrix: p = 1, 2, 4 at N = 256 and N = 1024, 100 generations, seed 42.
        foreach(n 256 1024)
          foreach(p 1 2 4)
            gol_mpi_test(mpi_${decomp}_${mode}_${kernel}_n${n}_p${p} ${p}
                         --n ${n} --gens 100 --seed 42 --check
                         --decomp ${decomp} --mode ${mode} --kernel ${kernel})
          endforeach()
        endforeach()
        # Uneven splits: p = 3 (rows 86/85/85, 3x1 grid) and p = 6 (3x2 grid, uneven rows).
        foreach(p 3 6)
          gol_mpi_test(mpi_${decomp}_${mode}_${kernel}_n256_p${p} ${p}
                       --n 256 --gens 100 --seed 42 --check
                       --decomp ${decomp} --mode ${mode} --kernel ${kernel})
        endforeach()
      endforeach()
    endforeach()
  endforeach()
  # Bit-packed blocks of unequal word width: N = 256 is 4 words per row, split over 3 columns.
  gol_mpi_test(mpi_block_blocking_bitpack_n256_p12 12 --n 256 --gens 100 --seed 42 --check
               --decomp block --mode blocking --kernel bitpack)
  gol_mpi_test(mpi_block_nonblocking_bitpack_n256_p12 12 --n 256 --gens 100 --seed 42 --check
               --decomp block --mode nonblocking --kernel bitpack)
  # Many generations across the wrap, tall thin blocks (N = 64 on 16 ranks: 4 rows per strip rank; 16x1 .. 4x4 blocks of 16 cells).
  gol_mpi_test(mpi_block_nonblocking_byte_n64_p16 16 --n 64 --gens 200 --seed 7 --check
               --decomp block --mode nonblocking --kernel byte)
  gol_mpi_test(mpi_strip_nonblocking_byte_n64_p16 16 --n 64 --gens 200 --seed 7 --check
               --decomp strip --mode nonblocking --kernel byte)
  # Halo volume: the number of elements one rank sends per generation must equal what the
  # geometry predicts (N = 4096, hand-derived, not computed by gol_mpi). Elements are bytes
  # for --kernel byte and uint64_t words for --kernel bitpack. A strip rank sends 2 rows; a
  # block rank sends 2 rows of w and 2 columns of h + 2 (corners ride with the columns), but
  # nothing in a direction where its neighbor is itself (2x1 grid: no column traffic).
  set(_gol_halo_cases
      "byte,strip,2,8192.0"    "byte,block,2,8192.0"
      "byte,strip,4,8192.0"    "byte,block,4,8196.0"
      "byte,strip,8,8192.0"    "byte,block,8,6148.0"
      "byte,strip,16,8192.0"   "byte,block,16,4100.0"
      "bitpack,strip,2,128.0"  "bitpack,block,2,128.0"
      "bitpack,strip,4,128.0"  "bitpack,block,4,4164.0"
      "bitpack,strip,8,128.0"  "bitpack,block,8,2116.0"
      "bitpack,strip,16,128.0" "bitpack,block,16,2084.0")
  foreach(_case IN LISTS _gol_halo_cases)
    string(REPLACE "," ";" _case "${_case}")
    list(GET _case 0 _k)
    list(GET _case 1 _d)
    list(GET _case 2 _p)
    list(GET _case 3 _e)
    # --csv off: stdout stays empty. mpirun forwards stdout and stderr in chunks and ctest
    # merges them, so a CSV row can land in the middle of the stats line (seen on WSL).
    gol_mpi_test(mpi_halo_${_d}_${_k}_p${_p} ${_p} --n 4096 --gens 2 --seed 42 --check --csv off
                 --decomp ${_d} --mode blocking --kernel ${_k})
    string(REPLACE "." "\\." _e_re "${_e}")
    set_tests_properties(mpi_halo_${_d}_${_k}_p${_p} PROPERTIES
      PASS_REGULAR_EXPRESSION "halo_elems_per_rank_gen=${_e_re} ")
  endforeach()
endif()
