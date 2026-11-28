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
