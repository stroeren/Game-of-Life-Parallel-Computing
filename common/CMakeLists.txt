add_library(gol_common STATIC
  grid.cpp
  cli.cpp
  registry.cpp
  report.cpp
)
# Everything includes from the repo root: #include "common/grid.hpp".
target_include_directories(gol_common PUBLIC ${PROJECT_SOURCE_DIR})
