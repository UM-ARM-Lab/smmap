# - Try to find GUROBI
# Once done this will define
#  NOMAD_FOUND - System has Gurobi
#  NOMAD_INCLUDE_DIRS - The Gurobi include directories
#  NOMAD_LIBRARIES - The libraries needed to use Gurobi

find_path(NOMAD_INCLUDE_DIR NAMES nomad.hpp PATHS "$ENV{NOMAD_HOME}/src")

find_library(NOMAD_LIBRARY NAMES nomad PATHS "$ENV{NOMAD_HOME}/lib")
# find_library(NOMAD_CXX_LIBRARY NAMES gurobi_c++ PATHS "$ENV{NOMAD_HOME}/lib")

set(NOMAD_INCLUDE_DIRS "${NOMAD_INCLUDE_DIR}")
set(NOMAD_LIBRARIES "${NOMAD_LIBRARY}")
# set(NOMAD_LIBRARIES "${NOMAD_CXX_LIBRARY};${NOMAD_LIBRARY}")


message(STATUS "-------------${NOMAD_INCLUDE_DIRS}")
message(STATUS "-------------${NOMAD_LIBRARIES}")

# use c++ headers as default
# set(NOMAD_COMPILER_FLAGS "-DIL_STD" CACHE STRING "Gurobi Compiler Flags")

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBCPLEX_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(NOMAD DEFAULT_MSG NOMAD_LIBRARY NOMAD_INCLUDE_DIR)

mark_as_advanced(NOMAD_INCLUDE_DIR NOMAD_LIBRARY)
