message(STATUS "Configuring Hybridis")

# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(hybridis SHARED ${SRCS})
set_target_properties(hybridis PROPERTIES INSTALL_RPATH "${EXEC_DIR};${Torch_LINK_DIRECTORIES}")
set_target_properties(hybridis PROPERTIES LINK_WHAT_YOU_USE TRUE)

# Hybridis (private) compile options
target_compile_options(hybridis PRIVATE -Wall -Wextra -pedantic -Werror)

# Hybridis headers
file(GLOB_RECURSE _HYBRIDIS_HEADERS ${NEML2_SOURCE_DIR}/include/*.h)
target_include_directories(hybridis PUBLIC ${NEML2_SOURCE_DIR}/include ${NEML2_BINARY_DIR}/include)
target_sources(hybridis
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS ${NEML2_SOURCE_DIR}/include
  FILES
  ${_HYBRIDIS_HEADERS}
)
install(TARGETS hybridis COMPONENT Development FILE_SET HEADERS)

# torch
target_include_directories(hybridis SYSTEM PUBLIC ${Torch_INCLUDE_DIRECTORIES})
target_link_directories(hybridis PUBLIC ${Torch_LINK_DIRECTORIES})
target_link_libraries(hybridis PUBLIC ${Torch_LIBRARIES})

# MPI
find_package(MPI REQUIRED)

if(MPI_CXX_FOUND)
  target_link_libraries(hybridis PUBLIC MPI::MPI_CXX)
else()
  message(FATAL_ERROR "MPI C++ compiler not found")
endif()
