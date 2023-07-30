include(Macros)

SUBDIRLIST(NEML2_SUBDIRS .)
get_filename_component(NEML2_SRCROOT . ABSOLUTE)

FOREACH(subdir ${NEML2_SUBDIRS})
  SETUNITYGROUP("NEML2" ${NEML2_SRCROOT} ${subdir})
ENDFOREACH()

# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(neml2 SHARED ${SRCS})
set_target_properties(neml2 PROPERTIES UNITY_BUILD_MODE GROUP)

# My headers
target_include_directories(neml2 PRIVATE "${CMAKE_SOURCE_DIR}/include")

# torch headers
# torch libraries
target_link_libraries(neml2 ${TORCH_LIBRARIES})

# hit for parsing
target_link_libraries(neml2 hit)
