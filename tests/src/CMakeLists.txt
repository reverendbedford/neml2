file(GLOB_RECURSE TEST_UTILS *.cxx)

add_library(testutils SHARED ${TEST_UTILS})

# Main library
target_link_libraries(testutils neml2)

# My headers
target_include_directories(testutils PRIVATE "../include")

# library headers
target_include_directories(testutils PRIVATE "../../include")
