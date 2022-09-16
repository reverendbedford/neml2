CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

project (BATCHEDMAT LANGUAGES CXX)

# Enable testing
enable_testing()

# c++ 17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# torch, for which we are going to rely on the system for now
find_package(Torch REQUIRED)

# Catch2, for testing
find_package(Catch2 REQUIRED)

# base library
add_subdirectory(src)

# tests
add_subdirectory(tests)
