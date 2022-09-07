CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

project (BATCHEDMAT LANGUAGES CXX)

# c++ 17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# torch
find_package(Torch REQUIRED)

# base library
add_subdirectory(src)
