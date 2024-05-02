# Installation Guide {#install}

[TOC]

## Prerequisites

Compiling the NEML2 core library requires
- A C++ compiler with C++17 support
- CMake >= 3.23

## Dependencies

### Required dependencies

- [PyTorch](https://pytorch.org/get-started/locally/), version 2.2.2.

Other PyTorch releases with a few minor versions around are likely to be compatible. In the PyTorch official download page, several download options are provided: conda, pip, libTorch, and source distribution.
- **Recommended** If you choose to download PyTorch using conda or pip, the NEML2 CMake script can automatically detect and use the PyTorch installation.
- If you choose to download libTorch or build PyTorch from source, you will need to set `LIBTORCH_DIR` to be the location of libTorch when using CMake to configure NEML2.
- If no PyTorch installation can be detected and `LIBTORCH_DIR` is not set at configure time, the NEML2 CMake script will automatically download and use the libTorch obtained from the official website. Note, however, that this method only works on Linux and Mac systems.

> The libTorch distributions from the official website come with two flavors: "Pre-cxx11 ABI" and "cxx11 ABI". Both variants are supported by NEML2. If you are unsure, we recommend the one with "cxx11 ABI".

### Optional dependencies

*No action is needed to manually obtain the optional dependencies.* The compatible optional dependencies will be automatically downloaded and configured by CMake depending on the build customization.

- [HIT](https://github.com/idaholab/moose/tree/master/framework/contrib/hit) for input file parsing.
- [Catch2](https://github.com/catchorg/Catch2) for unit and regression testing.
- [gperftools](https://github.com/gperftools/gperftools) for profiling.
- [Doxygen](https://github.com/doxygen/doxygen) for building the documentation.
- [Doxygen Awesome](https://github.com/jothepro/doxygen-awesome-css) the documentation theme.
- Python packages
  - PyYAML
  - pandas
  - matplotlib
  - pybind11
  - pybind11-stubgen
  - pytest

## Build, Test, and Install

First, obtain the NEML2 source code.

```
git clone https://github.com/reverendbedford/neml2.git
cd neml2
git checkout main
```

Then, configure NEML2. See [build customization](#build-customization) for possible configuration options.

```
mkdir build && cmake -B build .
```

Finally, compile NEML2.

```
cd build && make -j N
```
where `N` is the number of cores to use for parallel compilation.

After the compilation is complete, optionally run the tests to make sure the compilation was successful.

```
make test
```

The compiled NEML2 can be installed as a system library.

```
make install
```

## Build Customization {#build-customization}

Additional configuration options can be passed via command line using the `-DOPTION` or `-DOPTION=ON` format. For example,

```
cmake -DNEML2_DOC=ON -B build .
```
turns on the `NEML2_DOC` option, and additional targets for building the Doxygen documentation will be created inside the Makefile. Note that this would also download additional optional dependencies that are required to build the documentation.

Commonly used configuration options are summarized below. Default options are underlined.

| Option               | Values (<u>default</u>)                                     | Description                                                                               |
| :------------------- | :---------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| CMAKE_BUILD_TYPE     | <u>Debug</u>, Release, MinSizeRel, RelWithDebInfo, Coverage | CMake [Reference](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)     |
| CMAKE_INSTALL_PREFIX |                                                             | CMake [Reference](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html) |
| CMAKE_UNITY_BUILD    |                                                             | CMake [Reference](https://cmake.org/cmake/help/latest/variable/CMAKE_UNITY_BUILD.html)    |
| NEML2_DTYPE          | Float16, Float32, <u>Float64</u>                            | Default floating point integral type used in the material models                          |
| NEML2_INT_DTYPE      | Int8, Int16, Int32, <u>Int64</u>                            | Default fixed point integral type used in the material models                             |
| BUILD_TESTING        | <u>ON</u>, OFF                                              | Master knob for including/excluding all tests                                             |
| NEML2_UNIT           | <u>ON</u>, OFF                                              | Create the unit testing target                                                            |
| NEML2_REGRESSION     | <u>ON</u>, OFF                                              | Create the regression testing target                                                      |
| NEML2_VERIFICATION   | <u>ON</u>, OFF                                              | Create the verification testing target                                                    |
| NEML2_BENCHMARK      | ON, <u>OFF</u>                                              | Create the benchmark testing target                                                       |
| NEML2_PROFILING      | ON, <u>OFF</u>                                              | Create the profiling executable target                                                    |
| NEML2_DOC            | ON, <u>OFF</u>                                              | Create the documentation target                                                           |
| NEML2_PYBIND         | ON, <u>OFF</u>                                              | Create the Python bindings target                                                         |
