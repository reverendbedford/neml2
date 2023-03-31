# Getting started {#install}

[TOC]

## Choose the compute platform

NEML2 can be compiled to support two compute platforms:
- CPU
- CUDA

Choose the CPU compute platform if you only need to run NEML2 using CPUs. Choose the CUDA compute platform if you plan to take advantage of the GPU(s) of the computer.

## Dependencies

NEML2 depends on the following drivers/packages/libraries:
|  CPU  | CUDA  | Dependency   | Version |
| :---: | :---: | :----------- | :------ |
|   x   |   x   | C++ compiler | >=17    |
|   x   |   x   | CMake        | >=3.1   |
|   x   |   x   | libTorch     |         |
|       |   x   | CUDA toolkit |         |

See [notes on obtaining the dependencies](@ref NotesOnObtainingTheDependencies) for some guidance.

For developers, some additional dependencies are recommended (regardless of the compute platform):
- [Doxygen](https://www.doxygen.nl/), the documentation generator
- [clang-format](https://clang.llvm.org/docs/ClangFormat.html), the C++ code formatter

## Install NEML2

First, obtain the NEML2 source code.

```
git clone https://github.com/reverendbedford/neml2.git
cd neml2
git checkout main
git submodule update --recursive --init
```

Then, configure NEML2 and generate the Makefile.

```
cmake -DCMAKE_PREFIX_PATH=/path/to/torch/share/cmake .
```
where `/path/to/torch/share/cmake` is the path to the `share/cmake` directory of your libTorch installation.

Finally, compile NEML2.

```
make -j N
```
where `N` is the number of processors to use for parallel compilation.

After the compilation finishes, you can run the tests to make sure the build was successful:
```
make test
```

## Notes on obtaining the dependencies {#NotesOnObtainingTheDependencies}

There's no unique way of installing the dependencies. If you are unfamiliar with this process, instructions/suggestions can be found by googling "How to install X on Y" where X is the dependency you want to install, and Y is your operating system.

Typically, the C++ compiler, a reasonably modern CMake, Doxygen, and clang-format can be obtained via the system package manager.

libTorch can be downloaded from the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). Choose the compute platform in consistency with the NEML2 compute platform. Also take a note of the version of the libTorch-compatible CUDA if you chose CUDA as the compute platform. Note that there are two versions of libTorch: "Pre-cxx11 ABI" and "cxx11 ABI". Both versions are supported by NEML2. If you are unsure, we recommend the one with "cxx11 ABI".

There are many ways of installing the Nvidia driver and the CUDA toolkit. We will not try to make a recommendation here. However, do make sure the Nvidia driver is compatible with your GPU. It is also recommended to install a CUDA toolkit with the same version number as the libTorch CUDA version. 

## Next steps

Depending on your specific use case, the following resources might be useful:

- [Mathematical conventions](math.md) introduces the common mathematical notations used throughout this documentation.
- [Tutorials](tutorials/index.md) walks you through steps required for building a constitutive model. It uses the small deformation \f$J_2\f$ isotropic viscoplasticity as an example.
- [Developer notes](@ref primitive) describes the basics of the library architecture and design philosophy. They can be a good starting point if you want to create your own material model within the NEML2 framework.
- [Class list](annotated.html) is a complete list of class doxygen documentations.
