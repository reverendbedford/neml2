# Getting started {#install}

[TOC]

Compiling NEML2 requires 
- NEML2 source code
- a C++(>=17) compiler
- CMake
- libTorch

## Install dependencies

libTorch is currently the only supported backend tensor library. Follow the instructions [on this page](https://pytorch.org/get-started/locally/) to download/install the latest version of libTorch that is compatible with your system driver.

CMake is required to configure NEML2. CMake is usually available via the system package manager. Alternatively, follow [these instructions](https://cmake.org/install/) to obtain the latest version of CMake.

## Obtain NEML2 source

To clone the NEML2 source, run

```
git clone https://github.com/Argonne-National-Laboratory/neml2.git
cd neml2
git checkout main
```

The above commands will obtain a copy of the NEML2 source and switch to the stable branch named `main`. Alternatively, you can switch to the development branch `dev` to experiment with new features.

Next, to initialize all the submodules, run

```
git submodule update --recursive --init
```

The above command will initialize the following submodules:
- [Catch2](https://github.com/catchorg/Catch2/tree/v2.x), the testing framework NEML2 uses for unit and regression tests
- [HIT](https://github.com/idaholab/moose/tree/next/framework/contrib/hit), the input file parser
- [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css), the stylesheet used for generating NEML2 documentation

## Configure NEML2

Use CMake to configure NEML2, run

```
cmake -DCMAKE_PREFIX_PATH=/path/to/torch/share/cmake .
```
where `/path/to/torch/share/cmake` is the path to the `share/cmake` directory of your torch installation. Optionally, to view all the available configure options, run
```
cmake -LA
```

The most useful options are

|        Option | Description                                                   |
| ------------: | :------------------------------------------------------------ |
| BUILD_TESTING | Whether to build unit/regression/verification/profiling tests |
|          UNIT | Whether to build unit tests                                   |
|    REGRESSION | Whether to build regression tests                             |
|  VERIFICATION | Whether to build verification tests                           |
|     PROFILING | Whether to build profiling tests                              |
| DOCUMENTATION | Whether to build documentation                                |

## Compile NEML2

To compile NEML2, simply run
```
make -j N
```
where `N` is the number of processors to use for parallel compilation.

After the compilation finishes without errors, you can run the tests to make sure the compilation was successful:
```
make test
```

## Next steps

Depending on your specific use case, the following resources might be useful:

- [Mathematical conventions](math.md) introduces the common mathematical notations used throughout this documentation.
- [Tutorials](tutorials/index.md) walks you through steps required for building a constitutive model. It uses the small deformation \f$J_2\f$ isotropic viscoplasticity as an example.
- [Developer notes](@ref primitive) describes the basics of the library architecture and design philosophy. They can be a good starting point if you want to create your own material model within the NEML2 framework.
- [Class list](annotated.html) is a complete list of class doxygen documentations.
