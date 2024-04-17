# Getting Started {#install}

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

Feel free to create a ticket at [https://github.com/reverendbedford/neml2/issues](https://github.com/reverendbedford/neml2/issues) if you run into issues about installing the dependencies.

Typically, the C++ compiler, a reasonably modern CMake, Doxygen, and clang-format can be obtained via the system package manager.

libTorch can be downloaded from the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). Choose the compute platform in consistency with the NEML2 compute platform. Also take a note of the version of the libTorch-compatible CUDA if you chose CUDA as the compute platform. Note that there are two versions of libTorch: "Pre-cxx11 ABI" and "cxx11 ABI". Both versions are supported by NEML2. If you are unsure, we recommend the one with "cxx11 ABI".

> In the future we may provide an option to automatically install a compatible version of libTorch.

There are many ways of installing the NVIDIA driver and the CUDA toolkit. We will not try to make a recommendation here. However, do make sure the NVIDIA driver is compatible with your GPU. It is also recommended to install a CUDA toolkit with the same version number as the libTorch CUDA version.

## Quick Start {#user}

NEML2 uses the [HIT](https://github.com/idaholab/hit) format, a simple hierarchical text language, for model specification. More generally speaking, HIT is the canonical language used in NEML2 for serialization, deserialization, and archival purposes.

The NEML2 input files have extension `.i`. An example input file is shown below
```python
[Tensors]
  [end_time]
    type = LogSpaceTensor
    start = -1
    end = 5
    steps = 20
  []
  [times]
    type = LinSpaceTensor
    end = end_time
    steps = 100
  []
  [max_strain]
    type = InitializedSymR2
    values = '0.1 -0.05 -0.05'
    nbatch = 20
  []
  [strains]
    type = LinSpaceTensor
    end = max_strain
    steps = 100
  []
[]

[Drivers]
  [driver]
    type = SolidMechanicsDriver
    model = 'model'
    times = 'times'
    prescribed_strains = 'strains'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Solvers]
  [newton]
    type = Newton
  []
[]

[Models]
  [implicit_rate]
    type = ComposedModel
    models = 'mandel_stress vonmises yield normality flow_rate Eprate Erate Eerate elasticity integrate_stress'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]
```
There are four top-level sections:
- `[Tensors]`: Tensors specified from within the input file.
- `[Solvers]`: Solvers for solving an implicit model.
- `[Models]`: NEML2 constitutive models.
- `[Drivers]`: Drivers used to evaluate/test models.

The user has full control over the sub-sections under the top-level sections, called _objects_. The sub-section name is used as the name of the object. Each object reserves a special field named "type". NEML2 parses the value in the "type" field and constructs the corresponding object at run time. The syntax and options for all objects are listed in the [syntax documentation](@ref syntax).

Currently, NEML2 does not maintain a set of examples. However, the regression tests shall serve as decent input file templates. The regression tests are located in `/tests/regression` in the repository.


## Next steps

Depending on your specific use case, the following resources might be useful:

- [Mathematical conventions](@ref math) introduces the common mathematical notations used throughout this documentation.
- [Implementation](@ref impl) describes the basics of the library architecture and design philosophy. They can be a good starting point if you want to create your own material model within the NEML2 framework.
- [Model development](@ref devel) is a step-by-step guide for building a constitutive model. It uses the small deformation \f$J_2\f$ isotropic viscoplasticity as an example.
- [Class list](https://reverendbedford.github.io/neml2/annotated.html) is a complete list of class doxygen documentations.
