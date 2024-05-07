# Developer Guide {#dev}

[TOC]

## Writing a custom model {#custom-model}

The following tutorials serve as a developer-facing step-by-step guide for creating and testing a custom material model. A simple linear isotropic hardening is used as the example in this tutorial. The model can be mathematically written as
\f{align*}
  k &= H \bar{\varepsilon}^p,
\f}
where \f$\bar{\varepsilon}^p\f$ is the equivalent plastic strain and \f$k\f$ is the isotropic hardening. The input variable for this model is
\f$\mathbf{\varepsilon}\f$, the output variable for this model is \f$k\f$, and the parameters of the model is \f$H\f$.

### Naming conventions {#naming-conventions}

Recall that NEML2 models operates on _labeled tensors_, and that the collection of labels (with their corresponding layout) is called an labeled axis ([LabeledAxis](@ref neml2::LabeledAxis)). NEML2 predefines 5 sub-axes to categorize all the input, output and intermediate variables:
- State \f$\mathcal{S}\f$: Variables on the "state" sub-axis collectively characterize the current _state_ of the material subject to given external forces. The state variables are usually the output of a physically meaningful material model.
- Forces \f$\mathcal{F}\f$: Variables on the "forces" sub-axis define the _external_ forces that drive the response of the material.
- Old state \f$\mathcal{S}_n\f$: The state variables _prior to_ the current material update. In the time-discrete setting, these are the state variables from the previous time step.
- Old forces \f$\mathcal{F}_n\f$: The external forces _prior to_ the current material update. In the time-discrete setting, these are the forces from the previous time step.
- Residual \f$\mathcal{R}\f$: The residual defines an _implicit_ model/function. An implicit model is updated by solving for the state variables that result in zero residual.

In NEML2, the following naming conventions are recommended:
- User-facing variables and option names should be _as descriptive as possible_. For example, the equivalent plastic strain is named "equivalent_plastic_strain". Note that white spaces, quotes, and left slashes are not allowed in the names. Underscores are recommended as an replacement for white spaces.
- Developer-facing variables and option names should use simple alphanumeric symbols. For example, the equivalent plastic strain is named "ep" in consistency with most of the existing literature.
- Developner-facing member variables and option names should use the same alphanumeric symbols. For example, the member variable for the equivalent plastic strain is named `ep`. However, if the member variable is protected or private, it is recommended to prefix it with an underscore, i.e. `_ep`.
- Struct names and class names should use `PascalCase`.
- Function names should use `snake_case`.

### Declaring variables {#declaring-variables}

The development of every model begins with the declaration and registration of its input and output variables. Here, we first define an abstract base class that will be later used to define the linear isotropic hardening relation. The abstract base class defines the isotropic hardening relation
\f[
  k = f\left( \bar{\varepsilon}^p \right),
\f]
mapping the equivalent plastic strain to the isotropic hardening. The base class is named `IsotropicHardening` following the [naming conventions](@ref naming-conventions). The header file [IsotropicHardening.h](@ref neml2::IsotropicHardening) is displayed below
```cpp
#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class IsotropicHardening : public Model
{
public:
  static OptionSet expected_options();

  IsotropicHardening(const OptionSet & options);

protected:
  /// Equivalent plastic strain
  const Variable<Scalar> & _ep;

  /// Isotropic hardening
  Variable<Scalar> & _h;
};
} // namespace neml2
```
Since isotropic hardening _is a_ model, the class inherits from `Model`. The user-facing expected options are defined by the static method `expected_options`. NEML2 handles the parsing of user-specified options and pass them to the constructor (see [Input file syntax](@ref input-file-syntax) on how the input file works). The input variable of the model is the equivalent plastic strain, and the output variable of the model is the isotropic hardening. Their corresponding variable value references are stored as `_ep` and `_h`, respectively, again following the [naming conventions](@ref naming-conventions).

The expected options and the constructor are defined as
```cpp
#include "neml2/models/solid_mechanics/IsotropicHardening.h"

namespace neml2
{
OptionSet
IsotropicHardening::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("equivalent_plastic_strain") = VariableName("state", "internal", "ep");
  options.set<VariableName>("isotropic_hardening") = VariableName("state", "internal", "k");
  return options;
}

IsotropicHardening::IsotropicHardening(const OptionSet & options)
  : Model(options),
    _ep(declare_input_variable<Scalar>("equivalent_plastic_strain")),
    _h(declare_output_variable<Scalar>("isotropic_hardening"))
{
}
} // namespace neml2
```
Recall that variable names on `LabeledAxis` are always fully qualified, the equivalent plastic strain and the isotropic hardening are denoted as "state/internal/ep" and "state/internal/k", respectively. An instance of the class is constructed by extracting user-specified options (of type `OptionSet`). Note how `declare_input_variable<Scalar>` and `declare_output_variable<Scalar>` are used to declare and register the input and output variables.

### Declaring parameters {#declaring-parameters}

Now that the abstract base class `IsotropicHardening` has been implemented, we are ready to define our first concrete NEML2 model that describes a linear isotropic hardening relation
\f[
  k = H \bar{\varepsilon}^p.
\f]
Note that \f$H\f$ is a model parameter. Following the naming convention, the concrete class is named `LinearIsotropicHardening`. The header file is displayed below.
```cpp
#pragma once

#include "neml2/models/solid_mechanics/IsotropicHardening.h"

namespace neml2
{
/**
 * @brief Simple linear map between equivalent strain and hardening
 *
 */
class LinearIsotropicHardening : public IsotropicHardening
{
public:
  static OptionSet expected_options();

  LinearIsotropicHardening(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  const Scalar & _K;
};
} // namespace neml2
```
It derives from the abstract base class `IsotropicHardening` and implements the method `set_value` as the forward operator. The model parameter \f$H\f$ is stored as a protected member variable `_K`. The model implementation is shown below.
```cpp
#include "neml2/models/solid_mechanics/LinearIsotropicHardening.h"

namespace neml2
{
register_NEML2_object(LinearIsotropicHardening);

OptionSet
LinearIsotropicHardening::expected_options()
{
  OptionSet options = IsotropicHardening::expected_options();
  options.set<CrossRef<Scalar>>("hardening_modulus");
  return options;
}

LinearIsotropicHardening::LinearIsotropicHardening(const OptionSet & options)
  : IsotropicHardening(options),
    _K(declare_parameter<Scalar>("K", "hardening_modulus"))
{
}

void
LinearIsotropicHardening::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _h = _K * _ep;

  if (dout_din)
    _h.d(_ep) = _K;

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
```
Note that an additional option named "hardening_modulus" is requested from the user. The model parameter is registered using the API `declare_parameter<Scalar>`. In the `set_value` method, the current value of the input variable, equivalent plastic strain, is stored in the member `_ep`, and so the isotropic hardening can be computed as
```cpp
_K * _ep
```
The computed result is copied into the model output variable `_h` by
```cpp
_h = _K * _ep;
```
In addition, the first derivative of the forward operator is defined as
```cpp
_h.d(_ep) = _K;
```
Last but not the least, the model is registed in the NEML2 model factory using the macro
```cpp
register_NEML2_object(LinearIsotropicHardening);
```
so that an instance of the class can be created at runtime.

## Testing {#testing}

It is of paramount importance to ensure the correctness of the implementation. NEML2 offers 5 types of tests with different purposes.

### Catch tests {#catch-tests}

A Catch test refers to a test directly written in C++ source code within the Catch2 framework. It offers the highest level of flexibility, but requires more effort to set up. To understand how a Catch2 test works, please refer to the [official Catch2 documentation](https://github.com/catchorg/Catch2/blob/v2.x/docs/tutorial.md).

### Unit tests {#unit-tests}

A model unit test examines the outputs of a `Model` given a predefined set of inputs. Model unit tests can be directly designed using the input file syntax with the `ModelUnitTest` type. A variety of checks can be turned on and off based on input file options. To list a few: `check_first_derivatives` compares the implemented first order derivatives of the model against finite-differencing results, and the test is marked as passing only if the two derivatives are within tolerances specified with `derivative_abs_tol` and `derivative_rel_tol`; if `check_cuda` is set to `true`, all checks are repeated twice, once on CPU and once on GPU (if available), and pass only if the two evaluations yield same results within tolerances.

All input files for model unit tests should be stored inside `tests/unit/models`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model unit tests, use the following commands
```
cd tests
./unit_tests models
```

To run a specific model unit test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
cd tests
./unit_tests models -c solid_mechanics/LinearIsotropicElasticity.i
```

### Regression tests {#regression-tests}

A model regression test runs a `Model` using a user specified driver. The results are compared against a predefined reference (stored on the disk checked into the repository). The test passes only if the current results are the same as the predefined reference (again within specified tolerances). The regression tests ensure the consistency of implementations across commits. Currently, `TransientRegression` is the only supported type of regression test.

Each input file for model regression tests should be stored inside a separate folder inside `tests/regression`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model regression tests, use the `regression_tests` executable followed by the physics module, i.e.
```
cd tests
./regression_tests "solid mechanics"
```
To run a specific model regression test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
cd tests
./regression_tests "solid mechanics" -c viscoplasticity/chaboche/model.i
```
Note that the regression test expects an option `reference` which specifies the relative location to the reference solution.

### Verification tests {#verification-tests}

The model verification test is similar to the model regression test in terms of workflow. The difference is the a verification test defines the reference solution using NEML, the predecessor of NEML2. Since NEML was developed with strict software assurance, the verification tests ensure that the migration from NEML to NEML2 does not cause any regression in software quality.

Each input file for model verification tests should be stored inside a separate folder inside `tests/verification`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model verification tests, use the `verification_tests` executable followed by the physics module, i.e.
```
cd tests
./verification_tests "solid mechanics"
```

To run a specific model verification test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
cd tests
./verification_tests "solid mechanics" -c chaboche/chaboche.i
```
The regression test compares variables (specified using the `variables` option) against reference values (specified using the `references` option). The reference variables can be read using input objects with type `VTestTimeSeries`.

### Benchmarking {#benchmarking}

The benchmark tests can be authored within the [Catch2 microbenchmarking framework](https://github.com/catchorg/Catch2/blob/v2.x/docs/benchmarks.md). Before any benchmarks can be executed, the clock's resolution is estimated. A few other environmental artifacts are also estimated at this point, like the cost of calling the clock function, but they almost never have any impact in the results. The user code is executed a few times to obtain an estimate of the amount of runs that should be in each sample. This also has the potential effect of bringing relevant code and data into the caches before the actual measurement starts. Finally, all the samples are collected sequentially by performing the number of runs estimated in the previous step for each sample.

To run a benchmark test, use the `benchmark.sh` script inside the `scripts` directory with 3 positional arguments:
```
./scripts/benchmark.sh Chaboche 5 timings
```
The first positional argument specifies the name of the benchmark test to run. The second positional argument specifies the number of samples to repeat in each iteration. The third positional argument specifies the output directory of the benchmark results.

The Chaboche benchmark test is repeated with different batch sizes and on different devices (in this case CPU and GPU). The final benchmark results are summarized in the following figure.

![Chaboche benchmark results](@ref timings.png){html: width=50%, latex: width=10cm}

