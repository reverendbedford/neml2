# Model Development {#devel}

[TOC]

This tutorial serves as a developer-facing step-by-step guide for creating, testing, and using a constitutive model. A simple small deformation \f$J_2\f$ viscoplasticity model with isotropic hardening is used as the example in this tutorial. For convenience, the mathematical equations for the constitutive model are summarized below
\f{align*}
  \mathbf{M} &= \mathbf{\sigma}, \\
  \bar{\sigma} &= \frac{3}{2} \lVert \operatorname{dev}(\mathbf{M}) \rVert, \\
  k &= H \bar{\varepsilon}^p, \\
  f^p &= \bar{\sigma} - \sigma_y - k, \\
  \mathbf{N}_M &= \partial f^p / \partial \mathbf{M}, \\
  N_k &= \partial f^p / \partial k, \\
  \dot{\gamma} &= \left( \frac{\abs{f^p}}{\eta} \right)^n \operatorname{heav}\left( f^p \right), \\
  \dot{\mathbf{\varepsilon}}^p &= \dot{\gamma} \mathbf{N}_M, \\
  \dot{\bar{\varepsilon}}^p &= \dot{\gamma} N_k, \\
  \dot{\mathbf{\varepsilon}} &= \frac{\mathbf{\varepsilon} - \mathbf{\varepsilon}_n}{t - t_n}, \\
  \dot{\mathbf{\varepsilon}}^e &= \dot{\mathbf{\varepsilon}} - \dot{\mathbf{\varepsilon}}^p, \\
  \dot{\mathbf{\sigma}} &= K \operatorname{vol}\left( \dot{\mathbf{\varepsilon}}^e \right) + G\operatorname{dev}\left( \dot{\mathbf{\varepsilon}}^e \right), \\
  \mathbf{r} &=
  \begin{Bmatrix}
    \mathbf{\sigma} - \mathbf{\sigma}_n - \dot{\mathbf{\sigma}} (t - t_n) \\
    \bar{\varepsilon}^p - \bar{\varepsilon}^p_n - \dot{\bar{\varepsilon}}^p (t - t_n)
  \end{Bmatrix}, \\
  \left( \mathbf{\sigma}, \bar{\varepsilon}^p \right) &= \operatorname{sol}\left( \mathbf{r} = \mathbf{0} \right).
\f}
where \f$\mathbf{\sigma}\f$ is the Cauchy stress, \f$\mathbf{M}\f$ is the Mandel stress, \f$\bar{\sigma}\f$ is the effective stress, \f$\bar{\varepsilon}^p\f$ is the equivalent plastic strain, \f$k\f$ is the isotropic hardening, \f$f^p\f$ is the plastic yield function, \f$\mathbf{N}_M\f$ is the plastic flow direction, \f$N_k\f$ is the isotropic hardening direction, \f$\dot{\gamma}\f$ is the plastic flow rate, \f$\mathbf{\varepsilon}^p\f$ is the plastic strain, \f$\mathbf{\varepsilon}\f$ is the strain, \f$t\f$ is the time, and \f$\mathbf{\varepsilon}^e\f$ is the elastic strain.

Each of the equation above is a function mapping some input variables to some output variables, hence can be defined as a NEML2 model. The example constitutive model under consideration is the composition of these models. Recall that the input variables of a composed model is the set of root input variables, and the output variables of a composed model is the set of leaf output variables. By inspection, the input variables for this composed model are
\f[
  \mathbf{\varepsilon}, t, \mathbf{\varepsilon}_n, t_n, \mathbf{\sigma}_n, \bar{\varepsilon}^p_n,
\f]
and the output variables for this composed model are
\f[
  \mathbf{\sigma}, \bar{\varepsilon}^p.
\f]

Finally, a constitutive model is completely defined up to a set of _parameters_. In this example, the parameters of the composed model under consideration are
\f[
  H, \sigma_y, \eta, n, K, G.
\f]

## Naming conventions

Recall that NEML2 models operates on _labeled tensors_, and that the collection of labels (with their corresponding layout) is called an labeled axis ([LabeledAxis](@ref neml2::LabeledAxis)). NEML2 predefines 6 sub-axes to categorize all the input, output and intermediate variables:
- State \f$\mathcal{S}\f$: Variables on the "state" sub-axis collectively characterize the current _state_ of the material subject to given external forces. The state variables are usually the output of a physically meaningful constitutive model.
- Forces \f$\mathcal{F}\f$: Variables on the "forces" sub-axis define the _external_ forces that drive the response of the material.
- Old state \f$\mathcal{S}_n\f$: The state variables _prior to_ the current constitutive update. In the time-discrete setting, these are the state variables from the previous time step.
- Old forces \f$\mathcal{F}_n\f$: The external forces _prior to_ the current constitutive update. In the time-discrete setting, these are the forces from the previous time step.
- Residual \f$\mathcal{R}\f$: The residual defines an _implicit_ model/function. An implicit model is updated by solving for the state variables that result in zero residual.
- Trial state \f$\tilde{\mathcal{S}}\f$: The state variables during an implicit update that result in a nonzero residual.

In NEML2, the following naming conventions are recommended:
- User-facing variables and parameters should be _as descriptive as possible_. For example, the equivalent plastic strain is named "equivalent_plastic_strain". Note that white spaces, quotes, and left slashes are not allowed in the names. Underscores are recommended as an replacement for white spaces.
- Developer-facing variables and parameters should use simple alphanumeric symbols. For example, the equivalent plastic strain is named "ep" in consistency with most of the existing literature.
- Developner-facing member variables and parameters should use the same alphanumeric symbols. For example, the member variable for the equivalent plastic strain is named `ep`. However, if the member variable is protected or private, it is recommended to prefix it with an underscore, i.e. `_ep`.
- Struct names and class names should use `PascalCase`.
- Function names should use `snake_case`.

## Tutorial 1: Variable declaration and registration

The development of every model begins with the declaration and registration of its input and output variables. Here, we first define an abstract base class that will be later used to define the linear isotropic hardening relation. The abstract base class defines the isotropic hardening relation
\f[
  k = f\left( \bar{\varepsilon}^p \right),
\f]
mapping the equivalent plastic strain to the isotropic hardening. Recall that NEML2 emphasizes modularity, and so the definition of this model _does not_ concern any other model in the final composition. The base class is named `IsotropicHardening` following the naming convention. The header file [IsotropicHardening.h](@ref neml2::IsotropicHardening) is displayed below
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

  const LabeledAxisAccessor equivalent_plastic_strain;
  const LabeledAxisAccessor isotropic_hardening;
};
} // namespace neml2
```
Since isotropic hardening _is a_ model, the class inherits from `Model`. The user-facing expected parameters are defined by the static method `expected_options`. NEML2 handles the parsing of user parameters and pass them to the constructor. The input variable of the model is the equivalent plastic strain, and the output variable of the model is the isotropic hardening. Their corresponding variable accessors are stored as public member variables `equivalent_plastic_strain` and `isotropic_hardening`, respectively.

The expected parameters and the constructor are defined as
```cpp
#include "neml2/models/solid_mechanics/IsotropicHardening.h"

namespace neml2
{
OptionSet
IsotropicHardening::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<LabeledAxisAccessor>("equivalent_plastic_strain") = {{"state", "internal", "ep"}};
  options.set<LabeledAxisAccessor>("isotropic_hardening") = {{"state", "internal", "k"}};
  return options;
}

IsotropicHardening::IsotropicHardening(const OptionSet & options)
  : Model(options),
    equivalent_plastic_strain(declare_input_variable<Scalar>(
        options.get<LabeledAxisAccessor>("equivalent_plastic_strain"))),
    isotropic_hardening(
        declare_output_variable<Scalar>(options.get<LabeledAxisAccessor>("isotropic_hardening")))
{
  setup();
}
} // namespace neml2
```
Both the equivalent plastic strain and the isotropic hardening are internal state, denoted as "state/internal/ep" and "state/internal/k", respectively. The `IsotropicHardening` is constructed by extracting user-specified parameters. Note how `declare_input_variable<Scalar>` and `declare_output_variable<Scalar>` are used to declare and register the input and output variables.

## Tutorial 2: Parameter declaration and registraion

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
class LinearIsotropicHardening : public IsotropicHardening
{
public:
  static OptionSet expected_options();

  LinearIsotropicHardening(const OptionSet & options);

protected:
  /// Simple linear map between equivalent strain and hardening
  void set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din = nullptr,
                         LabeledTensor3D * d2out_din2 = nullptr) const override;

  Scalar _K;
};
} // namespace neml2
```
It derives from the abstract base class `IsotropicHardening` and implements the virtual method `set_value` as the forward operator. The model parameter \f$H\f$ is stored as a protected member variable `_K`. The model implementation is shown below.
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
LinearIsotropicHardening::set_value(const LabeledVector & in,
                                    LabeledVector * out,
                                    LabeledMatrix * dout_din,
                                    LabeledTensor3D * d2out_din2) const
{
  if (out)
    out->set(_K * in(equivalent_plastic_strain), isotropic_hardening);

  if (dout_din)
    dout_din->set(_K, isotropic_hardening, equivalent_plastic_strain);

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
```
Note that an additional option named "hardening_modulus" is requested from the user. The model parameter is registered using the API `declare_parameter<Scalar>`. In the `set_value` method, the current value of the input variable equivalent plastic strain is queried by
```cpp
in(equivalent_plastic_strain)
```
and the isotropic hardening is computed as
```cpp
_K * in(equivalent_plastic_strain)
```
The computed result is copied into the model output `out` by
```cpp
out->set(_K * in(equivalent_plastic_strain), isotropic_hardening);
```
In addition, the first derivative of the forward operator is defined as
```cpp
dout_din->set(_K, isotropic_hardening, equivalent_plastic_strain);
```
Finally, the model is registed in the NEML2 model factory using the macro
```cpp
register_NEML2_object(LinearIsotropicHardening);
```

## Tutorial 3: Testing

It is of paramount importance to ensure the correctness of the implementation. NEML2 offers 5 types of tests with different purposes:

### Catch2 test

A Catch2 test refers to a test directly written in C++ source code within the Catch2 framework. It offers the highest level of flexibility, but requires more effort to set up. To understand how a Catch2 test works, please refer to the [official Catch2 documentation](https://github.com/catchorg/Catch2/blob/v2.x/docs/tutorial.md).

### Model unit test

A model unit test examines the outputs of a `Model` given a predefined set of inputs. Model unit tests can be directly designed using the input file syntax with the `ModelUnitTest` type. A variety of checks can be turned on and off. To list a few: `check_first_derivatives` compares the implemented first order derivatives of the model against finite-differencing results, and passes only if the two derivatives are within tolerances specified with `derivative_abs_tol` and `derivative_rel_tol`; `check_cuda` repeats all checks twice: once on CPU and once on GPU (if available), and passes only if the two evaluations yield same results within tolerances.

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

### Model regression test

A model regression test runs a `Model` using a user specified driver. The results are compared against a predefined reference (stored on the disk inside the repository). The test passes only if the current results are the same as the predefined reference (again within specified tolerances). The regression tests ensure the consistency of implementations across commits. Currently, `TransientRegression` is the only supported type of regression test.

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
Note that the regression test expectes a option `reference` which specifies the relative location to the reference solution.

### Model verification test

The model verification test is similar to the model regression test in terms of workflow. The difference is the a verification test defines the reference solution using NEML, the predecessor of NEML2. Since NEML was developed with strict software assurance, the verification tests ensure that the migration from NEML to NEML2 is as smooth as it can be.

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

### Benchmarking

The benchmark tests can be authored within the [Catch2 microbenchmarking framework](https://github.com/catchorg/Catch2/blob/v2.x/docs/benchmarks.md). Before any benchmarks can be executed, the clock's resolution is estimated. A few other environmental artifacts are also estimated at this point, like the cost of calling the clock function, but they almost never have any impact in the results. The user code is executed a few times to obtain an estimate of the amount of runs that should be in each sample. This also has the potential effect of bringing relevant code and data into the caches before the actual measurement starts. Finally, all the samples are collected sequentially by performing the number of runs estimated in the previous step for each sample.

To run a benchmark test, use the `benchmark.sh` script inside the `scripts` directory with 3 positional arguments:
```
./scripts/benchmark.sh Chaboche 5 timings
```
The first positional argument specifies the name of the benchmark test to run. The second positional argument specifies the number of samples to repeat in each iteration. The third positional argument specifies the output directory of the benchmark results.

The Chaboche benchmark test is repeated with different batch sizes and on different devices (in this case CPU and GPU). The final benchmark results are summarized in the following figure.

![Chaboche benchmark results](@ref timings.png){html: width=50%, latex: width=10cm}
