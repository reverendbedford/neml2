# Testing {#testing}

[TOC]

It is of paramount importance to ensure the correctness of the implementation. NEML2 offers four types of tests for different purposes.

## Catch tests {#testing-catch-tests}

A Catch test refers to a test directly written in C++ source code within the Catch2 framework. It offers the highest level of flexibility, but requires more effort to set up. To understand how a Catch2 test works, please refer to the [official Catch2 documentation](https://github.com/catchorg/Catch2/blob/v2.x/docs/tutorial.md).

## Unit tests {#testing-unit-tests}

A model unit test examines the outputs of a `Model` given a predefined set of inputs. Model unit tests can be directly designed using the input file syntax with the `ModelUnitTest` type. A variety of checks can be turned on and off based on input file options. To list a few: `check_first_derivatives` compares the implemented first order derivatives of the model against finite-differencing results, and the test is marked as passing only if the two derivatives are within tolerances specified with `derivative_abs_tol` and `derivative_rel_tol`; if `check_cuda` is set to `true`, all checks are repeated a second time on GPU (if available).

All input files for model unit tests should be stored inside `tests/unit/models`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model unit tests, use the following commands
```
cd tests
../build/unit/unit_tests models
```

To run a specific model unit test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
cd tests
../build/unit/unit_tests models -c solid_mechanics/LinearIsotropicElasticity.i
```

## Regression tests {#testing-regression-tests}

A model regression test runs a `Model` using a user specified driver. The results are compared against a predefined reference (stored on the disk checked into the repository). The test passes only if the current results are the same as the predefined reference (again within specified tolerances). The regression tests ensure the consistency of implementations across commits. Currently, `TransientRegression` is the only supported type of regression test.

Each input file for model regression tests should be stored inside a separate folder inside `tests/regression`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model regression tests, use the `regression_tests` executable followed by the physics module, i.e.
```
cd tests
../build/regression/regression_tests "solid mechanics"
```
To run a specific model regression test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
cd tests
../build/regression/regression_tests "solid mechanics" -c viscoplasticity/chaboche/model.i
```
Note that the regression test expects an option `reference` which specifies the relative location to the reference solution.

## Verification tests {#testing-verification-tests}

The model verification test is similar to the model regression test in terms of workflow. The difference is the a verification test defines the reference solution using NEML, the predecessor of NEML2. Since NEML was developed with strict software assurance, the verification tests ensure that the migration from NEML to NEML2 does not cause any regression in software quality.

Each input file for model verification tests should be stored inside a separate folder inside `tests/verification`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model verification tests, use the `verification_tests` executable followed by the physics module, i.e.
```
cd tests
../build/verification/verification_tests "solid mechanics"
```

To run a specific model verification test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
cd tests
../build/verification/verification_tests "solid mechanics" -c chaboche/chaboche.i
```
The regression test compares variables (specified using the `variables` option) against reference values (specified using the `references` option). The reference variables can be read using input objects with type `VTestTimeSeries`.

