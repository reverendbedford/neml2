# NEML2 <!-- omit in toc -->

- [Promises of NEML2](#promises-of-neml2)
- [Tasks](#tasks)
- [Things we are not happy with](#things-we-are-not-happy-with)
- [Updates Oct. 7, 2022](#updates-oct-7-2022)
  - [Unifying batched and unbatched tensors](#unifying-batched-and-unbatched-tensors)
  - [Batched tensor (base of everything)](#batched-tensor-base-of-everything)
  - [A batched tensor with static base size](#a-batched-tensor-with-static-base-size)
  - [Scalar](#scalar)
  - [Symmetric second order tensor](#symmetric-second-order-tensor)
  - [Symmetric (minor) fourth order tensor](#symmetric-minor-fourth-order-tensor)
  - [Batch dimension of the primitive data types](#batch-dimension-of-the-primitive-data-types)
  - [Use AD to get the partials.](#use-ad-to-get-the-partials)
  - [Try linking with an external package, e.g. MOOSE.](#try-linking-with-an-external-package-eg-moose)
  - [Set up unit tests (CPU and GPU).](#set-up-unit-tests-cpu-and-gpu)
  - [Set up documentation.](#set-up-documentation)
  - [Set up coverage report.](#set-up-coverage-report)
  - [Set up benchmarks.](#set-up-benchmarks)
    - [Scalar](#scalar-1)
    - [SymR2](#symr2)
    - [SymSymR4](#symsymr4)
    - [Constitutive update](#constitutive-update)
- [Updates Oct. 7, 2022](#updates-oct-7-2022-1)
  - [State vs Force](#state-vs-force)
  - [Chunked map and parallel coordination](#chunked-map-and-parallel-coordination)
  - [The new benchmark](#the-new-benchmark)

## [Promises of NEML2](#neml2-)

1. Users focus on the actual constitutive updates, while NEML2 takes care of everything else.
2. User models can run in two modes: batched and unbatched.
3. User models can run both on CPU and on GPU.
4. NEML2 predefines commonly used constitutive models.

## [Tasks](#neml2-)

- [ ] Setup "primitive" data types.
  - [x] Batched tensor (base of everything)
  - [x] A batched tensor with static base size
  - [x] A batched tensor with named views
  - [x] Scalar
  - [x] Symmetric second order tensor
  - [x] Symmetric (minor) fourth order tensor
  - [ ] (?) Second order tensor
  - [ ] (?) Fourth order tensor (with other symmetries)
  - [ ] (?) Quaternion
- [ ] Setup a class structure for history-dependent models based on the user providing the implicit function definition of the model update.
- [x] Use AD to get the partials.
- [ ] Use AD to get the second derivatives (for return mapping).
- [x] Set up unit tests (CPU and GPU).
- [x] Set up documentation.
- [x] Set up coverage report.
- [x] Set up benchmarks.
- [ ] Migrate models from NEML to NEML2
  - [x] Linear elasticity
  - [ ] Hypoelasticity
  - [ ] (?) Hyperelasticity
  - [ ] Hypoelasticity-(perzyna/consistent)plasticity
  - [ ] (?) Hyperelasticity-(perzyna/consistent)plasticity
  - [ ] Damage
  - [ ] (?) Gradient damage
  - [ ] Utilities, e.g. interpolation,
- [ ] Play around with making the model parameters as Torch `Variables` and getting parameter partials.
- [ ] Write python bindings.
- [ ] Support model definition using input files.
- [x] Try linking with an external package, e.g. MOOSE.

## [Things we are not happy with](#neml2-)

1. The `State` and `LabeledMatrix` classes should share more code.
2. ~~The `Batched` and `Unbatched` classes should share more code.~~
3. I'm not content with the interface in `ConstitutiveModel`.  I want to
  - Think a bit about return by value and return by reference.
  - Think about how to provide the "tangent" and update with options
    for separate or simultaneous evaluation.
  - Expand the API to let people ask for the dot product of the
    tangent with some state (i.e. the actual linearized update).
4. We need some finite difference test classes to check derivatives of models and bits of models.
5. We probably want to typedef the scalar type.
6. For torch `TensorOptions`, we should be able to configure it with cmake and modify it at runtime.
7. Let's namespace everything.
8. A lot of runtime exceptions should be static.

## [Updates Oct. 7, 2022](#neml2-)

### [Unifying batched and unbatched tensors](#neml2-)

- Design choice: I no longer distinguish between batched and unbatched tensors. Instead, I insist that the size of each batch dimension is either 1 or Bn, where Bn is the batch size of the n-th batch dimension.
- Advantages:
  - We don't need to keep two specializations for every primitive data type.
  - This appears to be much more conformant with libtorch.
  - This has no influence on speed (I have benchmarked).
- Disadvantages:
  - I am not sure if this will affect locality, but who cares about the locality of a single batch tensor anyways?
  - An additional layer of indirection when constructing from a `torch::Tensor`, e.g. previously a single batch tensor (of size `(6,)`) can be constructed as

    ```cpp
    SymR2 a(torch::tensor({1, 2, 3, 4, 5, 6}));
    ```
    now we have to construct a single batch tensor (of size `(1, 6,)`) as
    ```cpp
    SymR2 a(torch::tensor({{1, 2, 3, 4, 5, 6}}));
    ```

### [Batched tensor (base of everything)](#neml2-)

- [`BatchTensor<N>`](doc/class-doc/html/classBatchTensor.html)
- Some additional getters and setters
- `batch_dim()` is now `constexpr`.
- `expand_batch(B)` expands a single batch tensor into shape `B`. This also claims ownership. We could implement another version that simply returns an expanded view.
- A very general [`einsum`](doc/class-doc/html/BatchTensor_8h.html)
  - Example
    ```cpp
    einsum({A, B, C}, {"ij", "jk", "klmn"}, "ilmn")
    ```
  - This is equivalent to
    ```cpp
    torch::einsum("...ij,...jk,...klmn->...ilmn", {A, B, C});
    ```
  - If we implement it carefully (which is, embarrassingly, not the case right now), the dispatch should be static (as `string_view` is `constexpr`), i.e. the correct CUDA program is selected at compile-time, if we are using GPU.

### [A batched tensor with static base size](#neml2-)

- [`FixedDimTensor`](doc/class-doc/html/classFixedDimTensor.html)
- All of the methods have been refactored into `BatchTensor`.
- `base_sizes` -> `_base_sizes` to avoid name conflict with the base method.

### [Scalar](#neml2-)

- [`Scalar`](doc/class-doc/html/classScalar.html)
- I kept the convenient conversion constructor from `double` with an optional batch size, e.g.
    ```cpp
    Scalar a = 1.5;
    Scalar b(1.5, 1000);
    ```
- Design choice: The base dim of the `Scalar` is `0` instead of `1`. Again, this is more conformant with libtorch, as almost all tensor operations automatically squeeze out the trailing dimension with size 1. To "unsqueeze" it we can simply do
    ```cpp
    Scalar a(1.5, 1000); // size (1000,)
    a = a.unsqueeze(1); // size (1000, 1,)
    ```

### [Symmetric second order tensor](#neml2-)

- [`SymR2`](doc/class-doc/html/classSymR2.html)
- As promised, all methods are implemented in a batch-agnostic way.

### [Symmetric (minor) fourth order tensor](#neml2-)

- [`SymSymR4`](doc/class-doc/html/classSymSymR4.html)
- As promised, all methods are implemented in a batch-agnostic way.

### [Batch dimension of the primitive data types](#neml2-)

- Right now I am assuming `Scalar`, `SymR2` and `SymSymR4` all have a batch dimension of `1`.
- Alternatively, we could template them on batch dimension `N`.

### [Use AD to get the partials.](#neml2-)

- [`ConstitutiveModel::linearized_state_update`](doc/class-doc/html/classConstitutiveModel.html)
  - Using hand-coded Jacobian
    ```cpp
    auto state = model.linearized_state_update(forces_np1, state_n, forces_n);
    ```
  - Using reverse-AD
    ```cpp
    auto state = model.linearized_state_update(forces_np1, state_n, forces_n, true);
    ```
- This assumes that states and forces are flattened (logically 1D).
- Mark probably has a different interface now.
- I am concerned about the performance:
  ```cpp
    forces_np1.requires_grad_();
    State state_np1(state(), forces_np1.batch_size());
    update(state_np1, forces_np1, state_n, forces_n);

    // Allocate space for Jacobian
    torch::Tensor jac =
        torch::zeros(add_shapes(state_np1.sizes(), forces_np1.base_sizes()), TorchDefaults);

    // Loop over components of the state to retrieve the derivatives
    for (TorchSize i = 0; i < state_np1.base_sizes()[0]; i++)
    {
      torch::Tensor grad_outputs = torch::zeros_like(state_np1);
      grad_outputs.index_put_({torch::indexing::Ellipsis, i}, 1);
      jac.index_put_({torch::indexing::Ellipsis, i, torch::indexing::Ellipsis},
                     torch::autograd::grad({state_np1}, {forces_np1}, {grad_outputs}, true)[0]);
    }

    return LabeledMatrix(state(), forces(), jac);
  ```

### [Try linking with an external package, e.g. MOOSE.](#neml2-)

So I've experimented this a little bit. At least on Ubuntu this is pretty straightforward. The easiest route is probably:
1. Get a somewhat modern compiler with ldd version greater than 2.35.
2. Compile petsc and libmesh manually, i.e. not from conda (as the ones provided by conda are linked to old GLIBC).
3. Configure moose to use libtorch. For example `./scripts/setup_libtorch.sh` followed by `./configure --with-libtorch`.
4. It's also necessary to exclude the files that includes neml2 from the UNITY build system, otherwise there will be namespace clashes.
5. Compile the application. That's it.

### [Set up unit tests (CPU and GPU).](#neml2-)

- Turns out Catch2 is pretty easy to use :-)
- Catch2 has very nice integration into VSCode.
- **All tests pass on CPU as well as on GPU.**

### [Set up documentation.](#neml2-)

1. Configure with `-DDOCUMENTATION=ON`.
2. Run `make doc`.
3. Doxygen is in `doc/doc/class-doc/html/`.

### [Set up coverage report.](#neml2-)

1. Configure with `-DCOVERAGE=ON`. This defines compile and linking options for `lcov`.
2. Recompile application with `make`.
3. Run `./scripts/coverage.sh`.
4. The coverage data will be generated in the `coverage/` directory.

- [Current coverage](coverage/index.html)

### [Set up benchmarks.](#neml2-)

1. Configure with `-DBENCHMARK=ON`.
2. Recompile application with `make`.
3. Run `./scripts/benchmark.sh out.csv` to write the benchmark timings into `out.csv`.
4. Optionally run `python scripts/analyze_timings.py cpu.csv gpu.csv` to compare the timings on CPU and GPU.

Before we look at timings, here are some details about the benchmark setup:
- my CPU: `Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz`
- my GPU: [`Quadro M6000 24GB`](https://images.nvidia.com/content/pdf/quadro/data-sheets/NV-DS-Quadro-M6000-24GB-US-NV-fnl-HR.pdf)
- I am NOT comparing CPU for-loops vs GPU vectorization. Instead, this is a comparison of CPU vectorization vs GPU vectorization.
- Catch2 [`benchmark`](https://github.com/catchorg/Catch2/blob/v2.x/docs/benchmarks.md). The results should be fairly accurate and reproducible (with the same device).

#### [Scalar](#neml2-)

![](benchmark_22_10_07/Benchmark%20Scalar/UScalar+BScalar.png)
![](benchmark_22_10_07/Benchmark%20Scalar/BScalar+BScalar.png)

#### [SymR2](#neml2-)

![](benchmark_22_10_07/Benchmark%20SymR2/USymR2+BSymR2.png)
![](benchmark_22_10_07/Benchmark%20SymR2/BSymR2+BSymR2.png)
![](benchmark_22_10_07/Benchmark%20SymR2/tr(BSymR2).png)
![](benchmark_22_10_07/Benchmark%20SymR2/vol(BSymR2).png)
![](benchmark_22_10_07/Benchmark%20SymR2/dev(BSymR2).png)
![](benchmark_22_10_07/Benchmark%20SymR2/det(BSymR2).png)
![](benchmark_22_10_07/Benchmark%20SymR2/norm(BSymR2).png)
![](benchmark_22_10_07/Benchmark%20SymR2/(USymR2)_ij(BSymR2)_ij.png)
![](benchmark_22_10_07/Benchmark%20SymR2/(BSymR2)_ij(BSymR2)_ij.png)
![](benchmark_22_10_07/Benchmark%20SymR2/(USymR2)_ij(BSymR2)_kl.png)
![](benchmark_22_10_07/Benchmark%20SymR2/(BSymR2)_ij(BSymR2)_kl.png)

#### [SymSymR4](#neml2-)

![](benchmark_22_10_07/Benchmark%20SymSymR4/USymSymR4+BSymSymR4.png)
![](benchmark_22_10_07/Benchmark%20SymSymR4/BSymSymR4+BSymSymR4.png)
![](benchmark_22_10_07/Benchmark%20SymSymR4/(USymSymR4)_ijkl(BSymSymR4)_klmn.png)
![](benchmark_22_10_07/Benchmark%20SymSymR4/(BSymSymR4)_ijkl(BSymSymR4)_klmn.png)
![](benchmark_22_10_07/Benchmark%20SymSymR4/(USymSymR4)_ijkl(BSymR2)_kl.png)
![](benchmark_22_10_07/Benchmark%20SymSymR4/(BSymSymR4)_ijkl(BSymR2)_kl.png)

#### [Constitutive update](#neml2-)

![](benchmark_22_10_07/Benchmark%20linear%20elasticity/update.png)
![](benchmark_22_10_07/Benchmark%20linear%20elasticity/tangent.png)
![](benchmark_22_10_07/Benchmark%20linear%20elasticity/ADtangent.png)

## [Updates Oct. 7, 2022](#neml2-)

### [State vs Force](#neml2-)

- [poll?](https://github.com/reverendbedford/batchedmat/issues/3)

### [Chunked map and parallel coordination](#neml2-)

I'll explain using some slides...

### [The new benchmark](#neml2-)

![](benchmark_22_10_14/Benchmark%20WorkDispatcher/update.png)
