# NEML2 tensor types {#primitive}

[TOC]

Currently, libTorch is the only supported tensor backend in NEML2. Therefore, all tensor types in NEML2 directly inherit from `torch::Tensor`. In the future, support for other tensor backend libraries may be added, but the public-facing interfaces will remain largely the same.

## BatchTensor

[BatchTensor](@ref neml2::BatchTensor) is a general purpose tensor type for batched `torch::Tensor`s. With a view towards vectorization, the same set of operations can be simultaneously applied to a large "batch" of (logically the same) tensors. To provide a unified user interface for dealing with such batched operation, NEML2 assumes that the *first* `N` dimensions of a tensor are batched dimensions, while the following dimensions are the base (logical) dimensions.

> Unlike libTorch, NEML2 explicitly distinguishes between batch dimensions and base (logical) dimensions.

The `BatchTensor` is templated on the number of batched dimensions `N`. Although the number of batched dimensions is known at compile time, the size of each dimension is not. The batch dimensions can be reshaped at runtime. For example, a `BatchTensor` can be created as
```cpp
BatchTensor<2> A = torch::rand({1, 1, 5, 2});
```
where `A` is a tensor with 2 batch dimensions. The batch sizes of `A` is `(1, 1,)`:
```cpp
auto batch_sz = A.batch_sizes();
// batch_sze == {1, 1}
```
and the base (logical) sizes of `A` is `(5, 2,)`:
```cpp
auto base_sz = A.base_sizes();
// batch_sze == {5, 2}
```
The base tensor can be reshaped (expanded and copied) at runtime along its batch dimensions using
```cpp
BatchTensor<2> B = A.batch_expand_copy({3, 4});
auto new_batch_sz = B.batch_sizes();
// new_batch_sz == {3, 4}
```

## FixedDimTensor

[FixedDimTensor](@ref neml2::FixedDimTensor) inherits from `BatchTensor`. It is additionally templated on the sizes of the base dimensions. For example,
```cpp
static_assert(FixedDimTensor<2, 6>::_base_sizes == {6});
```

## Primitive tensor types

All primitive tensor types inherit from `FixedDimTensor` with a *single* batch dimension. Currently implemented primitive tensor types include
- [Scalar](@ref neml2::Scalar), a (batched) scalar quantity derived from the specialization `FixedDimTensor<1, 1>`
- [SymR2](@ref neml2::SymR2), a (batched) symmetric second order tensor derived from the specialization `FixedDimTensor<1, 6>`
- [SymSymR4](@ref neml2::SymSymR4), a (batched) symmetric fourth order tensor derived from the specialization `FixedDimTensor<1, 6, 6>`

Furthermore, all primitive tensor types can be "registered" as variables on a `LabeledAxis`, which will be discussed in the following article on [labeled view](@ref labeledview).
