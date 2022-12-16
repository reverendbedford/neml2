# Tensor view and labeled tensor view {#labeledview}

[TOC]

## Tensor view

In defining the forward operator of a constitutive model, many logically different tensors representing inputs, outputs, and the Jacobian have to be created, copied, and destroyed on the fly. These operations occupy a significant amount of computing time, especially on GPUs.

To address this challenge, libTorch (hence NEML2) creates *views*, instead of copies, of tensors whenever possible. As its name suggests, the view of a tensor is a possibly different interpretation of the underlying data. Quoting the PyTorch documentation:

> For a tensor to be viewed, the new view size must be compatible with its original size and stride, i.e., each new view dimension must either be a subspace of an original dimension, or only span across original dimensions \f$d, d+1, ..., d+k\f$ that satisfy the following contiguity-like condition that \f$\forall i = d,...,d+k-1\f$,
> \f[
> \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]
> \f]
> Otherwise, it will not be possible to view self tensor as shape without copying it.

## Working with tensor views

In NEML2, to index a view of a `BatchTensor`, use [base_index](@ref neml2::BatchTensor::base_index) or [batch_index](@ref neml2::BatchTensor::batch_index):
```cpp
BatchTensor<1, 3> A = torch::tensor({{2, 3, 4}, {-1, -2, 3}, {6, 9, 7}});
// A = [[  2  3  4]
//      [ -1 -2  3]
//      [  6  9  7]]
using namespace torch::indexing;
BatchTensor<1, 3> B = A.batch_index({Slice(0, 2)});
// B = [[  2  3  4]
//      [ -1 -2  3]]
BatchTensor<1, 3> C = A.base_index({Slice(1, 3)});
// C = [[  3  4]
//      [ -2  3]
//      [  9  7]]
```
To modify a view of a `BatchTensor`, use [base_index_put](@ref neml2::BatchTensor::base_index_put) or [batch_index_put](@ref neml2::BatchTensor::batch_index_put):
```cpp
A.base_index_put({Slice(1, 3)}, torch::ones({3, 2}));
// A = [[  2  1  1]
//      [ -1  1  1]
//      [  6  1  1]]
A.batch_index_put({Slice(0, 2)}, torch::zeros({2, 3}));
// A = [[  0  0  0]
//      [  0  0  0]
//      [  6  1  1]]
```

Other tensor indexing APIs can be found in the [libTorch documentation](https://pytorch.org/cppdocs/notes/tensor_indexing.html).

## Labeled tensor view

In the context of constitutive modeling, often times views of tensors have practical/physical meanings. For example, given a logically 1D tensor with base size 9, its underlying data in an arbitrary batch may look like
```
equivalent plastic strain   2.1
            cauchy stress  -2.1
                              0
                            1.3
                           -1.1
                            2.5
                            2.5
              temperature 102.9
                     time   3.6
```
where component 0 stores the scalar-valued equivalent plastic strain, components 1-6 stores the tensor-valued cauchy stress, component 7 stores the scalar-valued temperature, and component 8 stores the scalar-valued time.

The string indicating the physical meaning of the view, e.g., "cauchy stress", is called "label", and the view of the tensor indexed by a label is called a "labeled view", i.e.,
```
            cauchy stress  -2.1
                              0
                            1.3
                           -1.1
                            2.5
                            2.5
```

NEML2 provides a data structure named [LabeledAxis](@ref neml2::LabeledAxis) to facilitate the creation and modification of labels, and a data structure named [LabeledTensor](@ref neml2::LabeledTensor) to facilitate the creation and modification of labeled views.

## LabeledAxis

The [LabeledAxis](@ref neml2::LabeledAxis) contains all information regarding how an axis of a `LabeledTensor` is labeled. The following naming convention is used:
- Item: A labelable chunk of data
- Variable: An item that is also of a [NEML2 primitive tensor type](@ref primitive)
- Sub-axis: An item of type `LabeledAxis`

So yes, an axis can be labeled recursively, e.g.,

```
     0 1 2 3 4 5     6     7 8 9 10 11 12   13   14
/// |-----------| |-----| |              | |  | |  |
///    sub a       sub b  |              | |  | |  |
/// |-------------------| |--------------| |--| |--|
///          sub                  a          b    c
```

The above example represents an axis of size 15. This axis has 4 items: `a`, `b`, `c`, and `sub`.
- "a" is a variable of type `SymR2`.
- "b" is a variable of type `Scalar`.
- "c" is a variable of type `Scalar`.
- "sub" is a sub-axis of type `LabeledAxis`. "sub" by itself represents an axis of size 7, containing 2 items:
  - "sub a" is a variable of type `SymR2`.
  - "sub b" is a variable of type `Scalar`.

> Due to performance considerations, a `LabeledAxis` can only be modified, e.g., adding/removing variables and sub-axis, at the time a model is constructed. After the model construction phase, the `LabeledAxis` associated with that model can no longer be modified for the lifetime of the simulation. Refer to the doxygen documentation for a complete list of APIs.

## LabeledTensor

[LabeledTensor](@ref neml2::LabeledTensor) is the primary data structure in NEML2 for working with labeled tensor views. Eacj `LabeledTensor` consists of one `BatchTensor` and multiple `LabeledAxis`s. The `LabeledTensor<N, D>` is templated on the batch dimension `N` and the base dimension `D`. [LabeledVector](@ref neml2::LabeledVector) (derived from `LabeledTensor<1, 1>`) and [LabeledMatrix](@ref neml2::LabeledMatrix) (derived from `LabeledTensor<1, 2>`) are the two most widely used data structures in NEML2.

The doxygen documentation provides a complete list of APIs. The commonly used methods are
- [operator()](@ref neml2::LabeledTensor::operator()()) for retrieving a labeled view without reshaping
- [get](@ref neml2::LabeledTensor::get) for retrieving a labeled view and reshaping it to the correct shape
- [set](@ref neml2::LabeledTensor::set) for setting values for a labeled view
- [slice](@ref neml2::LabeledTensor::slice) for slicing a sub-axis along a specific base dimension
- [block](@ref neml2::LabeledTensor::block) for sub-indexing the `LabeledTensor` with `D` sub-axis names
