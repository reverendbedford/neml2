# Implementation {#impl}

[TOC]

## NEML2 tensor types {#primitive}

Currently, libTorch is the only supported tensor backend in NEML2. Therefore, all tensor types in NEML2 directly inherit from `torch::Tensor`. In the future, support for other tensor backend libraries may be added, but the public-facing interfaces will remain largely the same.

### BatchTensor

[BatchTensor](@ref neml2::BatchTensor) is a general purpose tensor type for batched `torch::Tensor`s. With a view towards vectorization, the same set of operations can be "simultaneously" applied to a large "batch" of (logically the same) tensors. To provide a unified user interface for dealing with such batched operation, NEML2 assumes that the *first* \f$N\f$ dimensions of a tensor are batched dimensions, and the following dimensions are the base (logical) dimensions.

> Unlike libTorch, NEML2 explicitly distinguishes between batch dimensions and base (logical) dimensions.

The `BatchTensor` is templated on the number of batch dimensions \f$N\f$. Although the number of batched dimensions is known at compile time, the size of each dimension is not. The batch dimensions can be reshaped at runtime. For example, a `BatchTensor` can be created as
```cpp
BatchTensor<2> A = torch::rand({1, 1, 5, 2});
```
where `A` is a tensor with 2 batch dimensions. The batch sizes of `A` is `(1, 1)`:
```cpp
auto batch_sz = A.batch_sizes();
// batch_sz == {1, 1}
```
and the base (logical) sizes of `A` is `(5, 2)`:
```cpp
auto base_sz = A.base_sizes();
// batch_sze == {5, 2}
```
The base tensor can be reshaped (e.g., expanded and copied) at runtime along its batch dimensions using
```cpp
BatchTensor<2> B = A.batch_expand_copy({3, 4});
auto new_batch_sz = B.batch_sizes();
// new_batch_sz == {3, 4}
```

### FixedDimTensor

[FixedDimTensor](@ref neml2::FixedDimTensor) inherits from `BatchTensor`. It is additionally templated on the sizes of the base dimensions. For example,
```cpp
static_assert(FixedDimTensor<2, 6>::const_base_sizes == {6});
```

### Primitive tensor types

All primitive tensor types inherit from `FixedDimTensor` with a *single* batch dimension. Currently implemented primitive tensor types include
- [Scalar](@ref neml2::Scalar), a (batched) scalar quantity derived from the specialization `FixedDimTensor<1, 1>`
- [SymR2](@ref neml2::SymR2), a (batched) symmetric second order tensor derived from the specialization `FixedDimTensor<1, 6>`
- [SymSymR4](@ref neml2::SymSymR4), a (batched) symmetric fourth order tensor derived from the specialization `FixedDimTensor<1, 6, 6>`

Furthermore, all primitive tensor types can be "registered" as variables on a `LabeledAxis`, which will be discussed in the following section on [labeled view](@ref labeledview).

## Tensor view and label {#labeledview}

### Tensor view

In defining the forward operator of a constitutive model, many logically different tensors representing inputs, outputs, residuals, and Jacobians have to be created, copied, and destroyed on the fly. These operations occupy a significant amount of computing time, especially on GPUs.

To address this challenge, NEML2 creates *views*, instead of copies, of tensors whenever possible. As its name suggests, the view of a tensor is a possibly different interpretation of the underlying data. Quoting the PyTorch documentation:

> For a tensor to be viewed, the new view size must be compatible with its original size and stride, i.e., each new view dimension must either be a subspace of an original dimension, or only span across original dimensions \f$d, d+1, ..., d+k\f$ that satisfy the following contiguity-like condition that \f$\forall i = d,...,d+k-1\f$,
> \f[
> \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]
> \f]
> Otherwise, it will not be possible to view self tensor as shape without copying it.

### Working with tensor views

In NEML2, to index a view of a `BatchTensor`, use [base_index](@ref neml2::BatchTensor::base_index) for indexing the base dimensions and [batch_index](@ref neml2::BatchTensor::batch_index) for indexing the batch dimensions:
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
A detailed explanation on tensor indexing APIs is available as part of the official [PyTorch documentation](https://pytorch.org/cppdocs/notes/tensor_indexing.html).

### Labeled tensor view

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
where component 0 stores the scalar-valued equivalent plastic strain, components 1-6 store the tensor-valued cauchy stress (recall that we use the [Mandel](@ref mandel) notation for symmetric second order tensors), component 7 stores the scalar-valued temperature, and component 8 stores the scalar-valued time.

The string indicating the physical meaning of the view, e.g., "cauchy stress", is called a "label", and the view of the tensor indexed by a label is called a "labeled view", i.e.,
```
            cauchy stress  -2.1
                              0
                            1.3
                           -1.1
                            2.5
                            2.5
```

NEML2 provides a data structure named [LabeledAxis](@ref neml2::LabeledAxis) to facilitate the creation and modification of labels, and a data structure named [LabeledTensor](@ref neml2::LabeledTensor) to facilitate the creation and modification of labeled views.

### LabeledAxis

The [LabeledAxis](@ref neml2::LabeledAxis) contains all information regarding how an axis of a `LabeledTensor` is labeled. The following naming convention is used:
- Item: A labelable chunk of data
- Variable: An item that is also of a [NEML2 primitive tensor type](@ref primitive)
- Sub-axis: An item of type `LabeledAxis`

So yes, an axis can be labeled recursively, e.g.,

```
     0 1 2 3 4 5     6     7 8 9 10 11 12   13   14
/// |-----------| |-----| |              | |  | |  |
///       a          b    |              | |  | |  |
/// |-------------------| |--------------| |--| |--|
///          sub                  a          b    c
```
The above example represents an axis of size 15. This axis has 4 items: `a`, `b`, `c`, and `sub`.
- "a" is a variable of storage size 6 (possibly of type `SymR2`).
- "b" is a variable of type `Scalar`.
- "c" is a variable of type `Scalar`.
- "sub" is a sub-axis of type `LabeledAxis`. "sub" by itself represents an axis of size 7, containing 2 items:
  - "a" is a variable of storage size 6.
  - "b" is a variable of type `Scalar`.

Duplicate labels are *not* allowed on the same level of the axis, e.g. "a", "b", "c", and "sub" share the same level and so must be different. However, items on different levels of an axis can share the same label, e.g., "a" on the sub-axis "sub" has the same label as "a" on the main axis. In NEML2 convention, item names are always fully qualified, and a sub-axis is prefixed with a left slash, e.g. item "b" on the sub-axis "sub" can be denoted as "sub/b" on the main axis.

> A label cannot contain: white spaces, quotes, left slash (`/`), or new line. 

> Due to performance considerations, a `LabeledAxis` can only be modified, e.g., adding/removing variables and sub-axis, at the time a model is constructed. After the model construction phase, the `LabeledAxis` associated with that model can no longer be modified over the entire course of the simulation.

Refer to the doxygen documentation for a complete list of APIs for creating and modifying a [LabeledAxis](@ref neml2::LabeledAxis).

### LabeledTensor

[LabeledTensor](@ref neml2::LabeledTensor) is the primary data structure in NEML2 for working with labeled tensor views. Each `LabeledTensor` consists of one `BatchTensor` and one or more `LabeledAxis`s. The `LabeledTensor<N, D>` is templated on the batch dimension \f$N\f$ and the base dimension \f$D\f$. [LabeledVector](@ref neml2::LabeledVector) (derived from `LabeledTensor<1, 1>`) and [LabeledMatrix](@ref neml2::LabeledMatrix) (derived from `LabeledTensor<1, 2>`) are the two most widely used data structures in NEML2.

`LabeledTensor` handles the creation, modification, and accessing of labeled tensors. Recall that all primitive data types in a labeled tensor are flattened, e.g., a symmetric fourth order tensor of type `SymSymR4` with batch size `(5)` and base size `(6, 6)` are flattened to have base size `(36)` in the labeled tensor. The doxygen documentation provides a complete list of APIs. The commonly used methods are
- [operator()](@ref neml2::LabeledTensor::operator()()) for retrieving a labeled view into the raw (flattened) data without reshaping
- [get](@ref neml2::LabeledTensor::get) for retrieving a labeled view and reshaping it to the correct shape
- [set](@ref neml2::LabeledTensor::set) for setting values for a labeled view
- [slice](@ref neml2::LabeledTensor::slice) for slicing a sub-axis along a specific base dimension
- [block](@ref neml2::LabeledTensor::block) for sub-indexing the `LabeledTensor` with \f$D\f$ sub-axis names

## Model definition {#modeldefinition}

A NEML2 model is a function (in the context of mathematics)
\f[
  f: \mathbb{R}^m \to \mathbb{R}^n
\f]
mapping from the input space \f$\mathbb{R}^m\f$ of dimension \f$m\f$ to the output space \f$\mathbb{R}^n\f$ of dimension \f$n\f$. Recall that \f$\left[ \cdot \right]\f$ is the [flatten-concatenation operator](@ref math). The input vector is the concatenation of \f$p\f$ flattened variables, i.e.,
\f[
  x = \left[ x_i \right]_{i=1}^p \in \mathbb{R}^m, \quad \sum_{i=1}^p \lvert x_i \rvert = m,
\f]
where \f$\lvert x \rvert\f$ denotes the modulus of flattened variable \f$x\f$. Similarly, the output vector is the concatenation of \f$q\f$ flattened variables, i.e.,
\f[
  y = \left[ y_i \right]_{i=1}^q \in \mathbb{R}^n, \quad \sum_{i=1}^q \lvert y_i \rvert = n.
\f]

Translating the above mathematical definition into NEML2 is straightforward.
- A model following this definition derives from [Model](@ref neml2::Model).
- [declare_input_variable](@ref neml2::Model::declare_input_variable) declares an input variable \f$x_i\f$ in the input space \f$\mathbb{R}^m\f$.
- [declare_output_variable](@ref neml2::Model::declare_output_variable) declares an output variable \f$y_i\f$ in the output space \f$\mathbb{R}^n\f$.
- [set_value](@ref neml2::Model::set_value) is a method defining the forward operator \f$f\f$.

Both `declare_input_variable` and `declare_output_variable` are templated on the variable type -- recall that only a variable of the NEML2 primitive tensor type can be registered. Furthermore, both calls return a convenient accessor of type [LabeledAxisAccessor](@ref neml2::LabeledAxisAccessor) which can be later used to retrieve/modify the labeled view of the input/output vector.

> Declaration of the variables don't immediately set up the layout of the input/output `LabeledAxis`. The method [setup](@ref neml2::Model::setup) should be explicitly called in order to set up the memory layout of the `LabeledAxis`s. **Note that [setup](@ref neml2::Model::setup) must be called after all the variables have been added, and before the forward operator of the `Model` can be used.**

## Model composition {#modelcomposition}

Quoting [Wikipedia](https://en.wikipedia.org/wiki/Function_composition):
> In mathematics, function composition is an operation \f$\circ\f$ that takes two functions \f$f\f$ and \f$g\f$, and produces a function \f$h = g \circ f\f$ such that \f$h(x) = g(f(x))\f$.

Since NEML2 `Model` is a function (in the context of mathematics) at its core, it should be possible, in theory, to compose different NEML2 `Model`s into a new NEML2 `Model`. The [ComposedModel](@ref neml2::ComposedModel) is precisely for that purpose.

Similar to the statement "a composed function is a function" in the context of mathematics, in NEML2, the equivalent statement "a ComposedModel is a Model" also holds. In addition, the `ComposedModel` provides four key features to help simplify the composition and reduces computational cost:
- Automatic dependency registration
- Automatic input/output identification
- Automatic dependency resolution
- Automatic chain rule

### A symbolic example

To demonstrate the utility of the four key features of `ComposedModel`, let us consider the composition of three functions \f$f\f$, \f$g\f$, and \f$h\f$:
\f{align*}
  y_1 &= f(x_1, x_2), \\
  y_2 &= g(y_1, x_3), \\
  y &= h(y_1, y_2, x_4).
\f}

### Automatic dependency registration

It is obvious to us that the function \f$h\f$ _depends_ on functions \f$f\f$ and \f$g\f$ because the input of \f$h\f$ depends on the outputs of \f$f\f$ and \f$g\f$. Such dependency is automatically identified and registered while composing a `ComposedModel` in NEML2. This procedure is called "automatic dependency registration".

In order to identify dependencies among different `Model`s, we keep track of the set of _consumed_ variables, \f$\mathcal{I}_i\f$, and a set of _provided_ variables, \f$\mathcal{O}_i\f$, for each `Model` \f$f_i\f$. When a set of models (functions) are composed together, `Model` \f$f_i\f$ is said to _depend_ on \f$f_j\f$ if and only if \f$\exists x\f$ such that
\f[
  x \in \mathcal{I}_i \wedge x \in \mathcal{O}_j.
\f]

### Automatic input/output identification

The only possible composition \f$r\f$ of these three functions is
\f[
  y = r(x_1, x_2, x_3, x_4) := h(f(x_1, x_2), g(f(x_1, x_2), x_3), x_4).
\f]
The input variables of the composed function \f$r\f$ are \f$[x_1, x_2, x_3, x_4]\f$ (or their flattened concatenation), and the output variable of the composed function is simply \f$y\f$. The input/output variables are automatically identified while composing a `ComposedModel` in NEML2. This procedure is referred to as "automatic input/output identification".

In a `ComposedModel`, a _leaf_ model is a model which does not depend on any other model, and a _root_ model is a model which is not depent upon by any other model. A `ComposedModel` may have multiple leaf models and multiple root models. An input variable is said to be a _root_ input variable if it is not consumed by any other model, i.e. \f$x \in \mathcal{I}_i\f$ is a root input variable if and only if
\f[
    x \notin \mathcal{O}_j, \quad \forall i \neq j.
\f]
Similarly, an output variable is said to be a _leaf_ output variable if it is not provided by any other model, i.e. \f$x \in \mathcal{O}_i\f$ is a leaf output variable if an only if
\f[
    x \notin \mathcal{I}_j, \quad \forall i \neq j.
\f]
The input variables of a `ComposedModel` is the union of the set of all the root input variables, and the output variables of a `ComposedModel` is the set of all the leaf output variables.

### Automatic dependency resolution

To evaluate the forward operator of the composed model \f$r\f$, one has to first evaluate model \f$f\f$, then model \f$g\f$, and finally model \f$h\f$. The process of sorting out such evaluation order is called "dependency resolution".

While it is possible to sort the evaluation order "by hand" for this simple example composition, it is generally not a trivial task for practical compositions with more involved dependencies. To that end, NEML2 uses [topological sort](https://en.wikipedia.org/wiki/Topological_sorting) to sort the model evaluation order, such that by the time each model is evaluated, all of its dependent models have already been evaluated.

### Automatic chain rule

Chain rule can be applied to evaluate the derivative of the forward operator with respect to the input variables, i.e.,
\f{align*}
  \frac{\partial y}{\partial x_1} &= \left( \frac{\partial y}{\partial y_1} + \frac{\partial y}{\partial y_2} \frac{\partial y_2}{\partial y_1} \right) \frac{\partial y_1}{\partial x_1}, \\
  \frac{\partial y}{\partial x_2} &= \left( \frac{\partial y}{\partial y_1} + \frac{\partial y}{\partial y_2} \frac{\partial y_2}{\partial y_1} \right) \frac{\partial y_1}{\partial x_2}, \\
  \frac{\partial y}{\partial x_3} &= \frac{\partial y}{\partial y_2} \frac{\partial y_2}{\partial x_3}, \\
  \frac{\partial y}{\partial x_4} &= \frac{\partial y}{\partial x_4}.
\f}
Spelling out this chain rule can be cumbersome and error-prone, especially for more complicated model compositions. The evaluation of the chain rule is automated in NEML2, and the user is only responsible for implementing the partial derivatives of each model. For example, in the implementation of `Model` \f$f\f$, the user only need to define the partial derivatives
\f[
  \frac{\partial y_1}{\partial x_1}, \quad \frac{\partial y_1}{\partial x_2};
\f]
similarly, `Model` \f$g\f$ only defines
\f[
  \frac{\partial y_2}{\partial y_1}, \quad \frac{\partial y_2}{\partial x_3}
\f]
and `Model` \f$h\f$ only defines
\f[
  \frac{\partial y}{\partial y_1}, \quad \frac{\partial y}{\partial y_2}, \quad \frac{\partial y}{\partial x_4}.
\f]
The assembly of the partial derivatives into the total derivative \f$\partial y / \partial \mathbf{x}\f$ using the chain rule is handled by NEML2. This design serves as the fundation for a modular model implementation:
- Each model _does not_ need to know its composition with others.
- The same model partial derivatives can be reused in _any_ composition.
