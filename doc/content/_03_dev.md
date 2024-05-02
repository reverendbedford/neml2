# Developer Guide {#dev}

[TOC]

## Tensor types {#tensor-types}

Currently, PyTorch is the only supported tensor backend in NEML2. Therefore, all tensor types in NEML2 directly inherit from `torch::Tensor`. In the future, support for other tensor backends may be added, but the public-facing interfaces will remain largely the same.

### Dynamically shaped tensor {#dynamically-shaped-tensor}

[BatchTensor](@ref neml2::BatchTensor) is a general-purpose *dynamically shaped* tensor type for batched tensors. With a view towards vectorization, the same set of operations can be "simultaneously" applied to a "batch" of (logically the same) tensors. To provide a unified user interface for dealing with such batched operation, NEML2 assumes that the *first* \f$N\f$ dimensions of a tensor are batched dimensions, and the following dimensions are the base (logical) dimensions.

> Unlike PyTorch, NEML2 explicitly distinguishes between batch dimensions and base (logical) dimensions.

A `BatchTensor` can be created using
```cpp
BatchTensor A(torch::rand({1, 1, 5, 2}), 2);
```
where `A` is a tensor with 2 batch dimensions. The batch sizes of `A` is `(1, 1)`:
```cpp
auto batch_sz = A.batch_sizes();
neml2_assert(batch_sz == {1, 1});
```
and the base (logical) sizes of `A` is `(5, 2)`:
```cpp
auto base_sz = A.base_sizes();
neml2_assert(batch_sz == {5, 2});
```

### Statically shaped tensors {#statically-shaped-tensor}

[FixedDimTensor](@ref neml2::FixedDimTensor) is the parent class for all the tensor types with a *fixed* base shape. It is templated on the base shape of the tensor. NEML2 offers a rich collection of primitive tensor types inherited from `FixedDimTensor`. Currently implemented primitive tensor types are summarized below

| Tensor type                            | Base shape        | Description                                                      |
| :------------------------------------- | :---------------- | :--------------------------------------------------------------- |
| [Scalar](@ref neml2::Scalar)           | \f$()\f$          | Rank-0 tensor, i.e. scalar                                       |
| [Vec](@ref neml2::Vec)                 | \f$(3)\f$         | Rank-1 tensor, i.e. vector                                       |
| [R2](@ref neml2::R2)                   | \f$(3,3)\f$       | Rank-2 tensor                                                    |
| [SR2](@ref neml2::SR2)                 | \f$(6)\f$         | Symmetric rank-2 tensor                                          |
| [WR2](@ref neml2::WR2)                 | \f$(3)\f$         | Skew-symmetric rank-2 tensor                                     |
| [R3](@ref neml2::R3)                   | \f$(3,3,3)\f$     | Rank-3 tensor                                                    |
| [SFR3](@ref neml2::SFR3)               | \f$(6,3)\f$       | Rank-3 tensor with symmetry on base dimensions 0 and 1           |
| [R4](@ref neml2::R4)                   | \f$(3,3,3,3)\f$   | Rank-4 tensor                                                    |
| [SSR4](@ref neml2::SSR4)               | \f$(6,6)\f$       | Rank-4 tensor with minor symmetry                                |
| [R5](@ref neml2::R5)                   | \f$(3,3,3,3,3)\f$ | Rank-5 tensor                                                    |
| [SSFR5](@ref neml2::SSFR5)             | \f$(6,6,3)\f$     | Rank-5 tensor with minor symmetry on base dimensions 0-3         |
| [Rot](@ref neml2::Rot)                 | \f$(3)\f$         | Rotation tensor represented in the Rodrigues form                |
| [Quarternion](@ref neml2::Quarternion) | \f$(4)\f$         | Quarternion                                                      |
| [MillerIndex](@ref neml2::MillerIndex) | \f$(3)\f$         | Crystal direction or lattice plane represented as Miller indices |

Furthermore, all primitive tensor types can be "registered" as variables on a `LabeledAxis`, which will be discussed in the following section on [labeled view](@ref tensor-labeling).

## Working with tensors {#working-with-tensors}

### Tensor creation {#tensor-creation}

A factory tensor creation function produces a new tensor. All factory functions adhere to the same schema:
```cpp
<TensorType>::<function_name>(<function-specific-options>, const torch::TensorOptions & options);
```
where `<TensorType>` is the class name of the primitive tensor type listed above, and `<function-name>` is the name of the factory function which produces the new tensor. `<function-specific-options>` are any required or optional arguments a particular factory function accepts. Refer to each tensor type's class documentation for the concrete signature. The last argument `const torch::TensorOptions & options` configures the data type, device, layout and other "meta" properties of the produced tensor. The commonly used meta properties are
- `dtype`: the data type of the elements stored in the tensor. Available options are `kUInt8`, `kInt8`, `kInt16`, `kInt32`, `kInt64`, `kFloat32`, and `kFloat64`.
- `layout`: the striding of the tensor. Available options are `kStrided` (dense) and `kSparse`.
- `device`: the compute device where the tensor will be allocated. Available options are `kCPU` and `kCUDA`.
- `requires_grad`: whether the tensor is part of a function graph used by automatic differentiation to track functional relationship. Available options are `true` and `false`.

For example, the following code
```cpp
auto a = SR2::zeros({5, 3},
                    torch::TensorOptions()
                      .device(torch::kCPU)
                      .layout(torch::kStrided)
                      .dtype(torch::kFloat32));
```
creates a statically (base) shaped, dense, single precision tensor of type `SR2` filled with zeros, with batch shape \f$(5, 3)\f$, allocated on the CPU.

### Tensor broadcasting {#tensor-broadcasting}

Quoting Numpy's definition of broadcasting:

> The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes.

NEML2's broadcasting semantics is largely the same as those of Numpy and PyTorch. However, since NEML2 explicitly distinguishes between batch and base dimensions, the broadcasting semantics must also be extended. Two NEML2 tensors are said to be _batch-broadcastable_ if iterating backward from the last batch dimension, one of the following is satisfied:
1. Both tensors have the same size on the dimension;
2. One tensor has size 1 on the dimension;
3. The dimension does not exist in one tensor.

_Base-broadcastable_ follows a similar definition. Most binary operators on dynamically shaped tensors, i.e., those of type `BatchTensor`, require the operands to be both batch- _and_ base-broadcastable. On the other hand, most binary operators on statically base shaped tensors, i.e., those of pritimitive tensor types, only require the operands to be batch-broadcastable.

### Tensor indexing {#tensor-indexing}

In defining the forward operator of a material model, many logically different tensors representing inputs, outputs, residuals, and Jacobians have to be created, copied, and destroyed on the fly. These operations occupy a significant amount of computing time, especially on GPUs.

To address this challenge, NEML2 creates *views*, instead of copies, of tensors whenever possible. As its name suggests, the view of a tensor is a possibly different interpretation of the underlying data. Quoting the PyTorch documentation:

> For a tensor to be viewed, the new view size must be compatible with its original size and stride, i.e., each new view dimension must either be a subspace of an original dimension, or only span across original dimensions \f$d, d+1, ..., d+k\f$ that satisfy the following contiguity-like condition that \f$\forall i = d,...,d+k-1\f$,
> \f[
> \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]
> \f]
> Otherwise, it will not be possible to view self tensor as shape without copying it.

In NEML2, use [base_index](@ref neml2::BatchTensorBase::base_index) for indexing the base dimensions and [batch_index](@ref neml2::BatchTensorBase::batch_index) for indexing the batch dimensions:
```cpp
using namespace torch::indexing;
BatchTensor A(torch::tensor({{2, 3, 4}, {-1, -2, 3}, {6, 9, 7}}), 1);
// A = [[  2  3  4]
//      [ -1 -2  3]
//      [  6  9  7]]
BatchTensor B = A.batch_index({Slice(0, 2)});
// B = [[  2  3  4]
//      [ -1 -2  3]]
BatchTensor C = A.base_index({Slice(1, 3)});
// C = [[  3  4]
//      [ -2  3]
//      [  9  7]]
```
To modify the content of a tensor, use [base_index_put](@ref neml2::BatchTensorBase::base_index_put) or [batch_index_put](@ref neml2::BatchTensorBase::batch_index_put):
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

### Tensor labeling {#tensor-labeling}

In the context of material modeling, oftentimes views of tensors have practical/physical meanings. For example, given a logically 1D tensor with base size 9, its underlying data in an arbitrary batch may look like
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
where component 0 stores the scalar-valued equivalent plastic strain, components 1-6 store the tensor-valued cauchy stress (we use the Mandel notation for symmetric second order tensors), component 7 stores the scalar-valued temperature, and component 8 stores the scalar-valued time.

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

The [LabeledAxis](@ref neml2::LabeledAxis) contains all information regarding how an axis of a `LabeledTensor` is labeled. The following naming convention is used:
- Item: A labelable slice of data
- Variable: An item that is also of a [NEML2 primitive tensor type](@ref tensor-types)
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
- "a" is a variable of storage size 6 (possibly of type `SR2`).
- "b" is a variable of type `Scalar`.
- "c" is a variable of type `Scalar`.
- "sub" is a sub-axis of type `LabeledAxis`. "sub" by itself represents an axis of size 7, containing 2 items:
  - "a" is a variable of storage size 6.
  - "b" is a variable of type `Scalar`.

Duplicate labels are *not* allowed on the same level of the axis, e.g. "a", "b", "c", and "sub" share the same level and so must be different. However, items on different levels of an axis can share the same label, e.g., "a" on the sub-axis "sub" has the same label as "a" on the main axis. In NEML2 convention, item names are always fully qualified, and a sub-axis is prefixed with a left slash, e.g. item "b" on the sub-axis "sub" can be denoted as "sub/b" on the main axis.

> A label cannot contain: white spaces, quotes, left slash (`/`), or new line.
>
> Due to performance considerations, a `LabeledAxis` can only be modified, e.g., adding/removing variables and sub-axis, at the time a model is constructed. After the model construction phase, the `LabeledAxis` associated with that model can no longer be modified over the entire course of the simulation.

Refer to the documentation for a complete list of APIs for creating and modifying a [LabeledAxis](@ref neml2::LabeledAxis).

[LabeledTensor](@ref neml2::LabeledTensor) is the primary data structure in NEML2 for working with labeled tensor views. Each `LabeledTensor` consists of one `BatchTensor` and one or more `LabeledAxis`s. The `LabeledTensor` is templated on the base dimension \f$D\f$. [LabeledVector](@ref neml2::LabeledVector) and [LabeledMatrix](@ref neml2::LabeledMatrix) are the two most widely used data structures in NEML2.

`LabeledTensor` handles the creation, modification, and accessing of labeled tensors. Recall that all primitive data types in a labeled tensor are flattened, e.g., a symmetric fourth order tensor of type `SSR4` with batch size `(5)` and base size `(6, 6)` are flattened to have base size `(36)` in the labeled tensor. The documentation provides a complete list of APIs. The commonly used methods are
- [operator()](@ref neml2::LabeledTensor::operator()()) for retrieving a labeled view into the raw (flattened) data without reshaping
- [get](@ref neml2::LabeledTensor::get) for retrieving a labeled view and reshaping it to the correct shape
- [set](@ref neml2::LabeledTensor::set) for setting values for a labeled view
- [slice](@ref neml2::LabeledTensor::slice) for slicing a sub-axis along a specific base dimension
- [block](@ref neml2::LabeledTensor::block) for sub-indexing the `LabeledTensor` with \f$D\f$ sub-axis names

## Model {#model}

### Model definition {#model-definition}

A NEML2 model is a function (in the context of mathematics)
\f[
  f: \mathbb{R}^m \to \mathbb{R}^n
\f]
mapping from the input space \f$\mathbb{R}^m\f$ of dimension \f$m\f$ to the output space \f$\mathbb{R}^n\f$ of dimension \f$n\f$. \f$\left[ \cdot \right]\f$ be the flatten-concatenation operator, the input vector is the concatenation of \f$p\f$ flattened variables, i.e.,
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
- [set_value](@ref neml2::Model::set_value) is a method defining the forward operator \f$f\f$ itself.

Both `declare_input_variable` and `declare_output_variable` are templated on the variable type -- recall that only a variable of the NEML2 primitive tensor type can be registered. Furthermore, both methods return a `Variable<T> &` used for retrieving and setting the variable value inside the forward operator, i.e. `set_value`. Note that the reference returned by `declare_input_variable` is writable, while the reference returned by `declare_output_variable` is read-only.

### Model composition {#model-composition}

Quoting [Wikipedia](https://en.wikipedia.org/wiki/Function_composition):
> In mathematics, function composition is an operation \f$\circ\f$ that takes two functions \f$f\f$ and \f$g\f$, and produces a function \f$h = g \circ f\f$ such that \f$h(x) = g(f(x))\f$.

Since NEML2 `Model` is a function (in the context of mathematics) at its core, it should be possible, in theory, to compose different NEML2 `Model`s into a new NEML2 `Model`. The [ComposedModel](@ref neml2::ComposedModel) is precisely for that purpose.

Similar to the statement "a composed function is a function" in the context of mathematics, in NEML2, the equivalent statement "a `ComposedModel` is a `Model`" also holds. In addition, the `ComposedModel` provides four key features to help simplify the composition and reduces computational cost:
- Automatic dependency registration
- Automatic input/output identification
- Automatic dependency resolution
- Automatic chain rule

### A symbolic example {#a-symbolic-example}

To demonstrate the utility of the four key features of `ComposedModel`, let us consider the composition of three functions \f$f\f$, \f$g\f$, and \f$h\f$:
\f{align*}
  y_1 &= f(x_1, x_2), \\
  y_2 &= g(y_1, x_3), \\
  y &= h(y_1, y_2, x_4).
\f}

### Automatic dependency registration {#automatic-dependency-registration}

It is obvious to us that the function \f$h\f$ _depends_ on functions \f$f\f$ and \f$g\f$ because the input of \f$h\f$ depends on the outputs of \f$f\f$ and \f$g\f$. Such dependency is automatically identified and registered while composing a `ComposedModel` in NEML2. This procedure is called "automatic dependency registration".

In order to identify dependencies among different `Model`s, we keep track of the set of _consumed_ variables, \f$\mathcal{I}_i\f$, and a set of _provided_ variables, \f$\mathcal{O}_i\f$, for each `Model` \f$f_i\f$. When a set of models (functions) are composed together, `Model` \f$f_i\f$ is said to _depend_ on \f$f_j\f$ if and only if \f$\exists x\f$ such that
\f[
  x \in \mathcal{I}_i \wedge x \in \mathcal{O}_j.
\f]

### Automatic input/output identification {#automatic-input-output-identification}

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

### Automatic dependency resolution {#automatic-dependency-resolution}

To evaluate the forward operator of the composed model \f$r\f$, one has to first evaluate model \f$f\f$, then model \f$g\f$, and finally model \f$h\f$. The process of sorting out such evaluation order is called "dependency resolution".

While it is possible to sort the evaluation order "by hand" for this simple example composition, it is generally not a trivial task for practical compositions with more involved dependencies. To that end, NEML2 uses [topological sort](https://en.wikipedia.org/wiki/Topological_sorting) to sort the model evaluation order, such that by the time each model is evaluated, all of its dependent models have already been evaluated.

### Automatic chain rule {#automatic-chain-rule}

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

## Developing custom model {#custom-model}

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
Since isotropic hardening _is a_ model, the class inherits from `Model`. The user-facing expected options are defined by the static method `expected_options`. NEML2 handles the parsing of user-specified options and pass them to the constructor. The input variable of the model is the equivalent plastic strain, and the output variable of the model is the isotropic hardening. Their corresponding variable value references are stored as `_ep` and `_h`, respectively, again following the [naming conventions](@ref naming-conventions).

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

### Testing {#testing}

It is of paramount importance to ensure the correctness of the implementation. NEML2 offers 5 types of tests with different purposes.

A Catch2 test refers to a test directly written in C++ source code within the Catch2 framework. It offers the highest level of flexibility, but requires more effort to set up. To understand how a Catch2 test works, please refer to the [official Catch2 documentation](https://github.com/catchorg/Catch2/blob/v2.x/docs/tutorial.md).

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
