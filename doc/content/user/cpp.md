# C++ Backend {#cpp-backend}

[TOC]

## Loading a model from an input file

The following input file defines a linear isotropic elasticity material model:

```python
[Models]
  [model]
    type = LinearIsotropicElasticity
    youngs_modulus = 100
    poisson_ratio = 0.3
    strain = 'forces/E'
    stress = 'state/S'
  []
[]
```

The input file defines two parameters: Young's modulus of 100 and Poisson's ratio of 0.3. While optional, the input file also sets the variable names of strain and stress to be "forces/E" and "state/S", respectively (refer to the documentation on [tensor labeling](@ref tensor-labeling) for variable naming conventions).

Assuming the above input file is named "input_file.i", the C++ code snippet below parses the input file and loads the material model (into the heap).

```cpp
#include "neml2/base/Parser.h"
#include "neml2/base/Factory.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

int main() {
  load_model("input.i");
  auto & model = get_model("model");

  // ...

  return 0;
}
```

## Evaluate the model

Suppose we want to perform 3 material updates simultaneously, the model should be initialized using the neml2::Model::reinit method with the correct batch shape (refer to the [tensor system documentation](@ref system-tensors) for more detailed explanation on the term "batch"):

```cpp
  model.reinit({3});
```

Finally, the following code constructs the 3 input strains `in` and performs 3 material updates _simultaneously_. Output stresses are stored in the tensor `out`.

```cpp
  auto in = LabeledVector::empty({3}, {model.input_axis()});

  in.batch_index_put({0}, SR2::fill(0.1, 0.2, 0.3, -0.1, -0.1, 0.2));
  in.batch_index_put({1}, SR2::fill(0.2, 0.2, 0.1, -0.1, -0.2, -0.5));
  in.batch_index_put({2}, SR2::fill(0.3, -0.2, 0.05, -0.1, -0.3, 0.1));

  auto out = model.value(in);
```

## Inference Mode {#inference-mode}

By default, inference mode is disabled. During every model evaluation, variables are re-allocated, and variable views are reconfigured. This default behavior avoids in-place operations and is mandatory to use PyTorch automatic differentiation (AD) (i.e. for parameter gradient or AD forward operator).

The default behavior is flexible. However, if PyTorch AD is not needed, i.e., once the model is fully calibrated, the variable reallocation and view reconfiguration can be skipped. The inference mode is designed for that purpose.

When inference mode is activated, no function graph is built, tensor version numbers are not incremented, and the variables and variable views are setup once and for all, all of which speed up evaluation. See https://pytorch.org/cppdocs/notes/inference_mode.html for more details.

To enable inference mode, use the optional second argument passed to `neml2::get_model`, i.e.
```cpp
auto & model = get_model("model", /*inference_mode=*/true);
```
