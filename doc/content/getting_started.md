# Getting Started {#getting-started}

[TOC]

## Using NEML2 material models

The user interface of NEML2 is designed in such a way that no programing experience is required to compose custom material models and define how they are solved. This is achieved using _input files_. The input files are simply text files with a specific format that NEML2 can understand. NEML2 can _deserialize_ an input file, i.e., parse and create material models specified within the input file.

Since the input files are nothing more than text files saved on the disk, they can be used in any application that supports standard IO, easily exchanged among different devices running different operating systems, and archived for future reference.

### Input file syntax {#input-file-syntax}

Input files use the Hierarchical Input Text (HIT) format. The syntax looks like this:
```python
# Comments look like this
[block1]
  # This is a comment
  foo = 1
  bar = 3.14159
  baz = 'string value'
  [nested_block]
    # ...
  []
[]
```
where key-value pairs are defined under (nested) blocks denoted by square brackets. A value can be an integer, floating-point number, string, or array (as indicated by single quotes). Note that the block indentation is recommended for clarity but is not required.

All NEML2 capabilities that can be defined through the input file fall under a number of _systems_. Names of the top-level blocks specify the systems. For example, the following input file
```python
[Tensors]
  [E]
    # ...
  []
[]

[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 'E'
    poisson_ratio = 0.3
  []
[]
```
defines a tensor named "E" under the `[Tensors]` block and a model named "elasticity" under the `[Models]` block. The [Syntax Documentation](@ref syntax-tensors) provides a complete list of objects that can be defined by an input file. The [System Documentation](@ref system-tensors) provides detailed explanation of each system.

### Special syntax

**Boolean**: Oftentimes the behavior of the object is preferrably controlled by a boolean flag. However, since the HIT format only allows (array of) integer, floating-point number, and string, a special syntax shall be reserved for boolean values. In NEML2 input files, a string with value "true" can be parsed into a boolean `true`, and a string with value "false" can be parsed into a boolean `false`.

> On the other hand, other commonly used boolean flags such as "on"/"off", "1"/"0", "True"/"False", etc., cannot be parsed into boolean values. Trying to do so will trigger a `ParserException`.

**Variable name**: NEML2 material models work with named variables to assign physical meanings to different slices of a tensor (see e.g. [Tensor Labeling](@ref tensor-labeling)). A fully qualified variable name can be parsed from a string, and the delimiter "/" signifies nested sub-axes. For example, the string "forces/t" can be parsed into a variable named "t" defined on the sub-axis named "forces".

**Tensor shape**: Shape of a tensor can also be parsed from a string. The string must start with "(" and end with ")". An array of comma-separated integers must be enclosed by the parentheses. For example, the string "(5,6,7)" can be parsed into a shape tuple of value `(5, 6, 7)`. Note that white spaces are not allowed between the parentheses and could lead to undefined behavior. An empty array, i.e. "()", however, is allowed and fully supported.

## Using NEML2 as a C++ library

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
  auto & model = Factory::get_object<Model>("Models", "model");

  // ...

  return 0;
}
```

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

## Using the NEML2 Python package

With the NEML2 Python package, the above input file can be directly used in a Python script. The Python APIs closely ressembles the C++ APIs. For example, the above C++ source code translates to the following Python script.

```python
import torch
import neml2
from neml2.tensors import SR2, LabeledVector

model = neml2.load_model("input.i", "model")

model.reinit(3)

x = LabeledVector.empty(3, [model.input_axis()])

x.batch_index_put(0, SR2.fill([0.1, 0.2, 0.3, -0.1, -0.1, 0.2]))
x.batch_index_put(1, SR2.fill([0.2, 0.2, 0.1, -0.1, -0.2, -0.5]))
x.batch_index_put(2, SR2.fill([0.3, -0.2, 0.05, -0.1, -0.3, 0.1]))

y = model.value(x)
```

As is the same in the equivalent C++ source code, the above Python script parses the input file named "input.i", loads the linear elasticity model named "model", constructs 3 strain tensors, and finally performs the 3 material updates simultaneously.
