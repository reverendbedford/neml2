# Python Package {#python-package}

> **Note**
>
> The NEML2 Python package is experimental. APIs are expected to change.

With the NEML2 Python package, the same input file in the other [tutorial](#cpp-backend) can be directly used in a Python script. The Python APIs closely ressembles the C++ APIs. For example, the above C++ source code translates to the following Python script.

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
