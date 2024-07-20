# Python Package {#python-package}

\note
The NEML2 Python package is experimental. APIs are expected to change.

With the NEML2 Python package, the same input file in the other [tutorial](#cpp-backend) can be directly used in a Python script. The Python APIs closely ressembles the C++ APIs. For example, the previous C++ example translates to the following Python script.

```python
import neml2

model = neml2.load_model("input.i", "model")

model.reinit(3)

x = neml2.LabeledVector.empty(3, [model.input_axis()])

x.batch[0] = neml2.SR2.fill([0.1, 0.2, 0.3, -0.1, -0.1, 0.2])
x.batch[1] = neml2.SR2.fill([0.2, 0.2, 0.1, -0.1, -0.2, -0.5])
x.batch[2] = neml2.SR2.fill([0.3, -0.2, 0.05, -0.1, -0.3, 0.1])

y = model.value(x)
```

As is the same in the equivalent C++ example, the above Python script parses the input file named "input.i", loads the linear elasticity model named "model", constructs 3 strain tensors, and finally performs the 3 material updates simultaneously.
