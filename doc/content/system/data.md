# Data {#data}

[TOC]

## Buffer

The term "buffer" is inherited from PyTorch. In a `torch.nn.Module` (which can be thought of the PyTorch equivalent of NEML2's `Model`), a set of _parameters_ and _buffer_ can be declared. Both parameters and buffer are tensors registered to the model. When the model is sent to a different device, i.e., from CPU to GPU, the parameters and buffer registered with the model are sent to the target device.

The only notable difference between buffer and parameter is that buffer tensors are _NOT_ meant to part of the function graph (while calling the model's forward operator), while parameters are expected to be differentiated. Some examples of buffer, in the context of crystal plasticity, include crystal class, lattice vectors, slip planes, and slip directions of a crystal.

## Data

NEML2 provides the Data system to predefine commonly used buffer tensors that are oftentimes shared among material models. Data objects shall inherit from the base class `Data` and can register as many buffer tensors as necessary.

Objects that are defined as part of the Data system are different from models in several ways:
1. Data objects _do not_ contain parameters.
2. Data objects _do not_ define variables.
3. Data objects _do not_ define forward operator.
4. Data objects _do not_ participate in dependency resolution.

A model can register a Data object using the `register_data` method which returns a reference to the registered `Data`.
