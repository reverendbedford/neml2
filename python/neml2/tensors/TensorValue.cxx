// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <pybind11/pybind11.h>

#include "neml2/tensors/TensorValue.h"
#include "python/neml2/misc/types.h"

namespace py = pybind11;
using namespace neml2;

void
def_TensorValueBase(py::module_ & m)
{
  py::class_<TensorValueBase>(m, "TensorValueBase")
      .def("set", &TensorValueBase::set)
      // The following accessors/modifiers should mirror BatchTensorBase.h
      .def("tensor", [](const TensorValueBase & self) { return BatchTensor(self); })
      .def("defined", [](const TensorValueBase & self) { return BatchTensor(self).defined(); })
      .def("batched", [](const TensorValueBase & self) { return BatchTensor(self).batched(); })
      .def("dim", [](const TensorValueBase & self) { return BatchTensor(self).dim(); })
      .def_property_readonly("shape",
                             [](const TensorValueBase & self) { return BatchTensor(self).sizes(); })
      .def_property_readonly(
          "dtype", [](const TensorValueBase & self) { return BatchTensor(self).scalar_type(); })
      .def_property_readonly(
          "device", [](const TensorValueBase & self) { return BatchTensor(self).device(); })
      .def_property_readonly("requires_grad",
                             [](const TensorValueBase & self)
                             { return BatchTensor(self).requires_grad(); })
      .def("requires_grad_",
           [](const TensorValueBase & self, bool req)
           { return BatchTensor(self).requires_grad_(req); })
      .def_property_readonly("grad",
                             [](const TensorValueBase & self) { return BatchTensor(self).grad(); });
}
