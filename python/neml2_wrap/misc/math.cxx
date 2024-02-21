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

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/macros.h"

namespace py = pybind11;
using namespace neml2;

void
NEML2_MODULE_MATH(py::module_ & M)
{
  auto m = M.def_submodule("math");
  m.doc() = "NEML2 mathematical functions and utilities";

  // Methods
  m.def("bmm", &math::bmm);

  // Templated methods
  // These methods are special because the argument could be anything derived from BatchTensorBase,
  // so we need to bind every possible instantiation.
#define MATH_DEF_BATCHTENSORBASE(T)                                                                \
  m.def("sign", &math::sign<T>)                                                                    \
      .def("heaviside", &math::heaviside<T>)                                                       \
      .def("macaulay", &math::macaulay<T>)                                                         \
      .def("dmacaulay", &math::dmacaulay<T>)                                                       \
      .def("sqrt", &math::sqrt<T>)                                                                 \
      .def("exp", &math::exp<T>)                                                                   \
      .def("abs", &math::abs<T>)

  FOR_ALL_BATCHTENSORBASE(MATH_DEF_BATCHTENSORBASE);
}
