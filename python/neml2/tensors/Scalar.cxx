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

#include "python/neml2/tensors/FixedDimTensor.h"

namespace py = pybind11;
using namespace neml2;

void
def_Scalar(py::class_<Scalar> & c)
{
  // Named constructors
  c.def_static(
      "identity_map",
      [](NEML2_TENSOR_OPTIONS_VARGS) { return Scalar::identity_map(NEML2_TENSOR_OPTIONS); },
      py::kw_only(),
      PY_ARG_TENSOR_OPTIONS);

// Binary, unary operators
#define SCALAR_OP(T)                                                                               \
  c.def(T() + py::self)                                                                            \
      .def(py::self + T())                                                                         \
      .def(T() - py::self)                                                                         \
      .def(py::self - T())                                                                         \
      .def(T() * py::self)                                                                         \
      .def(py::self * T())                                                                         \
      .def(py::self * py::self)                                                                    \
      .def(T() / py::self)                                                                         \
      .def(py::self / T())                                                                         \
      .def("__pow__", [](const Scalar & a, const T & b) { return math::pow(a, b); })               \
      .def("__rpow__", [](const Scalar & b, const T & a) { return math::pow(a, b); })
  FOR_ALL_BATCHTENSORBASE(SCALAR_OP);
}
