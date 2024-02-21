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

// #include "neml2/tensors/macros.h"

namespace py = pybind11;

// Forward declarations
// #define TENSORS_FORWARD_DECL_T(T) void def_##T(py::module_ &)
// FOR_ALL_BATCHTENSORBASE(TENSORS_FORWARD_DECL_T);
void def_BatchTensor(py::module_ &);
void def_Scalar(py::module_ &);
void def_Vec(py::module_ &);
void def_Rot(py::module_ &);
void def_WR2(py::module_ &);
void def_R2(py::module_ &);

void
NEML2_MODULE_TENSORS(py::module_ & M)
{
  auto m = M.def_submodule("tensors");
  m.doc() = "NEML2 primitive tensor types";

  // #define TENSORS_DEF_T(T) def_##T(m)
  //   FOR_ALL_BATCHTENSORBASE(TENSORS_DEF_T);
  def_BatchTensor(m);
  def_Scalar(m);
  def_Vec(m);
  def_Rot(m);
  def_WR2(m);
  def_R2(m);
}
