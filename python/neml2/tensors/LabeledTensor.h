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

#pragma once

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <torch/python.h>

#include "neml2/tensors/LabeledTensor.h"

namespace py = pybind11;

namespace neml2
{

// Forward declarations
template <class Derived, Size D>
void def_LabeledTensor(py::class_<Derived> & c);

} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{

template <class Derived, Size D>
void
def_LabeledTensor(py::class_<Derived> & c)
{
  // Ctors, conversions, accessors etc.
  c.def(py::init<>())
      // I have absolutely no clue as to why the following constructor gives segfault in another
      // totally unrelated class constructor's binding :(
      //  .def(py::init<const torch::Tensor &, Size, const std::array<const LabeledAxis *, D>
      //  &>())
      .def(py::init<const Tensor &, const std::array<const LabeledAxis *, D> &>())
      .def(py::init<const Derived &>())
      .def("__repr__",
           [](const Derived & self)
           {
             std::ostringstream os;
             for (Size i = 0; i < D; i++)
               os << "Axis " << i << ":\n" << self.axis(i) << "\n\n";
             os << self.tensor() << '\n';
             os << "Batch shape: " << self.batch_sizes() << '\n';
             os << " Base shape: " << self.base_sizes() << '\n';
             return os.str();
           })
      .def("tensor", [](const Derived & self) { return self.tensor(); })
      .def("torch", [](const Derived & self) { return torch::Tensor(self); })
      .def("axis", &Derived::axis, py::return_value_policy::reference);
}

} // namespace neml2
