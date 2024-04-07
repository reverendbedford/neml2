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
#include <pybind11/stl.h>

#include "neml2/tensors/LabeledAxisAccessor.h"
#include "neml2/misc/utils.h"

namespace py = pybind11;
using namespace neml2;

void
def_LabeledAxisAccessor(py::module_ & m)
{
  auto c = py::class_<LabeledAxisAccessor>(m, "LabeledAxisAccessor");

  // Ctors
  c.def(py::init<>())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, const std::string &>())
      .def(py::init<const std::string &, const std::string &, const std::string &>())
      .def(py::init<const std::vector<std::string> &>())
      .def(py::init<const LabeledAxisAccessor &>())
      .def("empty", &LabeledAxisAccessor::empty)
      .def("size", &LabeledAxisAccessor::size)
      .def("with_suffix", &LabeledAxisAccessor::with_suffix)
      .def("append", &LabeledAxisAccessor::append)
      .def("on", &LabeledAxisAccessor::on)
      .def("start_with", &LabeledAxisAccessor::start_with);

  // Operators
  c.def("__repr__", [](const LabeledAxisAccessor & self) { return utils::stringify(self); })
      .def("__eq__",
           [](const LabeledAxisAccessor & a, const LabeledAxisAccessor & b) { return a == b; })
      .def("__ne__",
           [](const LabeledAxisAccessor & a, const LabeledAxisAccessor & b) { return a == b; });
}
