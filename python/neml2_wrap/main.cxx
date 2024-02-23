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

namespace py = pybind11;

// Each module, submodule, class, and method have a corresponding binding method.
// For example, the "tensors" module contains all of the bindings related to NEML2 primitive
// tensors, and is defined by the NEML2_MODULE_TENSORS_XXX methods.

// The module definition methods are grouped into two parts: declaration and definition. A natural
// question to ask is why bother separating declarations from definitions? The reason is for typing:
// Once we build the neml2 python library with all the necessary bindings, we will have to extract
// all the typing information (mostly function signature) from the library, which is needed by
// language servers like Pylance. We use pybind11-stubgen for that purpose. For a type to be
// deducible by pybind11-stubgen, a concrete definition of the binding class must exist at the point
// of method definition. Therefore, we need to first create all the class definitions before
// creating method bindings that use them as arguments.

// Forward declarations
void NEML2_MODULE_TENSORS(py::module_ &);
void NEML2_MODULE_MATH(py::module_ &);

PYBIND11_MODULE(neml2, m)
{
  py::module::import("torch");

  m.doc() = "NEML2, GPU-enabled vectorized material modeling library";

  NEML2_MODULE_MATH(m_math);
  NEML2_MODULE_TENSORS(m_tensors);
}
