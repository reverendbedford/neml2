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

#include "neml2/tensors/macros.h"

#include "python/neml2/tensors/TensorBase.h"
#include "python/neml2/tensors/LogicalTensor.h"
#include "python/neml2/tensors/VecBase.h"
#include "python/neml2/tensors/R2Base.h"

namespace py = pybind11;
using namespace neml2;

// Forward declarations
#define TENSOR_CUSTOM_DEF_FWD(T) void def_##T(py::class_<T> &)
FOR_ALL_TENSORBASE(TENSOR_CUSTOM_DEF_FWD);

void def_LabeledAxisAccessor(py::module_ & m);
void def_LabeledAxis(py::module_ & m);
void def_LabeledVector(py::module_ & m);
void def_LabeledMatrix(py::module_ & m);

// Instantiate global types
namespace pybind11
{
namespace detail
{
template struct type_caster<c10::SmallVector<Size>>;
template struct type_caster<c10::SmallVector<at::indexing::TensorIndex>>;
}
}

PYBIND11_MODULE(tensors, m)
{
  m.doc() = "NEML2 primitive tensor types";

  // Export enums
  auto tensor_type_enum = py::enum_<TensorType>(m, "TensorType");
#define TENSORTYPE_ENUM_ENTRY(T) tensor_type_enum.value(#T, TensorType::k##T)
  FOR_ALL_TENSORBASE(TENSORTYPE_ENUM_ENTRY);

  // Declare all the TensorBase derived tensors
  // This is as simple as calling py::class_, but it is important to do this for ALL tensors up
  // front. The reason is for typing: Once we build the neml2 python library with all the necessary
  // bindings, we will have to extract all the typing information (mostly function signature) from
  // the library, which is needed by language servers like Pylance. We use pybind11-stubgen for that
  // purpose. For a type to be deducible by pybind11-stubgen, a concrete definition of the binding
  // class must exist at the point of method definition. Therefore, we need to first create all the
  // class definitions before creating method bindings that use them as arguments.
#define TensorBASE_DECL(T) auto c_##T = py::class_<T>(m, #T);
  FOR_ALL_TENSORBASE(TensorBASE_DECL);

  // All of them have BatchView and BaseView
#define BATCHVIEW_DEF(T) def_BatchView<T>(m, #T "BatchView");
  FOR_ALL_TENSORBASE(BATCHVIEW_DEF);
#define BASEVIEW_DEF(T) def_BaseView<T>(m, #T "BaseView");
  FOR_ALL_TENSORBASE(BASEVIEW_DEF);

  // Common methods decorated by TensorBase
#define TensorBASE_DEF(T) def_TensorBase<T>(c_##T);
  FOR_ALL_TENSORBASE(TensorBASE_DEF);

  // Common methods decorated by LogicalTensor
#define LogicalTensor_DEF(T) def_LogicalTensor<T>(c_##T);
  FOR_ALL_LOGICALTENSOR(LogicalTensor_DEF);

  // Common methods decorated by VecBase
#define VECBASE_DEF(T) def_VecBase<T>(c_##T);
  FOR_ALL_VECBASE(VECBASE_DEF);

  // Common methods decorated by R2Base
#define R2BASE_DEF(T) def_R2Base<T>(c_##T);
  FOR_ALL_R2BASE(R2BASE_DEF);

  // Tensor specific methods
#define TENSOR_CUSTOM_DEF(T) def_##T(c_##T);
  FOR_ALL_TENSORBASE(TENSOR_CUSTOM_DEF);

  // Labeled tensors
  def_LabeledAxisAccessor(m);
  def_LabeledAxis(m);
  def_LabeledVector(m);
  def_LabeledMatrix(m);
}
