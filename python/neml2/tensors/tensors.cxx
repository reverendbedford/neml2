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

#include <nanobind/nanobind.h>

#include "neml2/tensors/macros.h"
#include "python/neml2/tensors/BatchTensorBase.h"
#include "python/neml2/tensors/FixedDimTensor.h"
#include "python/neml2/tensors/VecBase.h"
#include "python/neml2/tensors/R2Base.h"

namespace nb = nanobind;
using namespace neml2;

// Forward declarations
#define TENSOR_CUSTOM_DEF_FWD(T) void def_##T(nb::class_<T> &)
FOR_ALL_BATCHTENSORBASE(TENSOR_CUSTOM_DEF_FWD);
void def_LabeledAxisAccessor(nb::module_ & m);

NB_MODULE(tensors, m)
{
  m.doc() = "NEML2 primitive tensor types";

  // Declare all the BatchTensorBase derived tensors
  // This is as simple as calling nb::class_, but it is important to do this for ALL tensors up
  // front. The reason is for typing: Once we build the neml2 python library with all the necessary
  // bindings, we will have to extract all the typing information (mostly function signature) from
  // the library, which is needed by language servers like Pylance. We use pybind11-stubgen for that
  // purpose. For a type to be deducible by pybind11-stubgen, a concrete definition of the binding
  // class must exist at the point of method definition. Therefore, we need to first create all the
  // class definitions before creating method bindings that use them as arguments.
#define BATCHTENSORBASE_DECL(T) auto c_##T = nb::class_<T>(m, #T);
  FOR_ALL_BATCHTENSORBASE(BATCHTENSORBASE_DECL);

  // All of them have BatchView and BaseView
#define BATCHVIEW_DEF(T) def_BatchView<T>(m, #T "BatchView");
  FOR_ALL_BATCHTENSORBASE(BATCHVIEW_DEF);
#define BASEVIEW_DEF(T) def_BaseView<T>(m, #T "BaseView");
  FOR_ALL_BATCHTENSORBASE(BASEVIEW_DEF);

  // Common methods decorated by BatchTensorBase
#define BATCHTENSORBASE_DEF(T) def_BatchTensorBase<T>(c_##T);
  FOR_ALL_BATCHTENSORBASE(BATCHTENSORBASE_DEF);

  // Common methods decorated by FixedDimTensor
#define FIXEDDIMTENSOR_DEF(T) def_FixedDimTensor<T>(c_##T);
  FOR_ALL_FIXEDDIMTENSOR(FIXEDDIMTENSOR_DEF);

  // Common methods decorated by VecBase
#define VECBASE_DEF(T) def_VecBase<T>(c_##T);
  FOR_ALL_VECBASE(VECBASE_DEF);

  // Common methods decorated by R2Base
#define R2BASE_DEF(T) def_R2Base<T>(c_##T);
  FOR_ALL_R2BASE(R2BASE_DEF);

  // Tensor specific methods
#define TENSOR_CUSTOM_DEF(T) def_##T(c_##T);
  FOR_ALL_BATCHTENSORBASE(TENSOR_CUSTOM_DEF);

  // Labeled tensors
  def_LabeledAxisAccessor(m);
}
