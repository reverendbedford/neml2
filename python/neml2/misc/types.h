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

#include <torch/extension.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/DynamicTypes.h>

namespace nb = nanobind;

#define NEML2_TENSOR_OPTIONS_VARGS                                                                 \
  const torch::Dtype &dtype, const torch::Device &device, bool requires_grad

#define NEML2_TENSOR_OPTIONS                                                                       \
  torch::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad)

#define PY_ARG_TENSOR_OPTIONS                                                                      \
  py::arg("dtype") = torch::Dtype(NEML2_DTYPE), py::arg("device") = torch::Device(torch::kCPU),    \
  py::arg("requires_grad") = false

namespace nanobind
{
namespace detail
{
/**
 * @brief This instantiation enables type conversion between Python object <--> torch::Dtype
 */
template <>
struct type_caster<torch::Dtype>
{
public:
  NB_TYPE_CASTER(torch::Dtype, _("torch.dtype"));

  /**
   * NB_TYPE_CASTER defines a member field called value. Since at::Dtype cannot be
   * default-initialized, we provide this constructor to explicitly initialize that field. The value
   * doesn't matter as it will be overwritten after a successful call to load.
   */
  type_caster()
    : value(torch::kFloat64)
  {
  }

  bool from_python(handle src, uint8_t, cleanup_list *)
  {
    PyObject * obj = src.ptr();
    if (THPDtype_Check(obj))
    {
      value = reinterpret_cast<THPDtype *>(obj)->scalar_type;
      return true;
    }
    return false;
  }

  static handle from_cpp(const torch::Dtype & src, rv_policy, cleanup_list *)
  {
    return handle(reinterpret_cast<PyObject *>(torch::getTHPDtype(src)));
  }
};
}
}
