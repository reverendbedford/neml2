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

#include "neml2/misc/types.h"

namespace neml2
{
torch::TensorOptions &
default_tensor_options()
{
  static torch::TensorOptions _default_tensor_options =
      torch::TensorOptions().dtype(default_dtype()).device(default_device());
  return _default_tensor_options;
}

torch::TensorOptions &
default_int_tensor_options()
{
  static torch::TensorOptions _default_int_tensor_options =
      torch::TensorOptions().dtype(default_int_dtype()).device(default_device());
  return _default_int_tensor_options;
}

torch::Dtype &
default_dtype()
{
  static torch::Dtype _default_dtype = torch::kFloat64;
  return _default_dtype;
}

torch::Dtype &
default_int_dtype()
{
  static torch::Dtype _default_int_dtype = torch::kInt64;
  return _default_int_dtype;
}

torch::Device &
default_device()
{
  static torch::Device _default_device = torch::kCPU;
  return _default_device;
}
} // namespace neml2
