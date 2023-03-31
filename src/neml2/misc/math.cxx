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

#include "neml2/misc/math.h"
#include "neml2/misc/error.h"

namespace neml2
{
namespace math
{
torch::Tensor
linspace(torch::Tensor start,
         torch::Tensor end,
         TorchSize nstep,
         const torch::TensorOptions & options)
{
  neml_assert_dbg(start.dim() == 0 || start.sizes() == end.sizes(),
                  "If the start tensor is not of a scalar type, the start and stop tensors need "
                  "the same shape. The start tensor has shape ",
                  start.sizes(),
                  " while the end tensor has shape ",
                  end.sizes());

  auto res = start.unsqueeze(0);
  if (nstep > 1)
  {
    auto steps = torch::arange(nstep, options) / (nstep - 1);
    res = res + torch::einsum("i,...->i...", {steps, end - start});
  }
  return res;
}

torch::Tensor
logspace(torch::Tensor start,
         torch::Tensor end,
         TorchSize nstep,
         Real base,
         const torch::TensorOptions & options)
{
  auto exponent = linspace(start, end, nstep, options);
  return torch::pow(base, exponent);
}
} // namespace math
} // namespace neml2
