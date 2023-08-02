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

#include "neml2/misc/types.h"

namespace neml2
{
namespace math
{
constexpr Real sqrt2 = 1.4142135623730951;

constexpr TorchSize mandel_reverse_index[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};
constexpr TorchSize mandel_index[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

inline constexpr Real
mandel_factor(TorchSize i)
{
  return i < 3 ? 1.0 : sqrt2;
}

torch::Tensor linspace(const at::Tensor & start,
                       const at::Tensor & end,
                       TorchSize nstep,
                       const torch::TensorOptions & options = default_tensor_options);

torch::Tensor logspace(const at::Tensor & start,
                       const at::Tensor & end,
                       TorchSize nstep,
                       Real base = 10,
                       const torch::TensorOptions & options = default_tensor_options);

/**
 * This is (almost) equivalent to Torch's heaviside, except that the Torch's version is not
 * differentiable (back-propagatable). I said "almost" because torch::heaviside allows you to set
 * the return value in the case of input == 0. Our implementation always return 0.5 when the input
 * == 0.
 */
torch::Tensor heaviside(const torch::Tensor & self);
} // namespace math
} // namespace neml2
