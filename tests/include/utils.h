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

#include "neml2/base/Parser.h"
#include "neml2/base/Factory.h"
#include "neml2/tensors/Scalar.h"

/**
 * @brief A simple finite-differencing helper to numerically approximate the derivative of the
 * function at the given point.
 *
 * @tparam F The functor to differentiate
 * @tparam T Type of the input variable, must be _batched_
 * @param f The functor to differentiate, must accept the input of type `BatchTensor`
 * @param x The point where the derivative is evaluated
 * @param eps The relative perturbation (for each component in the case of non-Scalar input)
 * @param aeps The minimum perturbation to improve numerical stability
 * @return BatchTensor The derivative at the given point approximated using finite differencing
 */
template <typename F>
[[nodiscard]] neml2::BatchTensor
finite_differencing_derivative(F && f,
                               const neml2::BatchTensor & x,
                               neml2::Real eps = 1e-6,
                               neml2::Real aeps = 1e-6)
{
  using namespace neml2;
  using namespace torch::indexing;

  // The scalar case is trivial
  if (x.base_dim() == 0)
  {
    auto y0 = BatchTensor(f(x)).clone();

    auto dx = eps * Scalar(neml2::math::abs(x));
    dx.index_put_({dx < aeps}, aeps);

    auto x1 = x + dx;

    auto y1 = BatchTensor(f(x1)).clone();
    auto dy_dx = (y1 - y0) / dx;

    return dy_dx;
  }

  // Flatten x to support arbitrarily shaped input
  auto xf = BatchTensor(
      x.reshape(utils::add_shapes(x.batch_sizes(), utils::storage_size(x.base_sizes()))),
      x.batch_dim());

  auto y0 = BatchTensor(f(x)).clone();

  auto dy_dxf = BatchTensor::empty(
      x.batch_sizes(), utils::add_shapes(y0.base_sizes(), xf.base_sizes()), x.options());

  for (TorchSize i = 0; i < xf.base_sizes()[0]; i++)
  {
    auto dx = eps * Scalar(math::abs(xf.base_index({i})));
    dx.index_put_({dx < aeps}, aeps);

    auto xf1 = xf.clone();
    xf1.base_index_put({i}, xf1.base_index({i}) + dx);
    auto x1 = BatchTensor(xf1.reshape(x.sizes()), x.batch_dim());

    auto y1 = BatchTensor(f(x1)).clone();
    dy_dxf.base_index_put({Ellipsis, i}, (y1 - y0) / dx);
  }

  // Reshape the derivative back to the correct shape
  auto dy_dx = BatchTensor(
      dy_dxf.reshape(utils::add_shapes(x.batch_sizes(), y0.base_sizes(), x.base_sizes())),
      x.batch_dim());

  return dy_dx;
}
