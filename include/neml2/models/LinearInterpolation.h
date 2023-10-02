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

#include "neml2/models/Interpolation.h"

namespace neml2
{
template <typename T>
class LinearInterpolation : public Interpolation<T>
{
public:
  static OptionSet expected_options();

  LinearInterpolation(const OptionSet & options);

protected:
  virtual void interpolate(const Scalar & x, T * y, T * dy_dx, T * d2y_dx2) const override;

private:
  template <typename T2>
  T2 mask(const T2 & in, const torch::Tensor & m) const;

  /// Starting abscissa of each interval
  const Scalar & _a0;
  /// Ending abscissa of each interval
  const Scalar & _a1;
  /// Starting ordinate of each interval
  const T & _o0;
  /// Slope of each interval
  const T & _slope;
};

template <typename T>
template <typename T2>
T2
LinearInterpolation<T>::mask(const T2 & in, const torch::Tensor & m) const
{
  auto in_expand = in.expand(utils::add_shapes(m.sizes(), in.base_sizes()));
  auto in_mask = in_expand.index({m});
  return in_mask.reshape(in_expand.sizes().slice(1));
}

typedef_all_FixedDimTensor_suffix(LinearInterpolation, LinearInterpolation);
} // namespace neml2
