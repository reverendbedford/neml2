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

#include "neml2/models/LinearInterpolation.h"

using namespace torch::indexing;

namespace neml2
{
register_all_FixedDimTensor_suffix(LinearInterpolation, "LinearInterpolation");

template <typename T>
OptionSet
LinearInterpolation<T>::expected_options()
{
  OptionSet options = Interpolation<T>::expected_options();
  return options;
}

template <typename T>
LinearInterpolation<T>::LinearInterpolation(const OptionSet & options)
  : Interpolation<T>(options),
    _a0(this->template declare_buffer<Scalar>("A0", this->_abscissa.index({Slice(None, -1)}))),
    _a1(this->template declare_buffer<Scalar>("A1", this->_abscissa.index({Slice(1, None)}))),
    _o0(this->template declare_buffer<T>("O0", this->_ordinate.index({Slice(None, -1)}))),
    _slope(this->template declare_buffer<T>(
        "S", math::diff(this->_ordinate, 1, 0) / math::diff(this->_abscissa, 1, 0)))
{
  this->setup();
}

template <typename T>
void
LinearInterpolation<T>::interpolate(const Scalar & x, T * y, T * dy_dx, T * d2y_dx2) const
{
  const auto loc = torch::logical_and(torch::gt(x, _a0), torch::le(x, _a1));
  const auto si = mask<T>(_slope, loc);

  if (y)
  {
    const auto a0i = mask<Scalar>(_a0, loc);
    const auto o0i = mask<T>(_o0, loc);
    (*y) = o0i + si * (x - a0i);
  }

  if (dy_dx)
    (*dy_dx) = si;

  if (d2y_dx2)
    (*d2y_dx2) = T::zeros_like(si);
}

instantiate_all_FixedDimTensor(LinearInterpolation);
} // namespace neml2
