// Copyright 2024, UChicago Argonne, LLC
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

#include "neml2/models/SmoothLinearInterpolation.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ScalarSmoothLinearInterpolation);
register_NEML2_object(VecSmoothLinearInterpolation);
register_NEML2_object(SR2SmoothLinearInterpolation);

template <typename T>
OptionSet
SmoothLinearInterpolation<T>::expected_options()
{
  OptionSet options = Interpolation<T>::expected_options();
  options.doc() += " This object performs a _linear interpolation_.";
  options.set<Real>("sharpness") = 50;
  options.set("sharpness").doc() = "Sharpness of the smooth interpolation. The result aympototes "
                                   "to the discrete interpolation as sharpness gets larger.";
  return options;
}

template <typename T>
SmoothLinearInterpolation<T>::SmoothLinearInterpolation(const OptionSet & options)
  : Interpolation<T>(options),
    _k(options.get<Real>("sharpness")),
    _X0(this->template declare_buffer<Scalar>(
        "X0", this->_X.batch_index({indexing::Ellipsis, indexing::Slice(indexing::None, -1)}))),
    _X1(this->template declare_buffer<Scalar>(
        "X1", this->_X.batch_index({indexing::Ellipsis, indexing::Slice(1, indexing::None)}))),
    _Y0(this->template declare_buffer<T>(
        "Y0", this->_Y.batch_index({indexing::Ellipsis, indexing::Slice(indexing::None, -1)}))),
    _Y1(this->template declare_buffer<T>(
        "Y1", this->_Y.batch_index({indexing::Ellipsis, indexing::Slice(1, indexing::None)})))
{
}

template <typename T>
Scalar
SmoothLinearInterpolation<T>::smooth_interp_factor(const Scalar & x) const
{
  const auto s = Scalar(torch::sigmoid(_k * x));
  const auto s1 = Scalar(torch::sigmoid(_k * (x - 1)));
  return (1 - x) * (s - s1);
}

template <typename T>
Scalar
SmoothLinearInterpolation<T>::d_smooth_interp_factor(const Scalar & x) const
{
  const auto s = Scalar(torch::sigmoid(_k * x));
  const auto s1 = Scalar(torch::sigmoid(_k * (x - 1)));
  return -(s - s1) + _k * (1 - x) * (s * (1 - s) - s1 * (1 - s1));
}

template <typename T>
void
SmoothLinearInterpolation<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "This model does not define the second derivatives.");

  const auto x = Scalar(this->_x).batch_unsqueeze(-1);
  const auto dX = _X1 - _X0;
  const auto a = (x - _X0) / dX;
  const auto b = (_X1 - x) / dX;

  if (out)
  {
    const auto ia = smooth_interp_factor(a);
    const auto ib = smooth_interp_factor(b);
    this->_p = math::batch_sum(ia * _Y0 + ib * _Y1, -1);
  }

  if (dout_din)
    if (this->_x.is_dependent())
    {
      const auto dia = d_smooth_interp_factor(a);
      const auto dib = d_smooth_interp_factor(b);
      this->_p.d(this->_x) = math::batch_sum(dia * _Y0 + dib * _Y1, -1);
    }
}

template class SmoothLinearInterpolation<Scalar>;
template class SmoothLinearInterpolation<Vec>;
template class SmoothLinearInterpolation<SR2>;
} // namespace neml2
