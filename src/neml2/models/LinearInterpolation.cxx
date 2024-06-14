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
#define LINEARINTERPOLATION_REGISTER(T)                                                            \
  register_NEML2_object_alias(T##LinearInterpolation, #T "LinearInterpolation")
FOR_ALL_FIXEDDIMTENSOR(LINEARINTERPOLATION_REGISTER);

template <typename T>
OptionSet
LinearInterpolation<T>::expected_options()
{
  OptionSet options = Interpolation<T>::expected_options();
  options.doc() += " This object performs a _linear interpolation_.";
  return options;
}

template <typename T>
LinearInterpolation<T>::LinearInterpolation(const OptionSet & options)
  : Interpolation<T>(options),
    _interp_batch_sizes(
        utils::broadcast_sizes(this->_X.batch_sizes().slice(0, this->_X.batch_dim() - 1),
                               this->_Y.batch_sizes().slice(0, this->_Y.batch_dim() - 1))),
    _X0(this->template declare_buffer<Scalar>("X0",
                                              this->_X.batch_index({Ellipsis, Slice(None, -1)}))),
    _X1(this->template declare_buffer<Scalar>("X1",
                                              this->_X.batch_index({Ellipsis, Slice(1, None)}))),
    _Y0(this->template declare_buffer<T>("Y0", this->_Y.batch_index({Ellipsis, Slice(None, -1)}))),
    _slope(this->template declare_buffer<T>("S",
                                            math::diff(this->_Y, 1, this->_Y.batch_dim() - 1) /
                                                math::diff(this->_X, 1, this->_X.batch_dim() - 1)))
{
}

template <typename T>
void
LinearInterpolation<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  const auto x = Scalar(this->_x);
  const auto loc = torch::logical_and(torch::gt(x.batch_unsqueeze(-1), _X0),
                                      torch::le(x.batch_unsqueeze(-1), _X1));
  const auto si = mask<T>(_slope, loc);

  if (out)
  {
    const auto X0i = mask<Scalar>(_X0, loc);
    const auto Y0i = mask<T>(_Y0, loc);
    this->_p = Y0i + si * (x - X0i);
  }

  if (dout_din)
    this->_p.d(this->_x) = si;

  if (d2out_din2)
  {
    // zero
  }
}

#define LINEARINTERPOLATION_INSTANTIATE_FIXEDDIMTENSOR(T) template class LinearInterpolation<T>
FOR_ALL_FIXEDDIMTENSOR(LINEARINTERPOLATION_INSTANTIATE_FIXEDDIMTENSOR);
} // namespace neml2
