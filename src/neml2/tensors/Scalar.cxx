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

#include "neml2/tensors/Scalar.h"

namespace neml2
{
Scalar::Scalar(Real init, const torch::TensorOptions & options)
  : FixedDimTensor<1, 1>(torch::tensor({{init}}, options))
{
}

Scalar
Scalar::zero(const torch::TensorOptions & options)
{
  return Scalar(0, options);
}

Scalar
Scalar::operator-() const
{
  return -torch::Tensor(*this);
}

Scalar
Scalar::pow(Scalar n) const
{
  return torch::pow(*this, n);
}

Scalar
Scalar::identity_map(const torch::TensorOptions & options)
{
  return Scalar(1, options);
}

Scalar
operator+(const Scalar & a, const Real & b)
{
  return torch::operator+(a, b);
}

Scalar
operator+(const Real & a, const Scalar & b)
{
  return torch::operator+(a, b);
}

Scalar
operator+(const Scalar & a, const Scalar & b)
{
  return torch::operator+(a, b);
}

BatchTensor<1>
operator+(const BatchTensor<1> & a, const Scalar & b)
{
  TorchSlice net(a.base_dim() - b.base_dim(), torch::indexing::None);
  net.insert(net.begin(), torch::indexing::Ellipsis);
  return torch::operator+(a, b.index(net));
}

BatchTensor<1>
operator+(const Scalar & a, const BatchTensor<1> & b)
{
  return b + a;
}

Scalar
operator-(const Scalar & a, const Real & b)
{
  return torch::operator-(a, b);
}

Scalar
operator-(const Real & a, const Scalar & b)
{
  return torch::operator-(a, b);
}

Scalar
operator-(const Scalar & a, const Scalar & b)
{
  return torch::operator-(a, b);
}

BatchTensor<1>
operator-(const BatchTensor<1> & a, const Scalar & b)
{
  TorchSlice net(a.base_dim() - b.base_dim(), torch::indexing::None);
  net.insert(net.begin(), torch::indexing::Ellipsis);
  return torch::operator-(a, b.index(net));
}

BatchTensor<1>
operator-(const Scalar & a, const BatchTensor<1> & b)
{
  return -b + a;
}

Scalar
operator*(const Scalar & a, const Real & b)
{
  return torch::operator*(a, b);
}

Scalar
operator*(const Real & a, const Scalar & b)
{
  return torch::operator*(a, b);
}

Scalar
operator*(const Scalar & a, const Scalar & b)
{
  return torch::operator*(a, b);
}

BatchTensor<1>
operator*(const BatchTensor<1> & a, const Scalar & b)
{
  TorchSlice net(a.base_dim() - b.base_dim(), torch::indexing::None);
  net.insert(net.begin(), torch::indexing::Ellipsis);
  return torch::operator*(a, b.index(net));
}

BatchTensor<1>
operator*(const Scalar & a, const BatchTensor<1> & b)
{
  return b * a;
}

Scalar
operator/(const Scalar & a, const Real & b)
{
  return torch::operator/(a, b);
}

Scalar
operator/(const Real & a, const Scalar & b)
{
  return torch::operator/(a, b);
}

Scalar
operator/(const Scalar & a, const Scalar & b)
{
  return torch::operator/(a, b);
}

BatchTensor<1>
operator/(const BatchTensor<1> & a, const Scalar & b)
{
  TorchSlice net(a.base_dim() - b.base_dim(), torch::indexing::None);
  net.insert(net.begin(), torch::indexing::Ellipsis);
  return torch::operator/(a, b.index(net));
}

BatchTensor<1>
operator/(const Scalar & a, const BatchTensor<1> & b)
{
  TorchSlice net(b.base_dim() - a.base_dim(), torch::indexing::None);
  net.insert(net.begin(), torch::indexing::Ellipsis);
  return torch::operator/(a.index(net), b);
}

Scalar
macaulay(const Scalar & a)
{
  return a * Scalar(math::heaviside(a));
}

Scalar
dmacaulay(const Scalar & a)
{
  return math::heaviside(a);
}

Scalar
exp(const Scalar & a)
{
  return Scalar(torch::exp(a));
}
} // namespace neml2
