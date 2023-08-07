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

void
Scalar::operator+=(const Scalar & other)
{
  torch::Tensor::operator+=(other.to(*this));
}

void
Scalar::operator-=(const Scalar & other)
{
  torch::Tensor::operator-=(other.to(*this));
}

void
Scalar::operator*=(const Scalar & other)
{
  torch::Tensor::operator*=(other.to(*this));
}

void
Scalar::operator/=(const Scalar & other)
{
  torch::Tensor::operator/=(other.to(*this));
}

Scalar
Scalar::pow(Scalar n) const
{
  return torch::pow(*this, n.to(*this));
}

Scalar
Scalar::identity_map(const torch::TensorOptions & options)
{
  return Scalar(1, options);
}

Scalar
operator+(const Scalar & a, const Scalar & b)
{
  if (a.device() == torch::kCPU && b.device() != torch::kCPU)
    return torch::operator+(a.to(b), b);
  else if (a.device() != torch::kCPU && b.device() == torch::kCPU)
    return torch::operator+(a, b.to(a));
  else
    return torch::operator+(a, b);
}

BatchTensor<1>
operator+(const BatchTensor<1> & a, const Scalar & b)
{
  torch::Tensor tmp = b.to(a);
  for (TorchSize i = 1; i < a.base_dim(); i++)
    tmp = tmp.unsqueeze(-1);
  return torch::operator+(a, tmp);
}

BatchTensor<1>
operator+(const Scalar & a, const BatchTensor<1> & b)
{
  return b + a;
}

Scalar
operator-(const Scalar & a, const Scalar & b)
{
  if (a.device() == torch::kCPU && b.device() != torch::kCPU)
    return torch::operator-(a.to(b), b);
  else if (a.device() != torch::kCPU && b.device() == torch::kCPU)
    return torch::operator-(a, b.to(a));
  else
    return torch::operator-(a, b);
}

BatchTensor<1>
operator-(const BatchTensor<1> & a, const Scalar & b)
{
  torch::Tensor tmp = b.to(a);
  for (TorchSize i = 1; i < a.base_dim(); i++)
    tmp = tmp.unsqueeze(-1);
  return torch::operator-(a, tmp);
}

BatchTensor<1>
operator-(const Scalar & a, const BatchTensor<1> & b)
{
  return -b + a;
}

Scalar
operator*(const Scalar & a, const Scalar & b)
{
  if (a.device() == torch::kCPU && b.device() != torch::kCPU)
    return torch::operator*(a.to(b), b);
  else if (a.device() != torch::kCPU && b.device() == torch::kCPU)
    return torch::operator*(a, b.to(a));
  else
    return torch::operator*(a, b);
}

BatchTensor<1>
operator*(const BatchTensor<1> & a, const Scalar & b)
{
  torch::Tensor tmp = b.to(a);
  for (TorchSize i = 1; i < a.base_dim(); i++)
    tmp = tmp.unsqueeze(-1);
  return torch::operator*(a, tmp);
}

BatchTensor<1>
operator*(const Scalar & a, const BatchTensor<1> & b)
{
  return b * a;
}

Scalar
operator/(const Scalar & a, const Scalar & b)
{
  if (a.device() == torch::kCPU && b.device() != torch::kCPU)
    return torch::operator/(a.to(b), b);
  else if (a.device() != torch::kCPU && b.device() == torch::kCPU)
    return torch::operator/(a, b.to(a));
  else
    return torch::operator/(a, b);
}

BatchTensor<1>
operator/(const BatchTensor<1> & a, const Scalar & b)
{
  torch::Tensor tmp = b.to(a);
  for (TorchSize i = 1; i < a.base_dim(); i++)
    tmp = tmp.unsqueeze(-1);
  return torch::operator/(a, tmp);
}

BatchTensor<1>
operator/(const Scalar & a, const BatchTensor<1> & b)
{
  torch::Tensor tmp = a.to(b);
  for (TorchSize i = 1; i < a.base_dim(); i++)
    tmp = tmp.unsqueeze(-1);
  return torch::operator/(tmp, b);
}

Scalar
macaulay(const Scalar & a, const Scalar & a0)
{
  return a * Scalar(torch::heaviside(a, a0));
}

Scalar
dmacaulay(const Scalar & a, const Scalar & a0)
{
  return torch::heaviside(a, a0);
}

Scalar
exp(const Scalar & a)
{
  return Scalar(torch::exp(a));
}

} // namespace neml2
