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

#include "neml2/tensors/SymR2.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
SymSymR4
SymR2::identity_map(const torch::TensorOptions & options)
{
  return SymSymR4::init_identity_sym(options);
}

SymR2
SymR2::zero(const torch::TensorOptions & options)
{
  return torch::zeros({1, 6}, options);
}

SymR2
SymR2::init(const Scalar & a)
{
  const auto zero = torch::zeros_like(a);
  return torch::cat({a, a, a, zero, zero, zero}, -1);
}

SymR2
SymR2::init(const Scalar & a11, const Scalar & a22, const Scalar & a33)
{
  const auto zero = torch::zeros_like(a11);
  return torch::cat({a11, a22, a33, zero, zero, zero}, -1);
}

SymR2
SymR2::init(const Scalar & a11,
            const Scalar & a22,
            const Scalar & a33,
            const Scalar & a23,
            const Scalar & a13,
            const Scalar & a12)
{
  return torch::cat({a11, a22, a33, math::sqrt2 * a23, math::sqrt2 * a13, math::sqrt2 * a12}, -1);
}

SymR2
SymR2::identity(const torch::TensorOptions & options)
{
  return torch::tensor({{1, 1, 1, 0, 0, 0}}, options);
}

Scalar
SymR2::operator()(TorchSize i, TorchSize j) const
{
  TorchSize a = math::mandel_reverse_index[i][j];
  return base_index({a}).unsqueeze(-1) / math::mandel_factor(a);
}

SymR2
SymR2::operator-() const
{
  return -torch::Tensor(*this);
}

Scalar
SymR2::tr() const
{
  return torch::sum(base_index({torch::indexing::Slice(0, 3)}), {-1}, true);
}

SymR2
SymR2::vol() const
{
  return init(tr()) / 3;
}

SymR2
SymR2::dev() const
{
  return *this - vol();
}

Scalar
SymR2::det() const
{
  return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) -
         (*this)(0, 1) * ((*this)(2, 2) * (*this)(0, 1) - (*this)(1, 2) * (*this)(0, 2)) +
         (*this)(0, 2) * ((*this)(0, 1) * (*this)(1, 2) - (*this)(1, 1) * (*this)(0, 2));
}

Scalar
SymR2::inner(const SymR2 & other) const
{
  return einsum({*this, other}, {"i", "i"}).unsqueeze(-1);
}

Scalar
SymR2::norm_sq() const
{
  return inner(*this);
}

Scalar
SymR2::norm(Real eps) const
{
  return torch::sqrt(norm_sq() + eps);
}

SymSymR4
SymR2::outer(const SymR2 & other) const
{
  return einsum({*this, other}, {"i", "j"});
}

SymR2
SymR2::inverse() const
{
  return torch::linalg::inv(*this);
}

SymR2
operator+(const SymR2 & a, const Scalar & b)
{
  return torch::operator+(a, b.to(a));
}

SymR2
operator+(const Scalar & a, const SymR2 & b)
{
  return b + a;
}

SymR2
operator+(const SymR2 & a, const SymR2 & b)
{
  return torch::operator+(a, b);
}

SymR2
operator-(const SymR2 & a, const Scalar & b)
{
  return torch::operator-(a, b.to(a));
}

SymR2
operator-(const Scalar & a, const SymR2 & b)
{
  return torch::operator-(a.to(b), b);
}

SymR2
operator-(const SymR2 & a, const SymR2 & b)
{
  return torch::operator-(a, b);
}

SymR2
operator*(const SymR2 & a, const Scalar & b)
{
  return torch::operator*(a, b.to(a));
}

SymR2
operator*(const Scalar & a, const SymR2 & b)
{
  return b * a;
}

SymR2
operator/(const SymR2 & a, const Scalar & b)
{
  return torch::operator/(a, b.to(a));
}

SymR2
operator/(const Scalar & a, const SymR2 & b)
{
  return torch::operator/(a.to(b), b);
}

SymR2
operator/(const SymR2 & a, const SymR2 & b)
{
  return torch::operator/(a, b);
}
} // namespace neml2
