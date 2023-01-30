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
SymR2::identity_map()
{
  return SymSymR4::init(SymSymR4::FillMethod::identity_sym);
}

SymR2
SymR2::zeros()
{
  return torch::zeros({1, 6}, TorchDefaults);
}

SymR2
SymR2::zeros(TorchSize batch_size)
{
  return torch::zeros({batch_size, 6}, TorchDefaults);
}

SymR2
SymR2::init(const Scalar & a)
{
  torch::Tensor zero = torch::zeros_like(a, TorchDefaults);
  return torch::cat({a, a, a, zero, zero, zero}, {-1});
}

SymR2
SymR2::init(const Scalar & a11, const Scalar & a22, const Scalar & a33)
{
  torch::Tensor zero = torch::zeros_like(a11, TorchDefaults);
  return torch::cat({a11, a22, a33, zero, zero, zero}, {-1});
}

SymR2
SymR2::init(const Scalar & a11,
            const Scalar & a22,
            const Scalar & a33,
            const Scalar & a23,
            const Scalar & a13,
            const Scalar & a12)
{
  return torch::cat({a11, a22, a33, utils::sqrt2 * a23, utils::sqrt2 * a13, utils::sqrt2 * a12},
                    {-1});
}

SymR2
SymR2::identity()
{
  return SymR2(torch::tensor({{1, 1, 1, 0, 0, 0}}, TorchDefaults));
}

Scalar
SymR2::operator()(TorchSize i, TorchSize j) const
{
  TorchSize a = reverse_index[i][j];
  return base_index({a}).unsqueeze(-1) / utils::mandelFactor(a);
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
SymR2::norm() const
{
  return torch::sqrt(norm_sq());
}

SymSymR4
SymR2::outer(const SymR2 & other) const
{
  return einsum({*this, other}, {"i", "j"});
}

SymR2
operator+(const SymR2 & a, const Scalar & b)
{
  return torch::operator+(a, b);
}

SymR2
operator+(const Scalar & a, const SymR2 & b)
{
  return torch::operator+(a, b);
}

SymR2
operator+(const SymR2 & a, const SymR2 & b)
{
  return torch::operator+(a, b);
}

SymR2
operator-(const SymR2 & a, const Scalar & b)
{
  return torch::operator-(a, b);
}

SymR2
operator-(const Scalar & a, const SymR2 & b)
{
  return torch::operator-(a, b);
}

SymR2
operator-(const SymR2 & a, const SymR2 & b)
{
  return torch::operator-(a, b);
}

SymR2
operator*(const SymR2 & a, const Scalar & b)
{
  return torch::operator*(a, b);
}

SymR2
operator*(const Scalar & a, const SymR2 & b)
{
  return torch::operator*(a, b);
}

SymR2
operator*(const SymR2 & a, const SymR2 & b)
{
  return torch::operator*(a, b);
}

SymR2
operator/(const SymR2 & a, const Scalar & b)
{
  return torch::operator/(a, b);
}

SymR2
operator/(const Scalar & a, const SymR2 & b)
{
  return torch::operator/(a, b);
}

SymR2
operator/(const SymR2 & a, const SymR2 & b)
{
  return torch::operator/(a, b);
}
} // namespace neml2
