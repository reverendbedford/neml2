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

#include "neml2/tensors/Quaternion.h"

namespace neml2
{

Quaternion
Quaternion::init(const Scalar & q0, const Scalar & q1, const Scalar & q2, const Scalar & q3)
{
  return torch::cat({q0, q1, q2, q3}, -1);
}

Quaternion
Quaternion::identity()
{
  return torch::tensor({{1, 0, 0, 0}}, TorchDefaults);
}

Scalar
Quaternion::operator()(TorchSize i) const
{
  return base_index({i}).unsqueeze(-1);
}

Quaternion
Quaternion::operator-() const
{
  return -torch::Tensor(*this);
}

Quaternion
Quaternion::conj() const
{
  return Quaternion::init((*this)(0), -(*this)(1), -(*this)(2), -(*this)(3));
}

Scalar
Quaternion::inner(const Quaternion & other) const
{
  return einsum({*this, other}, {"i", "i"}).unsqueeze(-1);
}

Scalar
Quaternion::norm_sq() const
{
  return inner(*this);
}

Scalar
Quaternion::norm() const
{
  return torch::sqrt(this->norm_sq());
}

Scalar
Quaternion::normv_sq() const
{
  return einsum({base_index({torch::indexing::Slice({1, 4})}),
                 base_index({torch::indexing::Slice({1, 4})})},
                {"i", "i"})
      .unsqueeze(-1);
}

Scalar
Quaternion::normv() const
{
  return torch::sqrt(this->normv_sq());
}

Quaternion
Quaternion::inverse() const
{
  return this->conj() / this->norm_sq();
}

Quaternion
operator*(const Scalar & a, const Quaternion & b)
{
  return torch::operator*(a, b);
}

Quaternion
operator*(const Quaternion & a, const Scalar & b)
{
  return torch::operator*(a, b);
}

Quaternion
operator/(const Quaternion & a, const Scalar & b)
{
  return torch::operator/(a, b);
}

Quaternion
operator*(const Quaternion & a, const Quaternion & b)
{
  return Quaternion::init(a(0) * b(0) - (a(1) * b(1) + a(2) * b(2) + a(3) * b(3)),
                          a(0) * b(1) + a(1) * b(0) + a(2) * b(3) - a(3) * b(2),
                          a(0) * b(2) + a(2) * b(0) + a(3) * b(1) - a(1) * b(3),
                          a(0) * b(3) + a(3) * b(0) + a(1) * b(2) - a(2) * b(1));
}

Quaternion
exp(const Quaternion & a)
{
  Scalar normv = a.normv();
  Scalar sf = sin(normv) / normv;

  return Quaternion::init(cos(normv), a(1) * sf, a(2) * sf, a(3) * sf);
}

Quaternion
log(const Quaternion & a)
{
  Scalar norm = a.norm();
  Scalar normv = a.normv();

  Scalar sf = acos(a(0) / norm);

  return Quaternion::init(log(norm), a(1) / normv * sf, a(2) / normv * sf, a(3) / normv * sf);
}

} // namespace neml2
