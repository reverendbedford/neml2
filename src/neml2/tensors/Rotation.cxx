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

#include "neml2/tensors/Rotation.h"
#include "neml2/tensors/R3.h"

#include "neml2/misc/utils.h"

namespace neml2
{

Rotation
Rotation::init(const Scalar & r0, const Scalar & r1, const Scalar & r2)
{
  return torch::cat({r0, r1, r2}, -1);
}

Rotation
Rotation::identity()
{
  return torch::zeros({1, 3}, TorchDefaults);
}

Scalar
Rotation::operator()(TorchSize i) const
{
  return base_index({i}).unsqueeze(-1);
}

Rotation
Rotation::inverse() const
{
  return -torch::Tensor(*this);
}

Scalar
Rotation::dot(const Rotation & other) const
{
  return einsum({*this, other}, {"i", "i"}).unsqueeze(-1);
}

R2
Rotation::to_R2() const
{
  // We use the dot product several times
  auto rr = torch::Tensor(this->dot(*this));

  return 1.0 / (1.0 + rr) *
         ((1 - rr) * torch::eye(3, TorchDefaults) + 2.0 * einsum({*this, *this}, {"i", "j"}) -
          2.0 * einsum({R3::init(R3::levi_civita), *this}, {"ijk", "k"}));
}

Vector
Rotation::apply(const Vector & v) const
{
  return (this->to_R2()) * v;
}

Rotation
operator*(const Rotation & r1, const Rotation & r2)
{
  return (torch::Tensor(r1) + torch::Tensor(r2) + torch::linalg_cross(r1, r2)) /
         torch::Tensor((1.0 - r1.dot(r2)));
}

} // namemspace neml2
