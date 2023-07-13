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

RotRot
Rotation::identity_map()
{
  return RotRot::identity();
}

Rotation
Rotation::inverse() const
{
  return -torch::Tensor(*this);
}

R2
Rotation::to_R2() const
{
  // We use the dot product twice
  auto rr = this->dot(*this);

  return ((1 - rr) * R2::identity() + 2.0 * this->outer(*this) -
          2.0 * R3::levi_civita().contract_k(*this)) /
         (1.0 + rr);
}

Rotation
Rotation::apply(const Rotation & R) const
{
  return *this * R;
}

Vector
Rotation::apply(const Vector & v) const
{
  return (this->to_R2()) * v;
}

R2
Rotation::apply(const R2 & T) const
{
  R2 R = this->to_R2();
  return R * T * R.transpose();
}

SymR2
Rotation::apply(const SymR2 & T) const
{
  return this->apply(T.to_full()).to_symmetric();
}

R4
Rotation::apply(const R4 & T) const
{
  R2 R = this->to_R2();
  return einsum({R, R, R, R, T}, {"im", "jn", "ko", "lp", "mnop"});
}

SymSymR4
Rotation::apply(const SymSymR4 & T) const
{
  return this->apply(T.to_full()).to_symmetric();
}

RotRot
Rotation::dapply(const Rotation & R) const
{
  return RotRot::dRdR(*this, R);
}

Rotation
operator*(const Rotation & r1, const Rotation & r2)
{
  return (r1 + r2 + r1.cross(r2)) / (1.0 - r1.dot(r2));
}

} // namemspace neml2
