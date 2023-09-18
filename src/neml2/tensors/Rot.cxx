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

#include "neml2/tensors/Rot.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SymR2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/SymSymR4.h"

#include "neml2/tensors/RotRot.h"
#include "neml2/tensors/VecRot.h"
#include "neml2/tensors/R2Rot.h"
#include "neml2/tensors/SymR2Rot.h"
#include "neml2/tensors/R4Rot.h"
#include "neml2/tensors/SymSymR4Rot.h"

namespace neml2
{

Rot
Rot::init(const Scalar & r0, const Scalar & r1, const Scalar & r2)
{
  return torch::cat({r0, r1, r2}, -1);
}

Rot
Rot::init(const Real & r0, const Real & r1, const Real & r2, const torch::TensorOptions & options)
{
  return torch::tensor({{r0, r1, r2}}, options);
}

Rot
Rot::identity(const torch::TensorOptions & options)
{
  return torch::zeros({1, 3}, options);
}

RotRot
Rot::identity_map(const torch::TensorOptions & options)
{
  return RotRot::identity(options);
}

Rot
Rot::inverse() const
{
  return -torch::Tensor(*this);
}

Scalar
Rot::n2() const
{
  return this->dot(*this);
}

R2
Rot::to_R2() const
{
  // We use the dot product twice
  auto rr = this->n2();

  return ((1 - rr) * R2::identity() + 2.0 * this->outer(*this) -
          2.0 * R3::levi_civita().contract_k(*this)) /
         (1.0 + rr);
}

R3
Rot::dR2() const
{
  return torch::Tensor(2.0 / (1 + this->n2())).unsqueeze(-1).unsqueeze(-1) *
         (-einsum({this->to_R2(), *this}, {"ij", "m"}) -
          einsum({R2::identity(), *this}, {"ij", "m"}) +
          einsum({R2::identity(), *this}, {"im", "j"}) +
          einsum({R2::identity(), *this}, {"jm", "i"}) - torch::Tensor(R3::levi_civita()));
}

Rot
Rot::apply(const Rot & R) const
{
  return *this * R;
}

Vec
Rot::apply(const Vec & v) const
{
  // Use twice...
  auto rr = this->n2();

  return ((1 - rr) * v + 2 * this->dot(v) * Vec(*this) - 2 * v.cross(*this)) / (1 + rr);
}

R2
Rot::apply(const R2 & T) const
{
  R2 R = this->to_R2();
  return R * T * R.transpose();
}

SymR2
Rot::apply(const SymR2 & T) const
{
  return this->apply(T.to_full()).sym();
}

R4
Rot::apply(const R4 & T) const
{
  R2 R = this->to_R2();
  return einsum({R, R, R, R, T}, {"im", "jn", "ko", "lp", "mnop"});
}

SymSymR4
Rot::apply(const SymSymR4 & T) const
{
  return this->apply(T.to_full()).sym();
}

RotRot
Rot::dapply(const Rot & R) const
{
  return RotRot::derivative(*this, R);
}

VecRot
Rot::dapply(const Vec & v) const
{
  return VecRot::derivative(*this, v);
}

R2Rot
Rot::dapply(const R2 & T) const
{
  return R2Rot::derivative(*this, T);
}

SymR2Rot
Rot::dapply(const SymR2 & T) const
{
  return SymR2Rot::derivative(*this, T);
}

R4Rot
Rot::dapply(const R4 & T) const
{
  return R4Rot::derivative(*this, T);
}

SymSymR4Rot
Rot::dapply(const SymSymR4 & T) const
{
  return SymSymR4Rot::derivative(*this, T);
}

Rot
operator*(const Rot & r1, const Rot & r2)
{
  return (r1 + r2 + r1.cross(r2)) / (1.0 - r1.dot(r2));
}

} // namemspace neml2
