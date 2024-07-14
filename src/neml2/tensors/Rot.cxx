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
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/WR2.h"
#include "neml2/misc/math.h"

namespace neml2
{
Rot::Rot(const Vec & v)
  : Rot(Tensor(v))
{
}

Rot
Rot::identity(const torch::TensorOptions & options)
{
  return Rot::zeros(options);
}

Rot
Rot::inverse() const
{
  return -(*this);
}

R2
Rot::euler_rodrigues() const
{
  auto rr = norm_sq();
  auto E = R3::levi_civita(options());
  auto W = R2::skew(*this);

  return 1.0 / math::pow(1 + rr, 2.0) *
         (math::pow(1 + rr, 2.0) * R2::identity(options()) + 4 * (1.0 - rr) * W + 8.0 * W * W);
}

R3
Rot::deuler_rodrigues() const
{
  auto rr = norm_sq();
  auto I = R2::identity(options());
  auto E = R3::levi_civita(options());
  auto W = R2::skew(*this);

  return 8.0 * (rr - 3.0) / math::pow(1.0 + rr, 3.0) * R3(torch::einsum("...ij,...k", {W, *this})) -
         32.0 / math::pow(1 + rr, 3.0) * R3(torch::einsum("...ij,...k", {(W * W), *this})) -
         4.0 * (1 - rr) / math::pow(1.0 + rr, 2.0) * R3(torch::einsum("...kij->...ijk", {E})) -
         8.0 / math::pow(1.0 + rr, 2.0) *
             R3(torch::einsum("...kim,...mj", {E, W}) + torch::einsum("...im,...kmj", {W, E}));
}

Rot
Rot::rotate(const Rot & r) const
{
  return r * (*this);
}

R2
Rot::drotate(const Rot & r) const
{
  auto r1 = *this;
  auto r2 = r;

  auto rr1 = r1.norm_sq();
  auto rr2 = r2.norm_sq();
  auto d = 1.0 + rr1 * rr2 - 2 * r1.dot(r2);
  auto r3 = this->rotate(r);
  auto I = R2::identity(options());

  return 1.0 / d *
         (-r3.outer(2 * rr1 * r2 - 2.0 * r1) - 2 * r1.outer(r2) + (1 - rr1) * I - 2 * R2::skew(r1));
}

R2
Rot::drotate_self(const Rot & r) const
{
  auto r1 = r;
  auto r2 = *this;

  auto rr1 = r1.norm_sq();
  auto rr2 = r2.norm_sq();
  auto d = 1.0 + rr1 * rr2 - 2 * r1.dot(r2);
  auto r3 = this->rotate(r);
  auto I = R2::identity(options());

  return 1.0 / d *
         (-r3.outer(2 * rr1 * r2 - 2.0 * r1) - 2 * r1.outer(r2) + (1 - rr1) * I + 2 * R2::skew(r1));
}

Rot
Rot::shadow() const
{
  return -*this / this->norm_sq();
}

R2
Rot::dshadow() const
{
  auto ns = this->norm_sq();

  return (2.0 / ns * this->outer(*this) - R2::identity(options())) / ns;
}

Rot
operator*(const Rot & r1, const Rot & r2)
{
  auto rr1 = r1.norm_sq();
  auto rr2 = r2.norm_sq();

  return ((1 - rr2) * r1 + (1.0 - rr1) * r2 - 2.0 * r2.cross(r1)) /
         (1.0 + rr1 * rr2 - 2 * r1.dot(r2));
}

} // namemspace neml2
