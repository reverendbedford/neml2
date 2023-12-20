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

namespace neml2
{
Rot::Rot(const Vec & v)
  : Rot(BatchTensor(v))
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
  // We use the dot product twice
  auto rr = norm_sq();

  return ((1.0 - rr) * R2::identity(options()) + 2.0 * outer(*this) -
          2.0 * R3::levi_civita(options()).contract_k(*this)) /
         (1.0 + rr);
}

R3
Rot::deuler_rodrigues() const
{
  auto rr = norm_sq();
  auto R = euler_rodrigues();
  auto I = R2::identity(options());
  neml_assert_batch_broadcastable_dbg(*this, rr, R, I);

  return 2.0 / (1.0 + rr) *
         (R3(-torch::einsum("...ij,...m", {R, *this}) - torch::einsum("...ij,...m", {I, *this}) +
                 torch::einsum("...im,...j", {I, *this}) + torch::einsum("...jm,...i", {I, *this}),
             batch_dim()) -
          R3::levi_civita(options()));
}

Rot
Rot::rotate(const Rot & r) const
{
  return r * (*this);
}

R2
Rot::drotate(const Rot & r) const
{
  return 1.0 / (1.0 - r.dot(*this)) *
         (R2::identity(options()) - R2::skew(*this) + rotate(r).outer(*this));
}

R2
Rot::drotate_self(const Rot & r) const
{
  return 1.0 / (1.0 - r.dot(*this)) * (R2::identity(options()) + R2::skew(r) + rotate(r).outer(r));
}

Rot
operator*(const Rot & r1, const Rot & r2)
{
  return (r1 + r2 + r1.cross(r2)) / (1.0 - r1.dot(r2));
}

} // namemspace neml2
