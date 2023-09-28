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

#include "neml2/misc/math.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/Rot.h"

namespace neml2
{
R2::R2(const SR2 & S)
  : R2(math::mandel_to_full(S))
{
}

R2::R2(const Rot & r)
  : R2(r.euler_rodrigues())
{
}

R2
R2::fill(const Real & a, const torch::TensorOptions & options)
{
  return R2::fill(Scalar(a, options));
}

R2
R2::fill(const Scalar & a)
{
  auto zero = torch::zeros_like(a);
  return R2(torch::stack({torch::stack({a, zero, zero}, -1),
                          torch::stack({zero, a, zero}, -1),
                          torch::stack({zero, zero, a}, -1)},
                         -2),
            a.batch_dim());
}

R2
R2::fill(const Real & a11, const Real & a22, const Real & a33, const torch::TensorOptions & options)
{
  return R2::fill(Scalar(a11, options), Scalar(a22, options), Scalar(a33, options));
}

R2
R2::fill(const Scalar & a11, const Scalar & a22, const Scalar & a33)
{
  auto zero = torch::zeros_like(a11);
  return R2(torch::stack({torch::stack({a11, zero, zero}, -1),
                          torch::stack({zero, a22, zero}, -1),
                          torch::stack({zero, zero, a33}, -1)},
                         -2),
            a11.batch_dim());
}

R2
R2::fill(const Real & a11,
         const Real & a22,
         const Real & a33,
         const Real & a23,
         const Real & a13,
         const Real & a12,
         const torch::TensorOptions & options)
{
  return R2::fill(Scalar(a11, options),
                  Scalar(a22, options),
                  Scalar(a33, options),
                  Scalar(a23, options),
                  Scalar(a13, options),
                  Scalar(a12, options));
}

R2
R2::fill(const Scalar & a11,
         const Scalar & a22,
         const Scalar & a33,
         const Scalar & a23,
         const Scalar & a13,
         const Scalar & a12)
{
  return R2(torch::stack({torch::stack({a11, a12, a13}, -1),
                          torch::stack({a12, a22, a23}, -1),
                          torch::stack({a13, a23, a33}, -1)},
                         -2),
            a11.batch_dim());
}

R2
R2::fill(const Real & a11,
         const Real & a12,
         const Real & a13,
         const Real & a21,
         const Real & a22,
         const Real & a23,
         const Real & a31,
         const Real & a32,
         const Real & a33,
         const torch::TensorOptions & options)
{
  return R2::fill(Scalar(a11, options),
                  Scalar(a12, options),
                  Scalar(a13, options),
                  Scalar(a21, options),
                  Scalar(a22, options),
                  Scalar(a23, options),
                  Scalar(a31, options),
                  Scalar(a32, options),
                  Scalar(a33, options));
}

R2
R2::fill(const Scalar & a11,
         const Scalar & a12,
         const Scalar & a13,
         const Scalar & a21,
         const Scalar & a22,
         const Scalar & a23,
         const Scalar & a31,
         const Scalar & a32,
         const Scalar & a33)
{
  return R2(torch::stack({torch::stack({a11, a12, a13}, -1),
                          torch::stack({a21, a22, a23}, -1),
                          torch::stack({a31, a32, a33}, -1)},
                         -2),
            a11.batch_dim());
}

R2
R2::skew(const Vec & v)
{
  const auto z = torch::zeros_like(v(0));
  return R2(torch::stack({torch::stack({z, -v(2), v(1)}, -1),
                          torch::stack({v(2), z, -v(0)}, -1),
                          torch::stack({-v(1), v(0), z}, -1)},
                         -2),
            v.batch_dim());
}

R2
R2::identity(const torch::TensorOptions & options)
{
  return R2(torch::eye(3, options), 0);
}

R2
R2::rotate(const Rot & r) const
{
  R2 R = r.euler_rodrigues();
  return R * (*this) * R.transpose();
}

R3
R2::drotate(const Rot & r) const
{
  R2 R = r.euler_rodrigues();
  R3 F = r.deuler_rodrigues();

  return R3(torch::einsum("...itl,...tm,...jm", {F, *this, R}) +
                torch::einsum("...ik,...kt,...jtl", {R, *this, F}),
            broadcast_batch_dim(*this, F, R));
}

Scalar
R2::operator()(TorchSize i, TorchSize j) const
{
  return base_index({i, j});
}

R2
R2::inverse() const
{
  return torch::Tensor::inverse();
}

R2
R2::transpose() const
{
  return BatchTensorBase<R2>::base_transpose(0, 1);
}

R2
operator*(const R2 & A, const R2 & B)
{
  neml_assert_broadcastable_dbg(A, B);
  return R2(torch::einsum("...ik,...kj", {A, B}), broadcast_batch_dim(A, B));
}

Vec
operator*(const R2 & A, const Vec & b)
{
  neml_assert_batch_broadcastable_dbg(A, b);
  return Vec(torch::einsum("...ik,...k", {A, b}), broadcast_batch_dim(A, b));
}
} // namespace neml2
