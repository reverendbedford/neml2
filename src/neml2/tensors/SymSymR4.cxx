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

#include "neml2/tensors/SymSymR4.h"
#include "neml2/misc/utils.h"

namespace neml2
{
SymSymR4
SymSymR4::init(SymSymR4::FillMethod method, const std::vector<Scalar> & vals)
{
  switch (method)
  {
    case SymSymR4::FillMethod::zero:
      return SymSymR4::init_zero();
    case SymSymR4::FillMethod::identity_sym:
      return SymSymR4::init_identity_sym();
    case SymSymR4::FillMethod::identity_vol:
      return SymSymR4::init_identity() / 3;
    case SymSymR4::FillMethod::identity_dev:
      return SymSymR4::init_identity_sym() - SymSymR4::init_identity() / 3;
    case SymSymR4::FillMethod::isotropic_E_nu:
      return SymSymR4::init_isotropic_E_nu(vals[0], vals[1]);
    default:
      std::runtime_error("Unsupported fill method");
      return SymSymR4();
  }
}

SymSymR4
SymSymR4::init_identity()
{
  return SymSymR4(torch::tensor({{1, 1, 1, 0, 0, 0},
                                 {1, 1, 1, 0, 0, 0},
                                 {1, 1, 1, 0, 0, 0},
                                 {0, 0, 0, 0, 0, 0},
                                 {0, 0, 0, 0, 0, 0},
                                 {0, 0, 0, 0, 0, 0}},
                                TorchDefaults),
                  1);
}

SymSymR4
SymSymR4::init_zero()
{
  return torch::zeros({6, 6}, TorchDefaults);
}

SymSymR4
SymSymR4::init_identity_sym()
{
  return SymSymR4(torch::eye(6), 1);
}

SymSymR4
SymSymR4::init_isotropic_E_nu(const Scalar & E, const Scalar & nu)
{
  SymSymR4 C;
  C = C.batch_expand_copy(E.batch_sizes());

  Scalar pf = E / ((1 + nu) * (1 - 2 * nu));
  Scalar C1 = (1 - nu) * pf;
  Scalar C2 = nu * pf;
  Scalar C4 = (1 - 2 * nu) * pf;

  for (TorchSize i = 0; i < 3; i++)
  {
    for (TorchSize j = 0; j < 3; j++)
    {
      if (i == j)
        C.base_index_put({i, j}, C1.squeeze(-1));
      else
        C.base_index_put({i, j}, C2.squeeze(-1));
    }
  }

  for (TorchSize i = 3; i < 6; i++)
    C.base_index_put({i, i}, C4.squeeze(-1));

  return C;
}

SymSymR4
SymSymR4::init_R4(const R4 & T)
{
  SymSymR4 C;
  C = C.batch_expand_copy(T.batch_sizes());

  for (TorchSize a = 0; a < 6; a++)
  {
    for (TorchSize b = 0; b < 6; b++)
    {
      auto ij = utils::mandel_index[a];
      auto kl = utils::mandel_index[b];
      auto ij_f = utils::mandelFactor(a);
      auto kl_f = utils::mandelFactor(b);

      C.base_index_put({a, b},
                       ij_f * kl_f * 0.25 *
                           (T(ij[0], ij[1], kl[0], kl[1]) + T(ij[1], ij[0], kl[0], kl[1]) +
                            T(ij[0], ij[1], kl[1], kl[0]) + T(ij[1], ij[0], kl[1], kl[0])));
    }
  }

  return C;
}

R4
SymSymR4::to_full() const
{
  return R4::init(*this);
}

Scalar
SymSymR4::operator()(TorchSize i, TorchSize j, TorchSize k, TorchSize l) const
{
  TorchSize a = utils::mandel_reverse_index[i][j];
  TorchSize b = utils::mandel_reverse_index[k][l];

  return base_index({a, b}).unsqueeze(-1) / (utils::mandelFactor(a) * utils::mandelFactor(b));
}

SymSymR4
SymSymR4::operator-() const
{
  return -torch::Tensor(*this);
}

SymSymR4
SymSymR4::inverse() const
{
  return torch::linalg::inv(*this);
}

SymSymR4
operator+(const SymSymR4 & a, const SymSymR4 & b)
{
  return torch::operator+(a, b);
}

SymSymR4
operator-(const SymSymR4 & a, const SymSymR4 & b)
{
  return torch::operator-(a, b);
}

SymSymR4
operator*(const SymSymR4 & a, const Scalar & b)
{
  return torch::operator*(a, b.unsqueeze(-1));
}

SymSymR4
operator*(const Scalar & a, const SymSymR4 & b)
{
  return b * a;
}

SymR2
operator*(const SymSymR4 & a, const SymR2 & b)
{
  return torch::matmul(a, b.unsqueeze(2)).squeeze(2);
}

SymR2
operator*(const SymR2 & a, const SymSymR4 & b)
{
  return torch::matmul(a.unsqueeze(2), b).squeeze(2);
}

SymSymR4
operator*(const SymSymR4 & a, const SymSymR4 & b)
{
  return torch::matmul(a, b);
}

SymSymR4
operator/(const SymSymR4 & a, const Scalar & b)
{
  return torch::operator/(a, b.unsqueeze(-1));
}
} // namespace neml2
