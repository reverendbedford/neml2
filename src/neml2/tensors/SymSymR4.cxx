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

namespace neml2
{
SymSymR4
SymSymR4::init_identity(const torch::TensorOptions & options)
{
  return torch::tensor({{{1, 1, 1, 0, 0, 0},
                         {1, 1, 1, 0, 0, 0},
                         {1, 1, 1, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0}}},
                       options);
}

SymSymR4
SymSymR4::init_identity_sym(const torch::TensorOptions & options)
{
  return torch::eye(6, options).unsqueeze(0);
}

SymSymR4
SymSymR4::init_identity_vol(const torch::TensorOptions & options)
{
  return SymSymR4::init_identity(options) / 3;
}

SymSymR4
SymSymR4::init_identity_dev(const torch::TensorOptions & options)
{
  return SymSymR4::init_identity_sym(options) - SymSymR4::init_identity(options) / 3;
}

SymSymR4
SymSymR4::init_isotropic_E_nu(const Scalar & E, const Scalar & nu)
{
  const Scalar zero = torch::zeros_like(E);
  const Scalar pf = E / ((1 + nu) * (1 - 2 * nu));
  const Scalar C1 = (1 - nu) * pf;
  const Scalar C2 = nu * pf;
  const Scalar C4 = (1 - 2 * nu) * pf;

  return torch::stack({torch::cat({C1, C2, C2, zero, zero, zero}, -1),
                       torch::cat({C2, C1, C2, zero, zero, zero}, -1),
                       torch::cat({C2, C2, C1, zero, zero, zero}, -1),
                       torch::cat({zero, zero, zero, C4, zero, zero}, -1),
                       torch::cat({zero, zero, zero, zero, C4, zero}, -1),
                       torch::cat({zero, zero, zero, zero, zero, C4}, -1)},
                      -1);
}

SymSymR4
SymSymR4::init_isotropic_E_nu(const Real & E, const Real & nu, const torch::TensorOptions & options)
{
  return SymSymR4::init_isotropic_E_nu(Scalar(E, options), Scalar(nu, options));
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
operator+(const SymSymR4 & a, const Real & b)
{
  return torch::operator+(a, b);
}

SymSymR4
operator+(const Real & a, const SymSymR4 & b)
{
  return torch::operator+(a, b);
}

SymSymR4
operator+(const SymSymR4 & a, const SymSymR4 & b)
{
  return torch::operator+(a, b);
}

SymSymR4
operator-(const SymSymR4 & a, const Real & b)
{
  return torch::operator+(a, b);
}

SymSymR4
operator-(const Real & a, const SymSymR4 & b)
{
  return torch::operator+(a, b);
}

SymSymR4
operator-(const SymSymR4 & a, const SymSymR4 & b)
{
  return torch::operator-(a, b);
}

SymSymR4
operator*(const SymSymR4 & a, const Real & b)
{
  return torch::operator*(a, b);
}

SymSymR4
operator*(const Real & a, const SymSymR4 & b)
{
  return torch::operator*(a, b);
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
operator/(const SymSymR4 & a, const Real & b)
{
  return torch::operator/(a, b);
}

SymSymR4
operator/(const Real & a, const SymSymR4 & b)
{
  return torch::operator/(a, b);
}

SymSymR4
operator/(const SymSymR4 & a, const Scalar & b)
{
  return torch::operator/(a, b.unsqueeze(-1));
}
} // namespace neml2
